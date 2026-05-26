from torch.utils.data import Dataset
from torchvision.transforms import functional as transfunc
import torch
import os
import random
from PIL import Image
from collections import defaultdict
import numpy as np
import json
from src.data import tokenizer
from src.data.augment import AUGMENTATION, STYLE_AUGMENTATION
from src.data.constant import CHAR_WIDTH


class BaseDataset(Dataset):
    def __init__(
        self,
        image_path,
        dset_name="base",
        max_length=None,
        padding=False,
        truncating=False,
        full_dict_path=None,
        writer_dict_path=None,
        random_style=False,
        transform_level="none",
        tokenizer_type="Char_Tokenizer",
    ):
        self.image_path = image_path
        self.max_length = max_length

        self.data_dict = json.load(open(full_dict_path, "r"))
        self.writer_dict = json.load(open(writer_dict_path, "r"))

        self.transforms = AUGMENTATION[transform_level]
        self.indices = list(self.data_dict.keys())
        self.num_writers = len(self.writer_dict.keys())
        self.tokenizer = getattr(tokenizer, tokenizer_type)(max_length, dset_name, padding, truncating)
        # Use for testloader in case of wordstylist, it will use writer_dict from train dataset
        self.random_style = random_style

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]["image"]
        label = self.data_dict[self.indices[idx]]["label"]
        if self.random_style:
            wr_id = random.choice(list(self.writer_dict.values()))
            wr_id = torch.tensor(wr_id).to(torch.int64)
            ori_wid = None
        else:
            wr_id = self.data_dict[self.indices[idx]]["s_id"]
            ori_wid = wr_id
            wr_id = torch.tensor(self.writer_dict[wr_id]).to(torch.int64)

        img_path = os.path.join(self.image_path, image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)

        word_embedding = self.tokenizer.encode(label)
        word_embedding = np.array(word_embedding, dtype="int64")
        word_embedding = torch.from_numpy(word_embedding).long()

        return {
            "images": image,
            "image_names": image_name,
            "raw_labels": label,
            "text_embeddings": word_embedding,
            "wids": wr_id,
            "ori_wids": ori_wid,
        }

    def collate_function(self, batch):
        images = [item["images"] for item in batch]
        images = torch.stack(images).float()
        image_names = [item["image_names"] for item in batch]
        raw_labels = [item["raw_labels"] for item in batch]
        text_embeddings = [item["text_embeddings"] for item in batch]
        wids = [item["wids"] for item in batch]
        wids = torch.stack(wids).to(torch.int64)
        ori_wids = [item["ori_wids"] for item in batch]

        max_length = max([embedding.shape[0] for embedding in text_embeddings])
        batch_text_embeddings = (
            np.ones((len(text_embeddings), max_length), dtype="int64") * self.tokenizer.special_chars["PAD_TOKEN"]
        )
        for idx, embed in enumerate(text_embeddings):
            embed = embed.numpy()
            batch_text_embeddings[idx, : len(embed)] = embed
        batch_text_embeddings = torch.from_numpy(batch_text_embeddings).long()

        return {
            "images": images,
            "image_names": image_names,
            "raw_labels": raw_labels,
            "text_embeddings": batch_text_embeddings,
            "wids": wids,
            "ori_wids": ori_wids,
        }


class DatasetVariableStyleReference(BaseDataset):
    def __init__(
        self,
        max_width,
        min_width,
        image_path,
        dset_name="base",
        max_length=None,
        padding=False,
        truncating=False,
        full_dict_path=None,
        writer_dict_path=None,
        random_style=False,
        transform_level="none",
        tokenizer_type="Char_Tokenizer",
        random_style_image=False,
        use_predefine_json=None,
    ):
        super().__init__(
            image_path,
            dset_name,
            max_length,
            padding,
            truncating,
            full_dict_path,
            writer_dict_path,
            random_style,
            transform_level,
            tokenizer_type,
        )
        self.style_transform = STYLE_AUGMENTATION[transform_level]
        self.random_style_image = random_style_image
        self.use_predefine_json = use_predefine_json
        self.max_width = max_width
        self.min_width = min_width
        if use_predefine_json is not None:
            self.style_mapping = json.load(open(use_predefine_json, "r"))

    def _groupbywid(self):
        group_wids = defaultdict(list)
        for item in self.data_dict.values():
            group_wids[item["s_id"]].append(item)
        return group_wids

    def _cluster_wid_indices(self):
        group_wids = defaultdict(list)
        for item_idx, item in self.data_dict.items():
            group_wids[item["s_id"]].append(int(item_idx))
        return group_wids

    def __getitem__(self, idx):
        info = super().__getitem__(idx)
        if self.random_style_image:
            if self.use_predefine_json is not None:
                anchor_name = info["image_names"]
                style_name = self.style_mapping[anchor_name]
            else:
                wid = info["ori_wids"]
                same_wid_names = self._groupbywid()[wid]
                anchor_name = info["image_names"]
                remain_names = [item["image"] for item in same_wid_names if item["image"] != anchor_name]
                if len(remain_names) > 1:
                    style_name = random.choice(remain_names)
                else:
                    style_name = anchor_name

            style_img_path = os.path.join(self.image_path, style_name)
            style_image = Image.open(style_img_path).convert("RGB")
            style_image = self.style_transform(style_image)
            info["style_names"] = style_name
        else:
            img_path = os.path.join(self.image_path, info["image_names"])
            image = Image.open(img_path).convert("RGB")
            style_image = self.style_transform(image)
        info["style_images"] = style_image

        return info

    def collate_function(self, batch):
        images = [item["images"] for item in batch]
        style_images = [item["style_images"] for item in batch]
        image_names = [item["image_names"] for item in batch]

        if self.random_style_image:
            style_names = [item["style_names"] for item in batch]

        raw_labels = [item["raw_labels"] for item in batch]
        text_embeddings = [item["text_embeddings"] for item in batch]
        wids = [item["wids"] for item in batch]
        wids = torch.stack(wids).to(torch.int64)
        ori_wids = [item["ori_wids"] for item in batch]

        max_length = max([embedding.shape[0] for embedding in text_embeddings])
        batch_text_embeddings = (
            np.ones((len(text_embeddings), max_length), dtype="int64") * self.tokenizer.special_chars["PAD_TOKEN"]
        )

        for idx, embed in enumerate(text_embeddings):
            embed = embed.numpy()
            batch_text_embeddings[idx, : len(embed)] = embed

        batch_text_embeddings = torch.from_numpy(batch_text_embeddings).long()

        max_batch_style_width = max([img.shape[2] for img in style_images])
        if max_batch_style_width > self.max_width:
            max_batch_style_width = self.max_width
        if max_batch_style_width < self.min_width:
            max_batch_style_width = self.min_width

        batch_style_images = torch.ones(
            (
                len(style_images),
                style_images[0].shape[0],
                style_images[0].shape[1],
                max_batch_style_width,
            ),
            dtype=torch.float32,
        )
        for idx, img in enumerate(style_images):
            if img.shape[2] <= max_batch_style_width:
                batch_style_images[idx, :, :, : img.shape[2]] = img
            else:
                batch_style_images[idx, :, :, : img.shape[2]] = transfunc.resize(
                    img, size=(img.shape[1], max_batch_style_width)
                )

        batch_anchor_images = torch.ones(
            (
                len(images),
                images[0].shape[0],
                images[0].shape[1],
                max_batch_style_width,
            ),
            dtype=torch.float32,
        )
        for idx, img in enumerate(images):
            if img.shape[2] <= max_batch_style_width:
                batch_anchor_images[idx, :, :, : img.shape[2]] = img
            else:
                batch_anchor_images[idx, :, :, : img.shape[2]] = transfunc.resize(
                    img, size=(img.shape[1], max_batch_style_width)
                )

        max_batch_width = max([img.shape[2] for img in images])
        if max_batch_width < self.min_width:
            max_batch_width = self.min_width

        batch_images = torch.ones(
            (len(images), images[0].shape[0], images[0].shape[1], max_batch_width),
            dtype=torch.float32,
        )
        original_lens = []
        for idx, (img, target) in enumerate(zip(images, text_embeddings)):
            if img.shape[2] <= max_batch_width:
                batch_images[idx, :, :, : img.shape[2]] = img
                original_lens.append(int(target.shape[0] * CHAR_WIDTH))
            else:
                batch_images[idx, :, :, : img.shape[2]] = img[:, :, :max_batch_width]
                original_lens.append(max_batch_width)

        batch = {
            "images": batch_images,
            "style_images": batch_style_images,
            "anchor_images": batch_anchor_images,
            "image_names": image_names,
            "raw_labels": raw_labels,
            "text_embeddings": batch_text_embeddings,
            "wids": wids,
            "ori_wids": ori_wids,
            "original_lens": original_lens,
        }
        if self.random_style_image:
            batch["style_names"] = style_names
        return batch
