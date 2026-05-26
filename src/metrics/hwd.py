import torch
import torch.nn as nn
from typing import Any
from PIL import Image
import numpy as np
import math
from torchvision import models
from torchvision.transforms import Compose, ToTensor, ColorJitter
import pickle
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


class AdjustDims(torch.nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        if H == 1 and W == 1:
            return x.reshape(B, C)
        return x.reshape(B, C, -1)


class CropWhite:
    def __call__(self, img) -> Any:
        img_width, img_height = img.size
        tmp = np.array(img.convert("L")).mean(0)
        img_width = img_width - (tmp >= 250)[::-1].argmin()
        img_width = min(img_width, img_height)
        img = img.crop((0, 0, img_width, img_height))
        return img


class ResizeHeight:
    def __init__(self, height, interpolation=Image.NEAREST):
        self.height = height
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        return img.resize((int(self.height * w / h), self.height), self.interpolation)


class PaddingMin:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        c, w, h = img.shape
        width = max(self.width, w)
        height = max(self.height, h)
        return torch.nn.functional.pad(img, (0, height - h, 0, width - w), mode="constant", value=0)


class CropStart:
    def __init__(self, width):
        self.width = width

    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, self.width, h))


class CropStartSquare:
    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, h, h))


class ResizeSquare:
    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)


class ToNumpy:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img
        elif isinstance(img, Image.Image):
            return np.array(img)
        elif isinstance(img, torch.Tensor):
            return img.numpy()
        else:
            raise TypeError(f"Unknown type: {type(img)}")


class Flatten:
    def __call__(self, img):
        return img.reshape(-1)


class ToInceptionV3Input:
    def __init__(self, size=299):
        self.size = size

    def __call__(self, x):
        h_rep = math.ceil(self.size / x.shape[1])
        w_rep = math.ceil(self.size / x.shape[2])
        return x.repeat(1, h_rep, w_rep)[:, : self.size, : self.size]


fid_ganwriting_transforms = Compose([CropStart(64), ResizeSquare(64), ToTensor()])


def fid_ganwriting_color_transforms(val=0.5):
    return Compose(
        [
            CropStart(64),
            ResizeSquare(64),
            ColorJitter(brightness=val, contrast=val, saturation=val, hue=val),
            ToTensor(),
        ]
    )


fid_our_transforms = Compose([CropStartSquare(), ResizeSquare(64), ToTensor()])

fid_whole_transforms = Compose([ResizeHeight(299), ToTensor()])


def fid_our_color_transforms(val=0.5):
    return Compose(
        [
            CropStartSquare(),
            ResizeSquare(64),
            ColorJitter(brightness=val, contrast=val, saturation=val, hue=val),
            ToTensor(),
        ]
    )


gs_transforms = Compose(
    [
        CropStart(64),
        # ResizeSquare(64),
        ToNumpy(),
        Flatten(),
    ]
)

fred_transforms = Compose([ResizeHeight(32), ToTensor()])

hwd_transforms = Compose(
    [
        ResizeHeight(32),
        ToTensor(),
        PaddingMin(32, 32),
    ]
)

fved_beginning_transforms = Compose(
    [
        ResizeHeight(32),
        CropStartSquare(),
        ToTensor(),
        PaddingMin(32, 32),
    ]
)

fred_64_transforms = Compose([ResizeHeight(64), ToTensor()])


def fred_color_transforms(val=0.5):
    return Compose(
        [
            ResizeHeight(32),
            ColorJitter(brightness=val, contrast=val, saturation=val, hue=val),
            ToTensor(),
        ]
    )


class ProcessedDataset:
    def __init__(self, ids, labels, features):
        self.ids = ids
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, labels):
        if not isinstance(labels, (list, tuple, set)):
            labels = (labels,)
        mask = torch.Tensor([label in labels for label in self.labels])
        mask = mask.to(torch.bool)
        ids = self.ids[mask]
        labels = [label for label, m in zip(self.labels, mask) if m]
        features = self.features[mask]
        return ProcessedDataset(ids, labels, features)

    def subset(self, n):
        unique_ids = torch.unique(self.ids)
        assert n <= len(unique_ids)
        mask = torch.randperm(len(unique_ids))[:n]
        mask = set(mask.tolist())
        mask = torch.Tensor([id.item() in mask for id in self.ids]).to(torch.bool)
        ids = self.ids[mask]
        labels = [label for label, m in zip(self.labels, mask) if m]
        features = self.features[mask]
        return ProcessedDataset(ids, labels, features)

    def split(self, ratio=0.5):
        ids = torch.unique(self.ids)
        ids = ids[torch.randperm(len(ids))]
        split = int(len(ids) * ratio)
        ids1 = set(ids[:split].tolist())
        ids2 = set(ids[split:].tolist())
        mask1 = torch.Tensor([id.item() in ids1 for id in self.ids]).to(torch.bool)
        mask2 = torch.Tensor([id.item() in ids2 for id in self.ids]).to(torch.bool)
        dataset1 = ProcessedDataset(
            self.ids[mask1],
            [label for label, m in zip(self.labels, mask1) if m],
            self.features[mask1],
        )
        dataset2 = ProcessedDataset(
            self.ids[mask2],
            [label for label, m in zip(self.labels, mask2) if m],
            self.features[mask2],
        )
        assert len(dataset1) + len(dataset2) == len(self), f"{len(dataset1)} + {len(dataset2)} != {len(self)}"
        return dataset1, dataset2

    def save(self, path):
        data = {
            "ids": self.ids.cpu().numpy(),
            "labels": self.labels,
            "features": self.features.cpu().numpy(),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        data["ids"] = torch.Tensor(data["ids"])
        data["features"] = torch.Tensor(data["features"])
        return ProcessedDataset(data["ids"], data["labels"], data["features"])

    @property
    def device(self):
        assert self.features.device == self.ids.device
        return self.features.device

    def to(self, device):
        self.ids = self.ids.to(device)
        self.features = self.features.to(device)
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def __add__(self, other):
        assert isinstance(other, ProcessedDataset)
        assert self.device == other.device
        ids = torch.cat((self.ids, other.ids))
        labels = self.labels + other.labels
        features = torch.cat((self.features, other.features))
        return ProcessedDataset(ids, labels, features)


class BaseBackbone(nn.Module):
    @torch.no_grad()
    def __call__(self, dataset):
        """
        Extract features from a dataset
        :param dataset: dataset to extract features from
        :return: dataset with extracted features
        """
        raise NotImplementedError


class VGG16Backbone(BaseBackbone):
    def __init__(self, url, batch_size=1):
        super().__init__()
        self.url = url
        self.batch_size = batch_size
        self.model = self.load_model()

    def load_model(self):
        model = models.vgg16(num_classes=10400)

        if self.url is not None:
            checkpoint = torch.load(self.url, map_location="cpu")
            model.load_state_dict(checkpoint)

        modules = list(model.features.children())
        modules.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        modules.append(AdjustDims())
        return torch.nn.Sequential(*modules)

    @torch.inference_mode()
    def get_activations(self, loader, verbose=False):
        self.model.eval()

        features, labels, ids = [], [], []
        for i, (images, authors, _) in enumerate(loader):
            images = images.to(next(self.model.parameters()).device)

            pred = self.model(images)
            pred = pred.squeeze(-2)

            pred = pred.unsqueeze(0) if pred.ndim == 1 else pred
            labels.append(authors)
            ids.append([i * loader.batch_size + d for d in range(len(authors))])
            features.append(pred.cpu())

            if verbose:
                print(f"\rComputing activations {i + 1}/{len(loader)}", end="", flush=True)
        if verbose:
            print(" OK")

        ids = torch.Tensor(sum(ids, [])).long()
        labels = sum(labels, [])
        features = torch.cat(features, dim=0)
        return ids, labels, features

    def collate_fn(self, batch):
        imgs_width = [x[0].size(2) for x in batch]
        max_width = max(imgs_width)
        imgs = torch.stack(
            [torch.nn.functional.pad(x[0], (0, max_width - x[0].size(2))) for x in batch],
            dim=0,
        )
        ids = [x[1] for x in batch]
        return imgs, ids, torch.Tensor(imgs_width)

    def __call__(self, dataset, verbose=False):
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        ids, labels, features = self.get_activations(loader, verbose)
        return ProcessedDataset(ids, labels, features)


class BaseDistance:
    def __call__(self, data1, data2) -> float:
        """
        Compute the distance between two datasets
        :param data1: first dataset
        :param data2: second dataset
        :return: the distance between the two datasets
        """
        raise NotImplementedError


class EuclideanDistance(BaseDistance):
    def __init__(self):
        super().__init__()

    def __call__(self, data1, data2, **kwargs):
        assert isinstance(data1, ProcessedDataset)
        assert isinstance(data2, ProcessedDataset)
        tmp_1 = data1.features.mean(dim=0).unsqueeze(0)
        tmp_2 = data2.features.mean(dim=0).unsqueeze(0)
        return torch.cdist(tmp_1, tmp_2).item()


class BaseScore(nn.Module):
    def __init__(self, backbone, distance, transforms, device="cpu"):
        super().__init__()
        self.backbone = backbone
        self.distance = distance
        self.transforms = transforms
        self.device = torch.device(device)
        self.to(self.device)

    def __call__(self, dataset1, dataset2, **kwargs) -> float:
        data1 = self.digest(dataset1, **kwargs)
        data2 = self.digest(dataset2, **kwargs)
        return self.distance(data1, data2)

    def digest(self, dataset, **kwargs) -> ProcessedDataset:
        dataset.transform = self.transforms
        return self.backbone(dataset, **kwargs)


class BaseDataset(Dataset):
    def __init__(self, path, transform=None, nameset=None, preprocess=None):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): List of authors to consider.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """
        if nameset is not None:
            raise NotImplementedError("Nameset is not implemented yet.")

        self.path = path
        self.imgs = []
        self.labels = []
        self.transform = transform
        self.preprocess = preprocess
        self.nameset = nameset
        self.is_sorted = False
        self.author_ids = []

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is index of the target class.
        """
        img = self.imgs[index]
        label = self.labels[index]
        img = Image.open(img).convert("RGB")
        if self.preprocess is not None:
            img = self.preprocess(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def sort(self, verbose=False):
        if self.is_sorted:
            return
        imgs_width = []
        for i, (img, label) in enumerate(self):
            imgs_width.append(img.size(2))
            if verbose:
                print(f"\rSorting {i + 1}/{len(self)} ", end="", flush=True)
        self.imgs = [x for _, x in sorted(zip(imgs_width, self.imgs), key=lambda pair: pair[0])]
        if verbose:
            print(" OK")
        self.is_sorted = True


class FolderDataset(BaseDataset):
    def __init__(self, path, **kwargs):
        super(FolderDataset, self).__init__(path, **kwargs)
        self.path = Path(path)
        self.imgs = list(Path(path).rglob("*.*g"))
        assert len(self.imgs) > 0, "No images found."
        self.labels = [img.parent.name for img in self.imgs]
        self.author_ids = sorted(set(self.labels))


VGG16_10400_URL = "../../pretrained/HWD/VGG16_class_10400.pth"


class HWDScore(BaseScore):
    def __init__(self, batchsize=1):
        backbone = VGG16Backbone(VGG16_10400_URL, batchsize)
        distance = EuclideanDistance()
        transforms = Compose(
            [
                ResizeHeight(32),
                ToTensor(),
                PaddingMin(32, 32),
            ]
        )
        super().__init__(backbone, distance, transforms)


def compute_hwd(src1, src2, batchsize):
    fake_dataset = FolderDataset(src1)
    real_dataset = FolderDataset(src2)
    score = HWDScore(batchsize).cuda()
    info = {}
    results = []
    assert fake_dataset.author_ids == real_dataset.author_ids
    fake_pd = score.digest(fake_dataset, verbose=True)
    real_pd = score.digest(real_dataset, verbose=True)
    for auth_id in fake_dataset.author_ids:
        res = score.distance(fake_pd[auth_id], real_pd[auth_id], verbose=True)
        results.append(res)
        info[auth_id] = res
    results = sum(results) / len(results)
    info["final"] = results
    return info
