#!/usr/bin/env python

from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
import pandas as pd
import json
import time
import math
import click
import torch
import random
import numpy as np
from loguru import logger
import uuid
import sys
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torchvision.transforms import ToTensor
from typing import Any, Dict, List, Optional

from helpers import initialize_from_config, make_dirs, get_obj_from_str
from src.data.tokenizer import Char_Tokenizer
from src.data.constant import CHAR_WIDTH, MAX_LENGTH, FIXED_HEIGHT
import cv2


# Dataset configuration
DATASET_CONFIGS = {
    "IAM": {
        "config_path": "configs/constant_IAM.yaml",
        "ckpt_path": "ckpt/IAM/ckpt.pth",
        "sample_path": "assets/sample/IAM",
        "description": "IAM dataset",
        "preprocess_strategy": "variable",  # "variable" or "fixed"
        "fixed_size": 352,  # Only used if strategy is "fixed"
    },
    "IMGUR5K": {
        "config_path": "configs/constant_IMGUR5K.yaml",
        "ckpt_path": "ckpt/IMGUR5K/ckpt.pth",
        "sample_path": "assets/sample/IMGUR5K",
        "description": "IMGUR5K dataset",
        "preprocess_strategy": "variable",
        "fixed_size": 352,
    },
    "IIT_English_Words": {
        "config_path": "configs/constant_IIIT.yaml",
        "ckpt_path": "ckpt/IIIT/ckpt.pth",
        "sample_path": "assets/sample/IIIT_English",
        "description": "IIIT English Word dataset",
        "preprocess_strategy": "fixed",
        "fixed_size": 256,
    },
    "CASIA_HWDB": {
        "config_path": "configs/constant_CASIA.yaml",
        "ckpt_path": "ckpt/CASIA/ckpt.pth",
        "sample_path": "assets/sample/CASIA_HWDB",
        "description": "CASIA_HWDB dataset",
        "preprocess_strategy": "variable",
        "fixed_size": 352,
    },
    "ViHTGen": {
        "config_path": "configs/constant_ViHTGen.yaml",
        "ckpt_path": "ckpt/ViHTGen/ckpt.pth",
        "sample_path": "assets/sample/ViHTGen",
        "description": "ViHTGen dataset",
        "preprocess_strategy": "variable",
        "fixed_size": 352,
    },
}


class TextDataset(Dataset):
    """Dataset for text generation."""

    def __init__(self, tokenizer, texts: List[str]):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        word_embedding = self.tokenizer.encode(self.texts[idx])
        word_embedding = np.array(word_embedding, dtype="int64")
        label = torch.from_numpy(word_embedding).long()
        return {"text_embedding": label, "raw_text": self.texts[idx]}

    def collate_function(self, batch):
        text_embeddings = [item["text_embedding"] for item in batch]
        max_length = max([embedding.shape[0] for embedding in text_embeddings])
        batch_text_embeddings = (
            np.ones((len(text_embeddings), max_length), dtype="int64") * self.tokenizer.special_chars["PAD_TOKEN"]
        )
        original_lens = []
        for idx, embed in enumerate(text_embeddings):
            embed = embed.numpy()
            batch_text_embeddings[idx, : len(embed)] = embed
            original_lens.append(int(embed.shape[0] * CHAR_WIDTH))
        batch_text_embeddings = torch.from_numpy(batch_text_embeddings).long()
        return {
            "text_embeddings": batch_text_embeddings,
            "raw_texts": [item["raw_text"] for item in batch],
            "origin_lens": original_lens,
        }


def test_writer(
    imgs,
    image_dir,
    tokenizer,
    pipeline,
    text_corpus: List[str],
    save_dir: Path,
    device: str,
) -> Dict:
    """Generate handwriting for a specific writer."""
    ds = TextDataset(tokenizer, text_corpus)
    loader = DataLoader(
        ds,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=ds.collate_function,
    )

    wid_logging: Dict[str, List] = {"text": [], "ref_img": [], "gen_name": []}

    with torch.no_grad():
        for batch in loader:
            batch_label, batch_text, batch_ori_lens = (
                batch["text_embeddings"],
                batch["raw_texts"],
                batch["origin_lens"],
            )
            batch_label = batch_label.to(device)

            img_names = np.random.choice(imgs, size=min(len(imgs), len(batch_label)), replace=True)
            ref_imgs = [Image.open(Path(image_dir).joinpath(img_name)).convert("RGB") for img_name in img_names]
            batch_img_list = [ToTensor()(img).unsqueeze(0) for img in ref_imgs]
            batch_img = torch.cat(batch_img_list, dim=0).to(device)

            gen_imgs = pipeline.sampling(batch_ori_lens, batch_label, batch_img, log_progress=False)

            for text, gen_img, ref_name in zip(batch_text, gen_imgs, img_names):
                gen_img = gen_img * 255.0
                gen_img = gen_img.permute(2, 3, 1).detach().cpu().numpy().astype("uint8")
                gen_img = Image.fromarray(gen_img, mode="RGB")
                save_name = str(uuid.uuid4()) + ".png"
                gen_img.save(save_dir.joinpath(save_name))

                wid_logging["text"].append(text)
                wid_logging["gen_name"].append(save_name)
                wid_logging["ref_img"].append(ref_name)

    return wid_logging


def setup_logging(datetime_dir: Path):
    """Configure loguru logger."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        datetime_dir / "generation.log",
        rotation="500 MB",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """CONSTANT Handwriting Generation Tool

    Generate OCR corpus using CONSTANT model with various handwriting styles.
    """
    pass


def preprocess_image(image_pil, dataset_name, config, strategy="variable", fixed_size=352):
    """
    Preprocess image with configurable strategy.

    Args:
        image_pil: PIL Image to preprocess
        dataset_name: Dataset to use configuration from
        config: Dataset configuration (OmegaConf)
        strategy: preprocess image strategy (variable or fixed)
        fixed_size: Fixed image width if strategy = fixed
    Returns:
        Preprocessed tensor image in shape (1, C, H, W)
    """

    image_np = np.array(image_pil.convert("RGB"))
    h, w = image_np.shape[:2]

    # Resize to fixed height while maintaining aspect ratio
    new_width = int(w * (FIXED_HEIGHT / h))
    image = cv2.resize(image_np, (new_width, FIXED_HEIGHT), interpolation=cv2.INTER_LINEAR)
    logger.debug(f"Image shape after raw resize: {image.shape}")

    # Pad to multiple of CHAR_WIDTH
    pad_pixels = (CHAR_WIDTH - (new_width % CHAR_WIDTH)) % CHAR_WIDTH
    new_width += pad_pixels

    logger.debug(f"Resizing image so that width is divisible by CHAR_WIDTH ({CHAR_WIDTH})")
    image = cv2.resize(image, (new_width, FIXED_HEIGHT), interpolation=cv2.INTER_LINEAR)

    logger.debug(f"Reference image after height resize: H={image.shape[0]}, W={new_width}")

    if strategy == "fixed":
        if new_width > fixed_size:
            # If larger, resize down to fixed_size
            image = cv2.resize(image, (fixed_size, FIXED_HEIGHT), interpolation=cv2.INTER_LINEAR)
            logger.debug(f"Fixed strategy: Resized from {new_width} to {fixed_size}")

        elif new_width < fixed_size:
            pad_total = fixed_size - new_width
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)
            logger.debug(
                f"Fixed strategy: Padded from {new_width} to {fixed_size} (left={pad_left}, right={pad_right})"
            )

        else:
            logger.debug(f"Fixed strategy: Width already matches {fixed_size}")

    elif strategy == "variable":
        # Variable strategy: Cap width at max_width (from config) but don't force resize smaller images
        max_width = config.MAX_WIDTH
        if new_width > max_width:
            image = cv2.resize(image, (max_width, FIXED_HEIGHT), interpolation=cv2.INTER_LINEAR)
            logger.debug(f"Variable strategy: Resized from {new_width} to {max_width}")

        else:
            logger.debug("Variable strategy: Perform right-pad to max width")
            image = cv2.copyMakeBorder(
                image,
                0,
                0,
                0,
                int(max_width - new_width),
                cv2.BORDER_CONSTANT,
                value=255,
            )

    else:
        raise ValueError(f"Unknown preprocessing strategy: {strategy}")

    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0).float()
    image = image.permute(0, 3, 1, 2)

    logger.debug(f"Final tensor shape: {image.shape}")

    return image


def test_writer_single_style(
    style_img_path: Path,
    tokenizer,
    pipeline,
    config,
    text_corpus: List[str],
    save_dir: Path,
    device: str,
    dataset_name: str,
) -> Dict:
    """Generate handwriting for a single reference style image."""
    ds = TextDataset(tokenizer, text_corpus)
    loader = DataLoader(
        ds,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=ds.collate_function,
    )

    wid_logging: Dict[str, List] = {"text": [], "ref_img": [], "gen_name": []}

    dataset_config = DATASET_CONFIGS.get(dataset_name, {})
    strategy = dataset_config.get("preprocess_strategy", "variable")
    fixed_size = dataset_config.get("fixed_size", 352)

    style_img = Image.open(style_img_path).convert("RGB")
    style_tensor = preprocess_image(style_img, dataset_name, config, strategy, fixed_size)

    logger.info(f"Preprocessed reference image with strategy '{strategy}': shape={style_tensor.shape}")

    with torch.no_grad():
        for batch in loader:
            batch_label, batch_text, batch_ori_lens = (
                batch["text_embeddings"],
                batch["raw_texts"],
                batch["origin_lens"],
            )
            batch_label = batch_label.to(device)

            if strategy == "variable":
                input_shape = (
                    pipeline._diffusion.img_size,
                    int(batch_label.shape[1] * CHAR_WIDTH // pipeline._diffusion.down_sample),
                )
            else:
                input_shape = (
                    pipeline._diffusion.img_size,
                    int(fixed_size // pipeline._diffusion.down_sample),
                )

            # Replicate the style image for the entire batch
            batch_img = style_tensor.repeat(len(batch_label), 1, 1, 1).to(device)

            gen_imgs = pipeline.sampling(
                input_shape,
                batch_ori_lens if strategy == "variable" else None,
                batch_label,
                batch_img,
                log_progress=False,
            )
            if isinstance(gen_imgs, tuple):
                gen_imgs = list(gen_imgs)

            for text, gen_img in zip(batch_text, gen_imgs):
                gen_img = gen_img * 255.0
                gen_img = gen_img.permute(1, 2, 0).detach().cpu().numpy().astype("uint8")
                gen_img = Image.fromarray(gen_img, mode="RGB")
                save_name = str(uuid.uuid4()) + ".png"
                gen_img.save(save_dir.joinpath(save_name))

                wid_logging["text"].append(text)
                wid_logging["gen_name"].append(save_name)
                wid_logging["ref_img"].append(str(style_img_path.name))

    return wid_logging


@cli.command()
@click.argument("text_corpus", type=click.Path(exists=True), required=True)
@click.option(
    "--template-dataset",
    "-d",
    type=click.Choice(list(DATASET_CONFIGS.keys()), case_sensitive=True),
    default="IAM",
    help="Dataset to use as reference for generation (default: IAM)",
)
@click.option(
    "--save-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="output/ocr_synthesize_data",
    help="Directory to save generated images (default: output/generation)",
)
@click.option(
    "--num-styles",
    "-n",
    type=int,
    default=None,
    help="Number of reference styles to use. Default: use all available",
)
@click.option(
    "--total-sample",
    "-s",
    type=int,
    default=1000000,
    help="Total number of samples to generate (default: 1000000)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"], case_sensitive=False),
    default="cuda",
    help="Inference device (default: cuda)",
)
@click.option("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")
def generate_multi_style(
    text_corpus: str,
    template_dataset: str,
    save_dir: Path,
    num_styles: Optional[int],
    total_sample: int,
    device: str,
    seed: int,
):
    """Generate OCR corpus with specified parameters using multiple styles randomly selected from dataset.

    TEXT_CORPUS: Path to text file containing text to generate (one line per text)

    Example usage:
        constant-gen generate-multi-style corpus.txt --template-dataset IAM --num-styles 50 --total-sample 10000
    """
    random.seed(seed)
    np.random.seed(0)

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    datetime_dir = Path(save_dir) / datetime_str
    make_dirs(datetime_dir)

    save_img_dir = datetime_dir / "images"
    make_dirs(save_img_dir)

    setup_logging(datetime_dir)
    logger.info(f"Starting OCR corpus generation at {datetime_str}")
    logger.info(f"Output directory: {datetime_dir}")
    logger.info(f"Text corpus: {text_corpus}")
    logger.info(f"Template dataset: {template_dataset}")
    logger.info(f"Device: {device}")
    if num_styles:
        logger.info(f"Number of styles: {num_styles}")

    start_time = time.time()
    process_log: Dict[str, Any] = {
        "datetime": datetime_str,
        "text_corpus_path": text_corpus,
        "template_dataset": template_dataset,
        "num_styles": num_styles,
        "total_samples": int(total_sample),
        "device": device,
        "random_seed": seed,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "total_duration_seconds": None,
        "writers_processing": [],
    }

    try:
        reference_theme = DATASET_CONFIGS.get(template_dataset, None)
        if reference_theme is None:
            raise click.ClickException(f"Unknown dataset: {template_dataset}")

        ckpt_path = str(reference_theme["ckpt_path"])
        if not Path(ckpt_path).exists():
            raise click.ClickException(f"Checkpoint not found: {ckpt_path}")

        logger.info(f"Loading pipeline from config: {reference_theme['config_path']}")
        config = OmegaConf.load(str(reference_theme["config_path"]))

        dataset = initialize_from_config(config.dataset)
        tokenizer = Char_Tokenizer(max_length=MAX_LENGTH, dset_name=dataset.dset_name)
        if not hasattr(config.text_encoder.params, "input_size"):
            config.text_encoder.params.input_size = tokenizer.vocab_size

        pipeline = get_obj_from_str(config.pipeline.target)(
            vae_config=config.vae.vae_pretrained_path,
            diffusion_config=config.diffusion,
            style_extractor_config=config.style_extractor,
            text_encoder_config=config.text_encoder,
        )
        pipeline.load_state_dict(ckpt_path)
        pipeline.eval()
        device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        logger.success(f"Pipeline loaded on {device}")

        reference_wids = None
        if hasattr(dataset, "writer_dict"):
            reference_wids = dataset.writer_dict

        sample_data_path = Path(str(reference_theme["sample_path"]))
        if not sample_data_path.exists() or not sample_data_path.is_dir():
            raise click.ClickException(f"Sample path not found: {sample_data_path}")

        available_wids = [path.stem for path in list(sample_data_path.glob("*"))]
        available_wids = list(set(available_wids) & set(list(reference_wids.keys())))

        if len(available_wids) == 0:
            raise click.ClickException("No available writers found")

        if num_styles is not None:
            if num_styles > len(available_wids):
                logger.warning(f"Requested {num_styles} styles but only {len(available_wids)} available")
                num_styles = len(available_wids)
            random.shuffle(available_wids)
            selected_wids = available_wids[:num_styles]
        else:
            selected_wids = available_wids

        reference_wids = {k: reference_wids[k] for k in selected_wids}
        logger.info(f"Selected {len(reference_wids)} writers for generation")

        group_data_dict = defaultdict(list)
        for wid in selected_wids:
            group_data_dict[wid] = [
                p.relative_to(sample_data_path.joinpath(wid)) for p in sample_data_path.joinpath(wid).rglob("*.*g")
            ]

        logger.info("Loading text corpus...")
        with open(text_corpus, "r") as _f:
            texts = [line.strip() for line in _f.readlines()]
            filter_texts = list(filter(lambda t: len(t) > MAX_LENGTH, texts))
            if len(filter_texts):
                logger.info(f"Ignore {len(set(texts) - set(filter_texts))} cases that exceed max length {MAX_LENGTH}")
            texts = filter_texts

        logger.info(f"Loaded {len(texts)} text samples from corpus")
        logger.info(f"Total samples requested: {total_sample}")

        numtext_per_wid = math.floor(total_sample / len(reference_wids.keys()))
        num_styles = len(reference_wids.keys())
        numtext_per_wids = {
            wid: numtext_per_wid if idx < num_styles else (total_sample - (num_styles - 1) * numtext_per_wid)
            for idx, wid in enumerate(reference_wids.keys(), start=1)
        }

        gen_logger: Dict[str, List] = {"style_id": [], "text": [], "ref_img": [], "gen_name": []}

        logger.info(f"Starting generation for {len(numtext_per_wids)} writers...")
        for wid_idx, (wid, numtext_per_wid) in enumerate(numtext_per_wids.items(), 1):
            wid_start_time = time.time()
            random.shuffle(texts)

            np_texts = np.array(texts)
            if len(np_texts) > numtext_per_wid:
                replace = False
            else:
                replace = True

            wid_texts = np.random.choice(np_texts, size=numtext_per_wid, replace=replace).tolist()

            logger.info(
                f"[{wid_idx}/{len(numtext_per_wids)}] Processing writer {wid} with {len(wid_texts)} text samples"
            )

            save_wid_dir = save_img_dir.joinpath(wid)
            make_dirs(save_wid_dir)

            imgs = group_data_dict[wid]
            wid_logging = test_writer(
                imgs,
                sample_data_path / wid,
                tokenizer,
                pipeline=pipeline,
                text_corpus=wid_texts,
                save_dir=save_wid_dir,
                device=device,
            )

            wid_duration = time.time() - wid_start_time
            wid_log = {
                "writer_id": wid,
                "text_samples_processed": len(wid_texts),
                "duration_seconds": wid_duration,
                "samples_per_second": len(wid_texts) / wid_duration if wid_duration > 0 else 0,
            }
            process_log["writers_processing"].append(wid_log)

            gen_logger["style_id"] += [wid] * len(wid_logging["text"])
            gen_logger["ref_img"] += wid_logging["ref_img"]
            gen_logger["text"] += wid_logging["text"]
            gen_logger["gen_name"] += wid_logging["gen_name"]

            logger.info(
                f"  Completed writer {wid} in {wid_duration:.2f}s ({wid_log['samples_per_second']:.2f} samples/s)"
            )

        gen_logger_df = pd.DataFrame.from_dict(gen_logger)
        gen_logger_df.to_csv(datetime_dir / "generation_log.csv", index=False)
        logger.success(f"Generation log saved to {datetime_dir / 'generation_log.csv'}")

        end_time = time.time()
        process_log["end_time"] = datetime.now().isoformat()
        process_log["total_duration_seconds"] = end_time - start_time
        process_log["actual_samples_generated"] = len(gen_logger_df)
        process_log["num_writers_processed"] = len(numtext_per_wids)

        if process_log["writers_processing"]:
            total_writer_samples = sum(w["text_samples_processed"] for w in process_log["writers_processing"])
            total_writer_time = sum(w["duration_seconds"] for w in process_log["writers_processing"])
            process_log["average_samples_per_second"] = (
                total_writer_samples / total_writer_time if total_writer_time > 0 else 0
            )

        with open(datetime_dir / "process_log.json", "w") as f:
            json.dump(process_log, f, indent=2)

        # Print summary
        logger.success(f"{'=' * 60}")
        logger.success("Generation completed successfully!")
        logger.success(f"{'=' * 60}")
        logger.success(
            f"Total duration: {process_log['total_duration_seconds']:.2f} seconds ({process_log['total_duration_seconds'] / 60:.2f} minutes)"
        )
        logger.success(f"Total samples generated: {process_log['actual_samples_generated']}")
        logger.success(f"Average throughput: {process_log.get('average_samples_per_second', 0):.2f} samples/second")
        logger.success(f"Writers processed: {process_log['num_writers_processed']}")
        logger.success(f"Results saved to: {datetime_dir}")
        logger.success(f"{'=' * 60}")
        logger.info(f"Logs saved to: {datetime_dir / 'generation.log'}")

    except Exception as e:
        logger.exception(f"Error during generation: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument("text_corpus", type=click.Path(exists=True), required=True)
@click.option(
    "--ref-image",
    "-r",
    type=click.Path(exists=True),
    required=True,
    help="Path to reference style image (required)",
)
@click.option(
    "--template-dataset",
    "-d",
    type=click.Choice(list(DATASET_CONFIGS.keys()), case_sensitive=True),
    default="IAM",
    help="Dataset that the reference image belongs to (default: IAM)",
)
@click.option(
    "--save-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="output/ocr_synthesize_data",
    help="Directory to save generated images (default: output/ocr_synthesize_data)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"], case_sensitive=False),
    default="cuda",
    help="Inference device (default: cuda)",
)
def generate_single_style(text_corpus: str, ref_image: str, template_dataset: str, save_dir: Path, device: str):
    """Generate OCR corpus using a single reference style image.

    TEXT_CORPUS: Path to text file containing text to generate (one line per text)

    REF_IMAGE: Path to the reference style image to use for generation

    Example usage:
        constant-gen generate-single-style corpus.txt --ref-image /path/to/style.png --template-dataset IAM
    """
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    datetime_dir = Path(save_dir) / datetime_str
    make_dirs(datetime_dir)

    save_img_dir = datetime_dir / "images"
    make_dirs(save_img_dir)

    setup_logging(datetime_dir)
    logger.info(f"Starting single-style OCR corpus generation at {datetime_str}")
    logger.info(f"Output directory: {datetime_dir}")
    logger.info(f"Text corpus: {text_corpus}")
    logger.info(f"Reference image: {ref_image}")
    logger.info(f"Template dataset: {template_dataset}")
    logger.info(f"Device: {device}")

    start_time = time.time()
    process_log: Dict[str, Any] = {
        "datetime": datetime_str,
        "text_corpus_path": text_corpus,
        "template_dataset": template_dataset,
        "ref_image_path": ref_image,
        "device": device,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "total_duration_seconds": None,
        "generation_mode": "single_style",
    }

    try:
        reference_theme = DATASET_CONFIGS.get(template_dataset, None)
        if reference_theme is None:
            raise click.ClickException(f"Unknown dataset: {template_dataset}")

        ckpt_path = str(reference_theme["ckpt_path"])
        if not Path(ckpt_path).exists():
            raise click.ClickException(f"Checkpoint not found: {ckpt_path}")

        logger.info(f"Loading pipeline from config: {reference_theme['config_path']}")
        config = OmegaConf.load(str(reference_theme["config_path"]))

        dataset = initialize_from_config(config.dataset)
        tokenizer = Char_Tokenizer(max_length=MAX_LENGTH, dset_name=getattr(dataset, "dset_name", "base"))
        if not hasattr(config.text_encoder.params, "input_size"):
            config.text_encoder.params.input_size = tokenizer.vocab_size

        pipeline = get_obj_from_str(config.pipeline.target)(
            vae_config=config.vae.pretrained_path,
            diffusion_config=config.diffusion,
            style_extractor_config=config.style_extractor,
            text_encoder_config=config.text_encoder,
        )
        pipeline.load_state_dict(ckpt_path)
        pipeline.eval()
        device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        logger.success(f"Pipeline loaded on {device}")

        ref_image_path = Path(ref_image)
        if not ref_image_path.exists():
            raise click.ClickException(f"Reference image not found: {ref_image_path}")

        try:
            test_img = Image.open(ref_image_path).convert("RGB")
            test_img.close()
            logger.info(f"Reference image validated: {ref_image_path}")
        except Exception as e:
            raise click.ClickException(f"Invalid reference image: {e}")

        logger.info("Loading text corpus...")
        with open(text_corpus, "r") as _f:
            texts = [line.strip() for line in _f.readlines()]
            filter_texts = list(filter(lambda t: len(t) <= MAX_LENGTH, texts))
            if len(filter_texts) < len(texts):
                ignored = len(texts) - len(filter_texts)
                logger.info(f"Ignored {ignored} texts that exceed max length {MAX_LENGTH}")
            texts = filter_texts

        if len(texts) == 0:
            raise click.ClickException("No valid texts found in corpus after filtering")

        logger.info(f"Loaded {len(texts)} text samples from corpus")

        logger.info("Starting generation using single reference style image...")
        generation_start = time.time()

        gen_logger: Dict[str, List] = {"text": [], "ref_img": [], "gen_name": []}

        wid_logging = test_writer_single_style(
            ref_image_path,
            tokenizer,
            pipeline,
            config,
            text_corpus=texts,
            save_dir=save_img_dir,
            device=device,
            dataset_name=template_dataset,
        )

        generation_duration = time.time() - generation_start

        gen_logger["text"] += wid_logging["text"]
        gen_logger["ref_img"] += wid_logging["ref_img"]
        gen_logger["gen_name"] += wid_logging["gen_name"]

        gen_logger_df = pd.DataFrame.from_dict(gen_logger)
        gen_logger_df.to_csv(datetime_dir / "generation_log.csv", index=False)
        logger.success(f"Generation log saved to {datetime_dir / 'generation_log.csv'}")

        end_time = time.time()
        process_log["end_time"] = datetime.now().isoformat()
        process_log["total_duration_seconds"] = end_time - start_time
        process_log["generation_duration_seconds"] = generation_duration
        process_log["actual_samples_generated"] = len(gen_logger_df)
        process_log["samples_per_second"] = len(gen_logger_df) / generation_duration if generation_duration > 0 else 0

        with open(datetime_dir / "process_log.json", "w") as f:
            json.dump(process_log, f, indent=2)

        logger.success(f"{'=' * 60}")
        logger.success("Generation completed successfully!")
        logger.success(f"{'=' * 60}")
        logger.success(
            f"Total duration: {process_log['total_duration_seconds']:.2f} seconds ({process_log['total_duration_seconds'] / 60:.2f} minutes)"
        )
        logger.success(f"Generation duration: {process_log['generation_duration_seconds']:.2f} seconds")
        logger.success(f"Total samples generated: {process_log['actual_samples_generated']}")
        logger.success(f"Average throughput: {process_log['samples_per_second']:.2f} samples/second")
        logger.success(f"Reference image: {ref_image_path.name}")
        logger.success(f"Results saved to: {datetime_dir}")
        logger.success(f"{'=' * 60}")
        logger.info(f"Logs saved to: {datetime_dir / 'generation.log'}")

    except Exception as e:
        logger.exception(f"Error during generation: {e}")
        raise click.ClickException(str(e))


@cli.command()
def list_datasets():
    """List available template datasets."""
    click.echo("\nAvailable template datasets:\n")
    for name, config in DATASET_CONFIGS.items():
        click.echo(f"  {name:20s} - {config['description']}")
        click.echo(f"                       Config: {config['config_path']}")
        click.echo(f"                       Checkpoint: {config['ckpt_path']}")
        click.echo(f"                       Samples: {config['sample_path']}")
        click.echo("")


if __name__ == "__main__":
    cli()
