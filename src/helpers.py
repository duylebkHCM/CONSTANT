import torch
import random
import importlib
import math
import PIL
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from typing import List, Union
import urllib.request
from huggingface_hub import snapshot_download
import os
import psutil


def cal_elasped_time(cur_time, start_time):
    elapsed_time = cur_time - start_time
    min, sec = divmod(elapsed_time, 60)
    if min > 60:
        hour, min = divmod(min, 60)
        time_ = f"{hour} h, {round(min, 0)} min, {round(sec, 0)} s"
    else:
        time_ = f"{round(min, 0)} min, {round(sec, 0)} s"
    return time_


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_obj_from_str(name: str, reload: bool = False):
    module, cls = name.rsplit(".", 1)

    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)


def initialize_from_config(config: OmegaConf) -> object:
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def count_parameters(model: torch.nn.Module):
    full_network = sum([p.numel() for p in model.parameters() if p.requires_grad is True])
    params = {"Full network": full_network}
    for module_name, module in model.named_children():
        params[module_name] = sum([p.numel() for p in module.parameters() if p.requires_grad is True])
    return params


def make_dirs(dir_paths):
    if not Path(dir_paths).exists():
        Path(dir_paths).mkdir(parents=True)


def make_image_grid(
    images: List[PIL.Image.Image],
    rows: int,
    cols: int,
    maxsize: int = None,
    resize: bool = True,
    padding: int = 0,
) -> PIL.Image.Image:
    """
    Prepares a grid of images. Handles images with different sizes.

    Args:
        images: List of PIL Images
        rows: Number of rows
        cols: Number of columns
        maxsize: Optional maximum dimension
        resize: If True, resize all to max size. If False, center smaller images.
        padding: Padding between images in pixels
    """
    assert len(images) == rows * cols

    max_w = max(img.size[0] for img in images)
    max_h = max(img.size[1] for img in images)

    if maxsize is not None:
        if max_h > max_w:
            h = min(max_h, maxsize)
            w = math.ceil(max_w * (h / max_h))
        else:
            w = min(max_w, maxsize)
            h = math.ceil(max_h * (w / max_w))
        max_w, max_h = w, h

    if resize:
        images = [img.resize((max_w, max_h)) for img in images]

    cell_width = max_w + padding
    cell_height = max_h + padding
    grid = Image.new("RGB", size=(cols * cell_width, rows * cell_height))

    for i, img in enumerate(images):
        x = (i % cols) * cell_width + (padding // 2)
        y = (i // cols) * cell_height + (padding // 2)

        if img.size < (max_w, max_h):
            cell_canvas = Image.new("RGB", (max_w, max_h), (0, 0, 0))
            paste_x = (max_w - img.size[0]) // 2
            paste_y = (max_h - img.size[1]) // 2
            cell_canvas.paste(img, (paste_x, paste_y))
            grid.paste(cell_canvas, (x, y))
        else:
            grid.paste(img, (x, y))

    return grid


def pil_text_img(im, text, pos, color=(255, 0, 0), textSize=25):
    """Taken from HiGAN/networks/utils.py"""
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype("assets/font/arial.ttf", textSize)
    fillColor = color  # (0,0,0)
    position = pos  # (100,100)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


def words_to_images(texts, img_h, img_w, n_channel=1):
    word_imgs = np.ones((len(texts), img_h, img_w, n_channel)).astype(np.uint8)
    for i in range(len(texts)):
        word_imgs[i] = pil_text_img(word_imgs[i], texts[i], (1, 1), textSize=25)
    if n_channel > 1:
        word_imgs = word_imgs.astype(np.uint8)
    else:
        word_imgs = word_imgs.sum(axis=-1, keepdims=True).astype(np.uint8)
    word_imgs = torch.from_numpy(word_imgs).permute([0, 3, 1, 2]).float() / 128 - 1
    return word_imgs


def override_config(config: OmegaConf, overrides: List[str]) -> OmegaConf:
    """
    Override config values using dot-notation keys.

    Args:
        config: OmegaConf configuration object
        overrides: List of override strings in format "key.path=value"
                   Example: ["style_extractor.pretrained_path=/path/to/model.pth",
                            "training.iteration=50000",
                            "vae.vae_pretrained_path=/path/to/vae"]

    Returns:
        Updated OmegaConf configuration

    Raises:
        ValueError: If override format is invalid
        TypeError: If value type conversion fails
    """
    if not overrides:
        return config

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value'")

        key_path, value_str = override.split("=", 1)
        key_path = key_path.strip()
        value_str = value_str.strip()

        if not key_path:
            raise ValueError(f"Empty key path in override: '{override}'")

        # Convert value to appropriate type
        value = _parse_value(value_str)

        # Navigate through the config to set the value
        keys = key_path.split(".")
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                # Create intermediate dictionaries if they don't exist
                OmegaConf.update(current, key, {})
            current = current[key]

        # Set the final value
        final_key = keys[-1]
        OmegaConf.update(current, final_key, value)

    return config


def _parse_value(value_str: str):
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    if value_str.lower() in ["null", "none"]:
        return None

    try:
        return int(value_str)
    except ValueError:
        pass

    try:
        return float(value_str)
    except ValueError:
        pass

    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return value_str[1:-1]

    return value_str


def determine_optimal_num_workers(device="cuda", verbose=True):
    """
    Determine optimal num_workers for PyTorch DataLoader based on system resources.

    This function analyzes the system's CPU cores, available RAM, and GPU availability
    to recommend an optimal number of workers for data loading. The strategy balances
    parallel data loading benefits against resource contention and overhead.

    Strategy:
    - For GPU training: Use 4-8 workers depending on CPU cores and memory
    - For CPU training: Use fewer workers to avoid threading overhead
    - Consider available RAM to prevent OOM errors
    - Scale with GPU count for multi-GPU setups

    Args:
        device (str): Training device - 'cuda' or 'cpu'. Default: 'cuda'
        verbose (bool): Print system configuration and recommendation. Default: True

    Returns:
        int: Optimal number of workers for DataLoader

    Examples:
        >>> # GPU training with auto-detection
        >>> num_workers = determine_optimal_num_workers()
        >>> DataLoader(dataset, num_workers=num_workers)

        >>> # CPU training
        >>> num_workers = determine_optimal_num_workers(device='cpu')
        >>> DataLoader(dataset, num_workers=num_workers)

        >>> # Silent mode
        >>> num_workers = determine_optimal_num_workers(verbose=False)
    """
    # Get CPU information
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 4  # Default fallback

    # Get available RAM (in GB)
    available_ram_gb = psutil.virtual_memory().available / (1024**3)

    # Get GPU information if available
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    is_gpu_training = device.startswith("cuda") and gpu_count > 0

    # Determine optimal workers based on training mode and resources
    if is_gpu_training:
        # GPU training: Can use more workers for data loading
        # Rule of thumb: 4-8 workers per GPU, capped by CPU cores and memory
        base_workers_per_gpu = 4

        # Adjust based on available RAM
        # Each worker can use ~0.5-1GB RAM depending on batch size and image size
        # Conservative estimate: use 40% of available RAM for workers
        max_workers_by_ram = int(available_ram_gb * 0.4)

        # Calculate optimal workers
        optimal_workers = min(
            base_workers_per_gpu * gpu_count,  # Scale with GPU count
            cpu_count - 2,  # Leave some cores for main process and system
            max_workers_by_ram,  # Don't exceed RAM capacity
            16,  # Hard cap - diminishing returns beyond this
        )

        # Minimum workers for GPU training
        optimal_workers = max(optimal_workers, 2)

    else:
        # CPU training: Use fewer workers to avoid threading overhead
        # CPU-bound workloads don't benefit from multiple workers as much
        optimal_workers = min(
            max(1, cpu_count // 4),  # 25% of cores
            4,  # Hard cap for CPU training
        )

    # Log the decision if verbose
    if verbose:
        print("System Configuration:")
        print(f"  CPU cores: {cpu_count}")
        print(f"  Available RAM: {available_ram_gb:.2f} GB")
        if gpu_count > 0:
            print(f"  GPU count: {gpu_count}")
            print(f"  Training device: {device}")
        print(f"  Recommended num_workers: {optimal_workers}")

    return optimal_workers


def download_pretrained_model(
    url: str,
    cache_dir: str | Path = "pretrained",
    filename: str | None = None,
    force_download: bool = False,
) -> str:
    """
    Download pretrained model from URL if not already cached.

    Args:
        url: URL to download from (e.g., HuggingFace hub URL)
        cache_dir: Directory to cache downloaded files
        filename: Optional filename to use. If None, derives from URL
        force_download: If True, re-download even if file exists

    Returns:
        Path to downloaded/cached file
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Derive filename from URL if not provided
    if filename is None:
        filename = url.split("/")[-1]
        filename = filename.split("?")[0]

    cached_path = cache_dir / filename

    if cached_path.exists() and not force_download:
        print(f"Using cached pretrained model: {cached_path}")
        return str(cached_path)

    print(f"Downloading pretrained model from {url}...")
    print(f"Saving to: {cached_path}")

    try:
        # Download with progress bar
        def _report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(
                f"\rProgress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)",
                end="",
            )

        urllib.request.urlretrieve(url, cached_path, _report_progress)
        print(f"Successfully downloaded to {cached_path}")

    except Exception as e:
        if cached_path.exists():
            cached_path.unlink()
        raise RuntimeError(f"Failed to download pretrained model from {url}: {e}")

    return str(cached_path)


def resolve_pretrained_path(pretrained_config) -> Union[str, tuple]:
    """
    Resolve pretrained path from config - supports local paths, URLs, and HuggingFace identifiers.

    Args:
        pretrained_config: Can be:
            - Dict with 'url' and optional 'cache_dir', 'filename'
            - Dict with 'huggingface' and optional 'subfolder', 'cache_dir', 'filename'

    Returns:
        Resolved path to model weights (for URL: returns file path, for HuggingFace: returns cache_dir)

    Examples:
        >>> # URL
        >>> resolve_pretrained_path({
        ...     "url": "https://download.pytorch.org/models/inception-v3-2015-12-05-6726825d.pth"
        ... })

        >>> # HuggingFace (downloads and returns cache directory)
        >>> resolve_pretrained_path({
        ...     "huggingface": "stabilityai/sd-vae-ft-mse",
        ...     "subfolder": "vae",
        ...     "cache_dir": "pretrained/vae"
        ... })
    """
    if pretrained_config is None:
        return None

    pretrained_config = (
        OmegaConf.create(pretrained_config) if not isinstance(pretrained_config, OmegaConf) else pretrained_config
    )

    cache_dir = pretrained_config.get("cache_dir", "pretrained")

    if "url" in pretrained_config:
        url = pretrained_config.url
        filename = pretrained_config.get("filename", None)
        force_download = pretrained_config.get("force_download", False)
        return download_pretrained_model(url, cache_dir, filename, force_download)

    if "huggingface" in pretrained_config:
        repo_id = pretrained_config.huggingface
        subfolder = pretrained_config.get("subfolder", None)
        ignore_patterns = pretrained_config.get("ignore_patterns", None)
        force_download = pretrained_config.get("force_download", False)

        if Path(cache_dir).exists() and not force_download:
            print(f"Using cached pretrained model: {cache_dir}")
            return cache_dir

        print(f"Downloading model from HuggingFace: {repo_id}")
        if subfolder:
            print(f"  Subfolder: {subfolder}")

        try:
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=f"{subfolder}/*",
                ignore_patterns=ignore_patterns,
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
            )
            print(f"Model downloaded to: {cache_dir}")

            return cache_dir

        except Exception as e:
            raise RuntimeError(f"Failed to download model from HuggingFace ({repo_id}): {e}")

    raise ValueError(
        f"Invalid pretrained_config format: {pretrained_config}\n"
        f"Expected format:\n"
        f"  - Dict with 'url': {{'url': '...', 'cache_dir': '...'}}\n"
        f"  - Dict with 'huggingface': {{'huggingface': 'model_id', 'subfolder': '...', 'cache_dir': '...'}}"
    )
