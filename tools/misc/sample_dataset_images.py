#!/usr/bin/env python3
"""
Randomly sample a subset of images from each dataset in data/preprocess.

Loads train.json from transcriptions folder and samples images based on writer ID.
Output structure: assets/sample/<dataset>/images/<writer_id>/*.png

Usage:
    python tools/sample_dataset_images.py --num-writers 50 --num-samples-per-writer 10 [--seed 42]
    python tools/sample_dataset_images.py --dataset IAM --num-writers 30 --num-samples-per-writer 5
"""

import argparse
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


# Available datasets
DATASETS = [
    "IAM",
    "IMGUR5K",
    "IIIT_English",
    "CASIA_HWDB",
    "ViHTGen",
]

# Default number of writers and samples per writer
DEFAULT_NUM_WRITERS = 50
DEFAULT_SAMPLES_PER_WRITER = 10


def load_train_json(transcription_dir: Path) -> Dict:
    """
    Load train.json from transcriptions directory.

    Args:
        transcription_dir: Path to transcriptions directory

    Returns:
        Dictionary with image metadata (image name, writer id, label)
    """
    train_json_path = transcription_dir / "train.json"

    if not train_json_path.exists():
        raise FileNotFoundError(f"train.json not found at {train_json_path}")

    with open(train_json_path, "r") as f:
        data = json.load(f)

    return data


def group_images_by_writer(train_data: Dict, images_dir: Path) -> Dict[str, List[Path]]:
    """
    Group images by writer ID from train.json data.

    Args:
        train_data: Dictionary from train.json with image metadata
        images_dir: Directory containing the images

    Returns:
        Dictionary mapping writer_id to list of image paths
    """
    writer_images = defaultdict(list)

    for item in train_data.values():
        image_name = item.get("image")
        writer_id = item.get("s_id")  # s_id is the writer/style ID

        if not image_name or not writer_id:
            continue

        # Find the image file (could be in subdirectories)
        image_path = find_image_path(images_dir, image_name)

        if image_path and image_path.exists():
            writer_images[writer_id].append(image_path)

    return dict(writer_images)


def find_image_path(images_dir: Path, image_name: str) -> Path:
    """
    Find an image file in the images directory (supports subdirectories).

    Args:
        images_dir: Root images directory
        image_name: Name of the image file to find

    Returns:
        Path to the image if found, None otherwise
    """
    # First try direct path
    direct_path = images_dir / image_name
    if direct_path.exists():
        return direct_path

    # Search recursively in subdirectories
    for found_path in images_dir.rglob(image_name):
        if found_path.is_file():
            return found_path

    return None


def sample_writers_and_images(
    writer_images: Dict[str, List[Path]],
    num_writers: int,
    samples_per_writer: int,
    seed: int,
) -> Dict[str, List[Path]]:
    """
    First sample N writers, then sample M images per writer.

    Args:
        writer_images: Dictionary mapping writer_id to list of image paths
        num_writers: Number of writers to sample
        samples_per_writer: Number of samples to draw per sampled writer
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping writer_id to list of sampled image paths
    """
    random.seed(seed)
    sampled_writer_images = {}

    # Filter writers that have enough images
    eligible_writers = {
        writer_id: images for writer_id, images in writer_images.items() if len(images) >= samples_per_writer
    }

    if len(eligible_writers) < num_writers:
        print(f"  Warning: Only {len(eligible_writers)} writers have >= {samples_per_writer} images")
        num_writers = len(eligible_writers)

    # Sample writers
    if num_writers >= len(eligible_writers):
        selected_writers = sorted(eligible_writers.keys())
    else:
        selected_writers = sorted(random.sample(list(eligible_writers.keys()), num_writers))

    print(f"  Selected {len(selected_writers)} writers")

    # Sample images from each selected writer
    for writer_id in selected_writers:
        images = eligible_writers[writer_id]
        available = len(images)

        if samples_per_writer >= available:
            # Take all images from this writer
            sampled_writer_images[writer_id] = images
            print(f"    Writer {writer_id}: {available} images (using all)")
        else:
            # Randomly sample from this writer
            writer_samples = random.sample(images, samples_per_writer)
            sampled_writer_images[writer_id] = writer_samples
            print(f"    Writer {writer_id}: {samples_per_writer}/{available} images")

    return sampled_writer_images


def copy_images_by_writer(writer_images: Dict[str, List[Path]], target_root: Path) -> Dict[str, int]:
    """
    Copy images to target directory organized by writer ID folders.

    Output structure:
        target_root/
            <writer_id_1>/
                image1.png
                image2.png
                ...
            <writer_id_2>/
                image1.png
                ...

    Args:
        writer_images: Dictionary mapping writer_id to list of image paths to copy
        target_root: Root target directory

    Returns:
        Dictionary with copy statistics
    """
    stats = {
        "total_writers": len(writer_images),
        "total_images": sum(len(imgs) for imgs in writer_images.values()),
        "copied": 0,
        "skipped": 0,
        "errors": 0,
    }

    for writer_id, images in sorted(writer_images.items()):
        # Create writer-specific directory
        writer_dir = target_root / writer_id
        writer_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            try:
                # Use just the filename (not full path) in writer folder
                target_path = writer_dir / img_path.name

                # Copy file if it doesn't exist or is different
                if target_path.exists():
                    if target_path.stat().st_size == img_path.stat().st_size:
                        stats["skipped"] += 1
                        continue

                shutil.copy2(img_path, target_path)
                stats["copied"] += 1

            except Exception as e:
                print(f"  Error copying {img_path}: {e}")
                stats["errors"] += 1

    return stats


def process_dataset(
    dataset_name: str,
    num_writers: int,
    samples_per_writer: int,
    seed: int,
    base_dir: Path = Path("data/preprocess"),
    output_base: Path = Path("assets/sample"),
) -> None:
    """
    Process a single dataset: sample writers and images, then copy organized by writer ID.

    Args:
        dataset_name: Name of the dataset
        num_writers: Number of writers to sample
        samples_per_writer: Number of samples to extract per writer
        seed: Random seed
        base_dir: Base directory containing datasets
        output_base: Base output directory
    """
    print(f"\n{'=' * 60}")
    print(f"Processing {dataset_name}")
    print(f"{'=' * 60}")

    # Paths
    dataset_dir = base_dir / dataset_name
    source_dir = dataset_dir / "images"
    transcription_dir = dataset_dir / "transcriptions"

    # Check directories exist
    if not source_dir.exists():
        print(f"  Error: Directory not found: {source_dir}")
        return

    if not transcription_dir.exists():
        print(f"  Error: Directory not found: {transcription_dir}")
        return

    try:
        # Load train.json
        print("  Loading train.json...")
        train_data = load_train_json(transcription_dir)
        print(f"  Loaded {len(train_data)} entries")

        # Group images by writer ID
        print("  Grouping images by writer ID...")
        writer_images = group_images_by_writer(train_data, source_dir)
        num_writers_total = len(writer_images)
        total_images = sum(len(imgs) for imgs in writer_images.values())
        print(f"  Found {num_writers_total} writers with {total_images} images")

        if not writer_images:
            print("  Warning: No valid images found in train.json")
            return

        # Sample writers and images
        print(f"  Sampling {num_writers} writers with {samples_per_writer} images each (seed={seed})...")
        sampled_writer_images = sample_writers_and_images(writer_images, num_writers, samples_per_writer, seed)

        total_sampled = sum(len(imgs) for imgs in sampled_writer_images.values())
        print(f"  Total sampled: {total_sampled} images from {len(sampled_writer_images)} writers")

        # Target directory (organized by writer ID)
        target_dir = output_base / dataset_name

        # Copy organized by writer ID
        print(f"  Copying to {target_dir} (organized by writer ID)...")
        stats = copy_images_by_writer(sampled_writer_images, target_dir)

        # Print statistics
        print("\n  Statistics:")
        print(f"    Total writers available: {num_writers_total}")
        print(f"    Writers selected:       {stats['total_writers']}")
        print(f"    Total images sampled:   {stats['total_images']}")
        print(f"    Copied:                 {stats['copied']}")
        print(f"    Skipped (exist):        {stats['skipped']}")
        print(f"    Errors:                 {stats['errors']}")

    except FileNotFoundError as e:
        print(f"  Error: {e}")
    except json.JSONDecodeError as e:
        print(f"  Error: Failed to parse train.json: {e}")
    except Exception as e:
        print(f"  Unexpected error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample images from handwriting datasets by writer ID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool loads train.json from each dataset's transcriptions folder and samples
images from selected writers, organizing output by writer ID folders.

Output structure:
  assets/sample/<dataset>/images/<writer_id>/*.png

Examples:
  # Sample 50 writers with 10 images each from all datasets
  python tools/sample_dataset_images.py --num-writers 50 --num-samples-per-writer 10

  # Sample 30 writers with 5 images each from IAM dataset only
  python tools/sample_dataset_images.py --dataset IAM --num-writers 30 --num-samples-per-writer 5

  # Sample with specific random seed for reproducibility
  python tools/sample_dataset_images.py --num-writers 100 --num-samples-per-writer 20 --seed 123

  # Sample different amounts from multiple datasets
  python tools/sample_dataset_images.py --dataset IAM --num-writers 50 --num-samples-per-writer 10 \\
                                        --dataset IMGUR5K --num-writers 30 --num-samples-per-writer 5
        """,
    )

    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        help="Dataset to process (can specify multiple times). Default: all datasets",
    )

    parser.add_argument(
        "--num-writers",
        type=int,
        default=DEFAULT_NUM_WRITERS,
        help=f"Number of writers to sample per dataset (default: {DEFAULT_NUM_WRITERS})",
    )

    parser.add_argument(
        "--num-samples-per-writer",
        type=int,
        default=DEFAULT_SAMPLES_PER_WRITER,
        help=f"Number of samples to extract per writer ID (default: {DEFAULT_SAMPLES_PER_WRITER})",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/preprocess",
        help="Base directory containing datasets (default: data/preprocess)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/sample",
        help="Output directory for samples (default: assets/sample)",
    )

    args = parser.parse_args()

    # Determine which datasets to process
    datasets_to_process = args.dataset if args.dataset else DATASETS

    print("=" * 60)
    print("Dataset Image Sampling Tool (by Writer ID)")
    print("=" * 60)
    print(f"Datasets:               {', '.join(datasets_to_process)}")
    print(f"Number of writers:      {args.num_writers}")
    print(f"Samples per writer:     {args.num_samples_per_writer}")
    print(f"Random seed:            {args.seed}")
    print(f"Source base:            {args.base_dir}")
    print(f"Output base:            {args.output_dir}")

    base_dir = Path(args.base_dir)
    output_base = Path(args.output_dir)

    # Process each dataset
    for dataset in datasets_to_process:
        process_dataset(
            dataset_name=dataset,
            num_writers=args.num_writers,
            samples_per_writer=args.num_samples_per_writer,
            seed=args.seed,
            base_dir=base_dir,
            output_base=output_base,
        )

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
