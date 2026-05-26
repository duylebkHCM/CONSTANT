# Evaluation

This guide explains how to evaluate the quality of generated handwriting images using the provided metrics computation tool.

## Available Metrics

The evaluation script supports the following metrics:

- **FID** (Fréchet Inception Distance): Measures the quality and diversity of generated images
- **HWD** (Handwriting Direction Distance): Measures the structural similarity of handwriting strokes

## Running Evaluation

### Basic Usage

```bash
# Compute both FID and HWD metrics
uv run src/tools/compute_metrices.py \
    --real_dir path/to/real/images \
    --fake_dir path/to/generated/images \
    --metrics fid hwd \
    --batchsize 64 \
    --mode global
```

### Command-Line Arguments

- `--real_dir`: Path to directory containing real handwriting images
- `--fake_dir`: Path to directory containing generated handwriting images
- `--metrics`: Metrics to compute (space-separated: `fid hwd`)
- `--batchsize`: Batch size for computation (default: 64)
- `--mode`: Evaluation mode
  - `global`: Compute overall metrics across all images
  - `by_wids`: Compute metrics per writer ID

### Per-Writer Evaluation

To evaluate quality for each writer separately:

```bash
uv run src/tools/compute_metrices.py \
    --real_dir path/to/real/images \
    --fake_dir path/to/generated/images \
    --metrics fid hwd \
    --batchsize 64 \
    --mode by_wids
```

This will output individual metrics for each writer ID, which is useful for analyzing style consistency.
