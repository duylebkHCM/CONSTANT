## Training

Training uses config files in `configs/` directory. Each dataset has its own config:
- `configs/constant_IAM.yaml` - IAM dataset
- `configs/constant_IMGUR5K.yaml` - IMGUR5K dataset
- `configs/constant_IIIT.yaml` - IIIT-HW-English-Word dataset
- `configs/constant_CASIA.yaml` - CASIA-HWDB dataset
- `configs/constant_ViHTGen.yaml` - ViHTGen dataset

### Single GPU Training

The unified training script supports both single and multi-GPU training:

```bash
# Basic single GPU training
uv run tools/train.py --config-path configs/constant_IAM.yaml

# Override specific config values (e.g., pretrained paths)
uv run tools/train.py --config-path configs/constant_IAM.yaml \
  --config-override style_extractor.params.pretrained_path=/path/to/inception.pth \
  --config-override vae.vae_pretrained_path=/path/to/vae

# Resume from checkpoint
uv run tools/train.py --config-path configs/constant_IAM.yaml --ckpt_path path/to/checkpoint/ckpt.pth

# Train with pretrained weights
uv run tools/train.py --config-path configs/constant_IAM.yaml --pretrained_path path/to/pretrained.pth
```

### Multi-GPU Distributed Training (DDP)

For multi-GPU training, use torchrun with the same training script:

```bash
# Automatic DDP detection when using torchrun
uv run torchrun --nproc_per_node=4 tools/train.py --config-path configs/constant_IAM.yaml

# Specify which GPUs to use
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 tools/train.py --config-path configs/constant_IAM.yaml

# Multi-node training (advanced)
uv run torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr="10.0.0.1" --master_port=1234 \
  tools/train.py --config-path configs/constant_IAM.yaml
```

**Note**: When using DDP, the global batch size from config is automatically divided across GPUs. For example, if `training.batch_size: 128` in config and you use 4 GPUs, each GPU will process batches of 32 samples.

### Customize training configuration

```bash
# Override specific config values
uv run tools/train.py --config-path configs/constant_IAM.yaml \
  --config-override training.iteration=100000 \
  --config-override training.batch_size=64 \
  --config-override training.base_lr=0.0001
```

Training outputs are saved to `output/<config_name>/<timestamp>/` with:
- `checkpoint/` - Model checkpoints
- `images/` - Visualization samples during training
- `tbrun/` - TensorBoard logs
- `train.log` - Training logs

For DDP training, the output directory includes `_ddp_{num_gpus}gpu` suffix (e.g., `20240101120000_ddp_4gpu/`) to distinguish from single-GPU runs.


## Sampling

After training, use `tools/sampling.py` to generate handwriting images from a trained checkpoint. The script loads the config file automatically from the checkpoint's parent directory.

### Basic Usage

```bash
# Generate images using a trained checkpoint
uv run tools/sampling.py --weight-path output/constant_IAM/20240101120000/checkpoint/ckpt.pth

# Specify batch size and device
uv run tools/sampling.py --weight-path output/constant_IAM/20240101120000/checkpoint/ckpt.pth \
  --batchsize 32 --device cuda
```

### Overriding Sampling Parameters

You can override diffusion sampling parameters at inference time without modifying the config file:

```bash
# Override classifier-free guidance scale (default: 6)
uv run tools/sampling.py --weight-path path/to/ckpt.pth --cond_scale 4

# Override number of DDIM sampling steps (default: 50)
uv run tools/sampling.py --weight-path path/to/ckpt.pth --sampling_steps 100

# Enable/disable DDIM sampling
uv run tools/sampling.py --weight-path path/to/ckpt.pth --ddim_sampling true

# Combine multiple overrides
uv run tools/sampling.py --weight-path path/to/ckpt.pth \
  --cond_scale 4 --sampling_steps 100 --ddim_sampling true
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--weight-path` | Path to trained checkpoint (required) | — |
| `--batchsize` | Batch size for generation | 16 |
| `--device` | Inference device (`cuda` or `cpu`) | `cuda` |
| `--cond_scale` | Classifier-free guidance scale. Higher values increase style fidelity but may reduce diversity | From config |
| `--sampling_steps` | Number of DDIM denoising steps. More steps improve quality at the cost of speed | From config |
| `--ddim_sampling` | Whether to use DDIM sampling (`true`/`false`) | From config |

### Output Structure

Sampling outputs are saved under the checkpoint's experiment directory:

```
output/<config_name>/<timestamp>/eval/<checkpoint_name>/
├── evaluate_images/
│   ├── real/           # Ground truth images organized by writer ID
│   │   └── <writer_id>/
│   ├── gen/            # Generated images organized by writer ID
│   │   └── <writer_id>/
│   └── pair/           # Side-by-side comparison grids (real | generated | style reference)
│       └── <writer_id>/
└── test.log            # Sampling log with timing information
```

When override parameters are used, the eval subdirectory name includes the overridden values (e.g., `ckpt_4_100_true/`) to distinguish different sampling configurations.

### Evaluation

After sampling, compute metrics on the generated images using `tools/eval.py`:

```bash
# Compute FID score (global, style-agnostic)
uv run tools/eval.py \
  --real_dir output/.../eval/<eval_name>/evaluate_images/real \
  --fake_dir output/.../eval/<eval_name>/evaluate_images/gen \
  --metrics fid --mode global --batchsize 64

# Compute both FID and HWD metrics
uv run tools/eval.py \
  --real_dir path/to/evaluate_images/real \
  --fake_dir path/to/evaluate_images/gen \
  --metrics fid hwd --mode global

# Per-writer evaluation
uv run tools/eval.py \
  --real_dir path/to/evaluate_images/real \
  --fake_dir path/to/evaluate_images/gen \
  --metrics fid hwd --mode by_writer
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--real_dir` | Path to ground truth images (required) | — |
| `--fake_dir` | Path to generated images (required) | — |
| `--metrics` | Metrics to compute (`fid`, `hwd`, or both) | `fid` |
| `--mode` | `global` for style-agnostic or `by_writer` for per-writer evaluation | `global` |
| `--batchsize` | Batch size for metric computation | 64 |
| `--device` | Evaluation device | `cuda` |
