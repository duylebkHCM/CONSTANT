# Repository Structure

```text
.
├── api/                          # Application interfaces
│   ├── cli_tool.py               # CLI tool for inference/generation
│   └── gradio_app.py             # Gradio web demo interface
├── assets/                       # Static assets
│   ├── font/arial.ttf            # Font for text rendering
│   └── text_corpus/sample.txt    # Sample text corpus
├── configs/                      # Training configuration files
│   ├── constant_CASIA.yaml       # CASIA-HWDB dataset config
│   ├── constant_IAM.yaml         # IAM dataset config
│   ├── constant_IIIT.yaml        # IIIT-HW-English-Word config
│   ├── constant_IMGUR5K.yaml     # IMGUR5K dataset config
│   └── constant_ViHTGen.yaml     # ViHTGen dataset config
├── docker/                       # Docker deployment
│   └── Dockerfile                # Container build definition
├── docs/                         # Documentation
│   ├── CLI.md                    # CLI usage guide
│   ├── CODE_INFO.md              # This file
│   ├── DATASET.md                # Dataset preparation guide
│   ├── EVALUATING.md             # Evaluation guide
│   └── TRAINING_SAMPLING.md      # Training and sampling guide
├── scripts/                      # Shell scripts
│   └── start_docker.sh           # Docker launch script
├── src/                          # Core source code
│   ├── data/                     # Data pipeline
│   │   ├── augment.py            # Data augmentation transforms
│   │   ├── constant.py           # Constants (image dimensions, max length)
│   │   ├── dataset.py            # Dataset classes (BaseDataset, DatasetVariableStyleReference)
│   │   ├── sampler.py            # ByWriterIDSampler for diverse batches
│   │   ├── tokenizer.py          # Character-level tokenizer
│   │   └── vocab/                # Vocabulary files
│   │       ├── cn.txt            # Chinese characters
│   │       ├── eng.txt           # English characters
│   │       └── vi.txt            # Vietnamese characters
│   ├── metrics/                  # Evaluation metrics
│   │   ├── fid.py                # Fréchet Inception Distance
│   │   ├── hwd.py                # Handwriting Distance
│   │   └── inception.py          # Inception network for FID
│   ├── model/                    # Model definitions
│   │   ├── SAQ.py                # Style-Aware Quantization
│   │   ├── base.py               # Base model class
│   │   ├── content_encoder.py    # Character encoder (text conditioning)
│   │   ├── diffusion.py          # Diffusion engine (noise schedule, DDIM sampling)
│   │   ├── pce.py                # Patch Contrastive Enhancement
│   │   ├── pipeline.py           # Main HandwritingGenerationPipeline
│   │   └── sce.py                # Style Contrastive Enhancement (CLIP-based)
│   ├── modules/                  # Reusable building blocks
│   │   ├── attention.py          # Attention mechanisms
│   │   ├── attention_pool2d.py   # 2D attention pooling
│   │   ├── codebook.py           # Vector quantization codebook
│   │   ├── context_embedding.py  # Context embedding layers
│   │   ├── drop_path.py          # Stochastic depth (DropPath)
│   │   ├── ema.py                # Exponential moving average
│   │   ├── helpers.py            # Helper functions
│   │   ├── patch_nce.py          # Patch NCE loss
│   │   ├── resnet.py             # ResNet blocks
│   │   ├── scheduler/            # Learning rate schedulers
│   │   │   ├── __init__.py
│   │   │   └── build_scheduler.py
│   │   ├── unet.py               # UNet backbone for diffusion
│   │   └── utils.py              # Module utilities
│   └── utils.py                  # General utilities
├── tools/                        # Training, evaluation, and data scripts
│   ├── converter/                # Dataset preprocessing converters
│   │   ├── CASIA_HWDB/preprocess.py
│   │   ├── IAM/preprocess.py
│   │   ├── IIIT_English/preprocess.py
│   │   └── IMGUR5K/
│   │       ├── parse_data.py
│   │       └── preprocess.py
│   ├── eval.py                   # Evaluation script (FID, HWD metrics)
│   ├── misc/
│   │   └── sample_dataset_images.py  # Dataset visualization utility
│   ├── sampling.py               # Inference / image generation
│   └── train.py                  # Training entry point (single-GPU and DDP)
├── pyproject.toml                # Project metadata and dependencies
├── requirements.txt              # Pip dependencies
└── README.md                     # Project README
```
