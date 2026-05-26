# CONSTANT Handwriting Generation CLI Tool

A command-line interface for generating OCR corpus using the CONSTANT handwriting generation model.

## Installation

After installing the project dependencies, the CLI tool will be available as `constant-gen`:

```bash
# Install with uv
uv pip install -e .
```

## Available Commands

### 1. Generate OCR Corpus (Multi-Style)

Generate handwriting samples using multiple writer styles randomly selected from a dataset:

```bash
uv run constant-gen generate-multi-style TEXT_CORPUS [OPTIONS]
```

**Arguments:**
- `TEXT_CORPUS`: Path to text file containing text to generate (one line per text)

**Options:**
- `-d, --template-dataset`: Dataset to use as reference (choices: IAM, IMGUR5K, IIT_English_Words, CASIA_HWDB, ViHTGen, default: IAM)
- `-o, --save-dir`: Output directory (default: output/ocr_synthesize_data)
- `-n, --num-styles`: Number of reference styles/writers to use (default: all available)
- `-s, --total-sample`: Total number of samples to generate (default: 1000000)
- `--device`: Inference device - cuda or cpu (default: cuda)
- `--seed`: Random seed for reproducibility (default: 0)

**Examples:**

```bash
# Basic usage with IAM dataset
uv run constant-gen generate-multi-style corpus.txt

# Generate with specific number of styles
uv run constant-gen generate-multi-style corpus.txt --num-styles 50

# Generate with different template dataset
uv run constant-gen generate-multi-style corpus.txt --template-dataset IMGUR5K

# Generate specific number of samples
uv run constant-gen generate-multi-style corpus.txt --total-sample 10000 --save-dir my_output

# Run on CPU
uv run constant-gen generate-multi-style corpus.txt --device cpu
```

### 2. Generate OCR Corpus (Single Style)

Generate handwriting samples using a single reference style image:

```bash
uv run constant-gen generate-single-style TEXT_CORPUS [OPTIONS]
```

**Arguments:**
- `TEXT_CORPUS`: Path to text file containing text to generate (one line per text)

**Required Options:**
- `-r, --ref-image`: Path to reference style image to use for generation

**Optional Options:**
- `-d, --template-dataset`: Dataset that the reference image belongs to (choices: IAM, IMGUR5K, IIT_English_Words, CASIA_HWDB, ViHTGen, default: IAM)
- `-o, --save-dir`: Output directory (default: output/ocr_synthesize_data)
- `--device`: Inference device - cuda or cpu (default: cuda)

**Examples:**

```bash
# Generate using a specific reference image
uv run constant-gen generate-single-style corpus.txt --ref-image /path/to/style.png

# Generate with reference image from specific dataset
uv run constant-gen generate-single-style corpus.txt --ref-image style.png --template-dataset IMGUR5K

# Generate using reference image with custom output directory
uv run constant-gen generate-single-style corpus.txt --ref-image style.png --save-dir my_output
```

### 3. List Available Datasets

View available template datasets and their configurations:

```bash
uv run constant-gen list-datasets
```

## Output Structure

Generated files will be organized differently depending on the generation mode used.

### Multi-Style Output Structure

```
output/ocr_synthesize_data/
└── YYYYMMDD_HHMMSS/
    ├── images/
    │   ├── writer_id_1/
    │   │   ├── uuid1.png
    │   │   └── uuid2.png
    │   ├── writer_id_2/
    │   │   └── uuid3.png
    │   └── ...
    ├── generation_log.csv          # Contains: style_id, text, ref_img, gen_name
    ├── process_log.json            # Processing statistics and metadata
    └── generation.log              # Detailed execution logs
```

### Single-Style Output Structure

```
output/ocr_synthesize_data/
└── YYYYMMDD_HHMMSS/
    ├── images/
    │   ├── uuid1.png
    │   ├── uuid2.png
    │   ├── uuid3.png
    │   └── ...
    ├── generation_log.csv          # Contains: text, ref_img, gen_name
    ├── process_log.json            # Processing statistics and metadata
    └── generation.log              # Detailed execution logs
```

## Process Log Format

The `process_log.json` contains comprehensive generation metadata with different formats depending on the generation mode used.

### Multi-Style Generation Process Log

```json
{
  "datetime": "20260508_143022",
  "text_corpus_path": "/path/to/corpus.txt",
  "template_dataset": "IAM",
  "num_styles": 50,
  "total_samples": 100000,
  "device": "cuda",
  "random_seed": 0,
  "start_time": "2026-05-08T14:30:22.123456",
  "end_time": "2026-05-08T15:45:30.654321",
  "total_duration_seconds": 4508.53,
  "actual_samples_generated": 100000,
  "num_writers_processed": 50,
  "average_samples_per_second": 22.18,
  "writers_processing": [
    {
      "writer_id": "000-01-01",
      "text_samples_processed": 2000,
      "duration_seconds": 89.45,
      "samples_per_second": 22.36
    }
  ]
}
```

### Single-Style Generation Process Log

```json
{
  "datetime": "20260508_143022",
  "text_corpus_path": "/path/to/corpus.txt",
  "template_dataset": "IAM",
  "ref_image_path": "/path/to/style.png",
  "device": "cuda",
  "start_time": "2026-05-08T14:30:22.123456",
  "end_time": "2026-05-08T14:45:30.654321",
  "total_duration_seconds": 908.53,
  "generation_mode": "single_style",
  "generation_duration_seconds": 850.32,
  "actual_samples_generated": 5000,
  "samples_per_second": 5.88
}
```

## Usage Tips

### 1. Text Corpus Format

Ensure your corpus file has one text sample per line:
```
hello world
example text
another sample
```

Texts longer than `MAX_LENGTH` (dataset-dependent, typically 10 characters) will be automatically filtered out during generation.

**Sample Corpus**: A sample text corpus is available at `assets/text_corpus/sample.txt` that you can use for quick testing:
```bash
# Quick test with multi-style generation
uv run constant-gen generate-multi-style assets/text_corpus/sample.txt --num-styles 5 --total-sample 100

# Quick test with single-style generation
uv run constant-gen generate-single-style assets/text_corpus/sample.txt --ref-image assets/sample/IAM/000-01-01/000_01_0001.png
```


### 2. Extending Sample Reference Images

You can extend the number of reference writer styles and the number of images for each writer by running:

```bash
uv run scripts/sample_dataset_images.py -dataset <dataset_name> --num-writers <desired_num_writers> --num-samples-per-writer <desired_num_images>
```

This creates additional reference images in `assets/sample/<dataset_name>/` organized by writer ID.

In order to do this, first you will need to download and preprocess available datasets follow the instruction in [DATASET.md](DATASET.md)

### 3. Memory Management

For large-scale generation, consider:
- **Multi-style**: Use `--num-styles` to limit the number of writers (reduces memory overhead)
- **Single-style**: More memory efficient as it uses only one reference image
- Reduce `--total-sample` for testing
- Use `--device cpu` if GPU memory is limited (though generation will be slower)
