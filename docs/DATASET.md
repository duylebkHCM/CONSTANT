# Data Preparation

This document describes how to download, preprocess, and configure the datasets used in CONSTANT.

## Supported Datasets

We conduct experiments on the following publicly available handwriting datasets. Download each dataset following the instructions below and place the raw data under `data/raw/<dataset_name>`.

| Dataset | Source | Notes |
|---------|--------|-------|
| **IAM** | [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | Ground truth follows GANWriting format ([link](https://github.com/koninik/WordStylist/tree/main/gt)): `gan.iam.tr_va.gt.filter27` (train) and `gan.iam.test.gt.filter27` (test). |
| **IMGUR5K** | [IMGUR5K Repository](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset) | Clone the repository and run `download_imgur5k.py` to download raw images into the `images/` folder. |
| **IIIT-HW-English-Word** | [IIIT-HW Dataset](https://cvit.iiit.ac.in/images/datasets/english_handwritten/dataset.zip) | The raw data folder should contain two sub-folders: `word_level_images/` and `file/`. |
| **CASIA-HWDB** | [CASIA-HWDB](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html) | We use the `comp_test_data` partition. |

For the proposed **ViHTGen** dataset, a preprocessed version is available at [Link]().

## Preprocessing

Each dataset has a dedicated preprocessing script. Run the appropriate command below after downloading the raw data.

### IAM

```bash
uv run tools/data_converter/IAM/preprocess.py <path_to_raw_data> <path_to_train_groundtruth> <path_to_test_groundtruth>
```

### IMGUR5K

Preprocessing requires two sequential steps.

**Step 1 — Parse raw annotations:**

```bash
uv run tools/data_converter/IMGUR5K/parse_data.py --path_to_repo <path_to_IMGUR5K_repo>
```

Parsed image–label pairs are written to folders named `imgur5k_<split>_0_None`, where `<split>` is `train`, `val`, or `test`. Only the `train` and `test` splits are used in this project.

**Step 2 — Preprocess parsed data:**

```bash
uv run tools/data_converter/IMGUR5K/preprocess.py <path_to_raw_data>
```

### IIIT-HW-English-Word

```bash
uv run tools/data_converter/IIIT_English/preprocess.py <path_to_raw_data>
```

### CASIA-HWDB

```bash
uv run tools/data_converter/CASIA_HWDB/preprocess.py <path_to_raw_data>
```

### Output Structure

All preprocessed data is written to `./data/processed/<dataset_name>/` with the following layout:

```text
data/processed/<dataset_name>/
├── images/            # Preprocessed word images
├── transcriptions/    # train.json and test.json — each entry contains an
│                      #   'image' field with a path relative to images/
└── writers_dict/      # train.json and test.json — lists of writer IDs
```

## Using a Custom Dataset

To train on your own dataset, ensure the raw data satisfies two requirements:

1. Each sample must be associated with a **writer ID**.
2. Each writer ID maps to a list of **(image, text)** pairs.

Refer to [tools/converter/IAM/preprocess.py](tools/converter/IAM/preprocess.py) for an example preprocessing script that produces the expected output format.

### Image Preprocessing Rules

1. Resize each image to a fixed height of **64 px** while preserving the aspect ratio.
2. Adjust the width to be divisible by `CHAR_WIDTH` (default: **32**, following the literature).

### Text Label Length

In our experiments, `MAX_LENGTH` is set to **10** characters. Adjust this value as needed for your dataset.

### Configuration

After preprocessing, update the dataset paths in your training configuration file. See [TRAINING_SAMPLING.md](TRAINING_SAMPLING.md) for full details.

```yaml
dataset:
  params:
    dset_name: <vocab_type>  # Supported: english, chinese, vietnamese
    full_dict_path: <path_to_your_preprocessed_data>/transcriptions/train.json
    image_path: <path_to_your_preprocessed_data>/images
    writer_dict_path: <path_to_your_preprocessed_data>/writers_dict/train.json

test_dataset:
  params:
    dset_name: <vocab_type>
    full_dict_path: <path_to_your_preprocessed_data>/transcriptions/test.json
    image_path: <path_to_your_preprocessed_data>/images
    writer_dict_path: <path_to_your_preprocessed_data>/writers_dict/test.json
```
