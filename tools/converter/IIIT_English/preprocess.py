import sys
import json
from pathlib import Path
from PIL import Image
from src.data.constant import CHAR_WIDTH, FIXED_HEIGHT, MAX_LENGTH


def parse_groundtruth(img_path, label_path, split):
    img_list = [p.strip() for p in open(img_path).readlines()]
    label_list = [p.strip() for p in open(label_path).readlines()]
    assert len(img_list) == len(label_list)
    transcriptions = {}
    writer_dict = {}
    writer_idx = 0
    sample_idx = 0

    for idx, (img_path, label) in enumerate(zip(img_list, label_list)):
        if len(label) > MAX_LENGTH:
            continue

        w_id = Path(img_path).parent.name
        img_name = Path(*Path(img_path).parts[-2:])
        transcriptions[sample_idx] = {
            "image": str(img_name),
            "s_id": w_id,
            "label": label,
        }
        sample_idx += 1

        if w_id not in writer_dict:
            writer_dict[w_id] = writer_idx
            writer_idx += 1

    print("number of train writer styles", len(writer_dict))

    # create json object from dictionary if you want to save writer ids
    json_dict = json.dumps(writer_dict)
    f = open(f"{OUTPUT_DIR}/writers_dict/{split}.json", "w")
    f.write(json_dict)
    f.close()

    json_dict = json.dumps(transcriptions)
    f = open(f"{OUTPUT_DIR}/transcriptions/{split}.json", "w")
    f.write(json_dict)
    f.close()

    return transcriptions


def resize_aspect_ratio(img, label):
    img_width, img_height = img.size
    img = img.resize((int(img_width * FIXED_HEIGHT), FIXED_HEIGHT))
    rescale_width = len(label) * CHAR_WIDTH
    img = img.resize((rescale_width, FIXED_HEIGHT), Image.Resampling.BILINEAR)
    return img


if __name__ == "__main__":
    raw_data_path = sys.argv[1]

    OUTPUT_DIR = Path("./data/preprocess/IIIT_English")
    OUTPUT_DIR.joinpath("writers_dict").mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.joinpath("transcriptions").mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.joinpath("images").mkdir(exist_ok=True, parents=True)

    train_infos = parse_groundtruth(
        f"{raw_data_path}/file/train_images.txt",
        f"{raw_data_path}/file/train_labels.txt",
        split="train",
    )
    test_infos = parse_groundtruth(
        f"{raw_data_path}/file/val_images.txt",
        f"{raw_data_path}/file/val_labels.txt",
        split="test",
    )

    for trans_idx, trans in train_infos.items():
        img = Image.open(f"{raw_data_path}/word_level_images/{trans['image']}")
        resize_img = resize_aspect_ratio(img, trans["label"])
        img_name = Path(trans["image"])
        if not OUTPUT_DIR.joinpath("images", img_name.parent).exists():
            OUTPUT_DIR.joinpath("images", img_name.parent).mkdir(parents=True)
        resize_img.save(OUTPUT_DIR.joinpath("images", img_name), "PNG")

    for trans_idx, trans in test_infos.items():
        img = Image.open(f"{raw_data_path}/word_level_images/{trans['image']}")
        resize_img = resize_aspect_ratio(img, trans["label"])
        img_name = Path(trans["image"])
        if not OUTPUT_DIR.joinpath("images", img_name.parent).exists():
            OUTPUT_DIR.joinpath("images", img_name.parent).mkdir(parents=True)
        resize_img.save(OUTPUT_DIR.joinpath("images", img_name), "PNG")
