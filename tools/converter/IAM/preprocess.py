import sys
import json
from pathlib import Path
import PIL
from PIL import Image
from src.data.constant import CHAR_WIDTH, FIXED_HEIGHT


def parse_groundtruth(gt_path, split="train"):
    with open(gt_path, "r") as f:
        train_data = f.readlines()
        train_data = [i.strip().split(" ") for i in train_data]

        wr_dict = {}
        full_dict = {}
        image_wr_dict = {}
        img_word_dict = {}
        wr_index = 0
        idx = 0

        for i in train_data:
            s_id = i[0].split(",")[0]
            image = i[0].split(",")[1] + ".png"
            transcription = i[1]

            full_dict[idx] = {"image": image, "s_id": s_id, "label": transcription}
            image_wr_dict[image] = s_id
            img_word_dict[image] = transcription

            idx += 1
            if s_id not in wr_dict.keys():
                wr_dict[s_id] = wr_index
                wr_index += 1

        print("number of train writer styles", len(wr_dict))

    # create json object from dictionary if you want to save writer ids
    json_dict = json.dumps(wr_dict)
    f = open(f"{OUTPUT_DIR}/writers_dict/{split}.json", "w")
    f.write(json_dict)
    f.close()

    json_dict = json.dumps(full_dict)
    f = open(f"{OUTPUT_DIR}/transcriptions/{split}.json", "w")
    f.write(json_dict)
    f.close()

    return full_dict


def resize_aspect_ratio(img, label):
    img_width, img_height = img.size
    img = img.resize((int(img_width * FIXED_HEIGHT), FIXED_HEIGHT))
    rescale_width = len(label) * CHAR_WIDTH
    img = img.resize((rescale_width, FIXED_HEIGHT), Image.Resampling.BILINEAR)
    return img


if __name__ == "__main__":
    raw_data_path = sys.argv[1]
    train_gt_path = sys.argv[2]
    test_gt_path = sys.argv[3]

    OUTPUT_DIR = Path("./data/preprocess/IAM")
    OUTPUT_DIR.joinpath("writers_dict").mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.joinpath("transcriptions").mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.joinpath("images").mkdir(exist_ok=True, parents=True)

    train_infos = parse_groundtruth(train_gt_path, split="train")
    test_infos = parse_groundtruth(test_gt_path, split="test")

    total_images = {img.stem: img for img in list(Path(raw_data_path).rglob("*.*g"))}

    ignore_transcripts = []
    print("before train infos", len(train_infos.keys()))
    for trans_idx, trans in train_infos.items():
        if trans["image"][:-4] not in total_images:
            ignore_transcripts.append(trans_idx)
            continue
        try:
            img = Image.open(total_images[trans["image"][:-4]])
        except PIL.UnidentifiedImageError:
            print(f"skip processing image {total_images[trans['image'][:-4]]}")
        resize_img = resize_aspect_ratio(img, trans["label"])
        resize_img.save(OUTPUT_DIR.joinpath("images", trans["image"]), "PNG")
    train_infos = {key: value for key, value in train_infos.items() if key not in ignore_transcripts}
    print("after train infos", len(train_infos.keys()))
    train_infos = json.dumps(train_infos)
    with open(f"{OUTPUT_DIR}/transcriptions/train.json", "w") as ftrain:
        ftrain.write(train_infos)

    ignore_transcripts = []
    print("before test infos", len(test_infos.keys()))
    for trans_idx, trans in test_infos.items():
        if trans["image"][:-4] not in total_images:
            ignore_transcripts.append(trans_idx)
            continue
        try:
            img = Image.open(total_images[trans["image"][:-4]])
        except PIL.UnidentifiedImageError:
            print(f"skip processing image {total_images[trans['image'][:-4]]}")
        resize_img = resize_aspect_ratio(img, trans["label"])
        resize_img.save(OUTPUT_DIR.joinpath("images", trans["image"]), "PNG")
    test_infos = {key: value for key, value in test_infos.items() if key not in ignore_transcripts}
    print("after test infos", len(test_infos.keys()))
    test_infos = json.dumps(test_infos)
    with open(f"{OUTPUT_DIR}/transcriptions/test.json", "w") as ftest:
        ftest.write(test_infos)
