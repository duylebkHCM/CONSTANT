import sys
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [10, 8]
from src.data.constant import CHAR_WIDTH, FIXED_HEIGHT, MAX_LENGTH

ENG_VOCAB = open(
    Path(__file__).absolute().parent.parent.parent.parent / "data" / "vocab" / "eng.txt",
    encoding="utf-8",
).read()


def load_json(path):
    with open(path, mode="r", encoding="utf-8") as f:
        json_dict = json.load(f)
        f.close()
    return json_dict


def write_json_file(data_dict: dict, path) -> None:
    with open(path, mode="w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)
        f.close()


def read_multi_lines(path):
    with open(path, mode="r", encoding="utf-8") as f:
        content = f.readlines()
        f.close()
    content = [x.strip() for x in content]
    return content


def read_file(path):
    with open(path, mode="r", encoding="utf-8") as f:
        data = f.read()
        f.close()
    return data


def ensure_dir(p_dir: Path):
    if not p_dir.exists():
        p_dir.mkdir(exist_ok=True, parents=True)


def is_oov(text):
    return any([c not in ENG_VOCAB for c in list(text)])


def resize_aspect_ratio(img, label):
    img_width, img_height = img.size
    img = img.resize((int(img_width * FIXED_HEIGHT), FIXED_HEIGHT))
    rescale_width = len(label) * CHAR_WIDTH
    img = img.resize((rescale_width, FIXED_HEIGHT), Image.Resampling.BILINEAR)
    return img


def process(root_dir, split="train"):
    counter = 0
    transcription_infos = dict()

    writer_infos = dict()
    writer_idx_count = defaultdict(int)
    writer_count = 0

    img_dir = root_dir.joinpath(f"imgur5k_{split}_0_None")
    ground_truth_path = root_dir.joinpath(f"imgur5k_{split}_0_None", "gt.txt")
    lst_data = read_multi_lines(ground_truth_path)

    for line_info in tqdm(lst_data):
        try:
            img_path, word_label = line_info.split("\t")

            if len(word_label) > MAX_LENGTH:
                continue

            if is_oov(word_label):
                continue

            w_id = Path(img_path).stem.split("_")[0]

            if w_id not in writer_infos:
                writer_infos[str(w_id)] = writer_count
                writer_count += 1

            img_path = img_dir.joinpath(img_path)

            word_img = Image.open(img_path).convert("RGB")
            resized_img = resize_aspect_ratio(word_img, word_label)

            save_img_path = f"{OUTPUT_DIR}/images/IMGUR5K_{w_id}_{writer_idx_count[str(w_id)]}.png"

            trans_info = {
                "image": f"IMGUR5K_{w_id}_{writer_idx_count[str(w_id)]}.png",
                "s_id": str(w_id),
                "label": word_label,
            }
            writer_idx_count[str(w_id)] += 1

            transcription_infos[str(counter)] = trans_info
            counter += 1

            resized_img.save(save_img_path, "PNG")

        except Exception:
            print(f"Error => {line_info}")

    trans_name = f"{OUTPUT_DIR}/transcriptions/{split}.json"
    write_json_file(transcription_infos, trans_name)

    writer_name = f"{OUTPUT_DIR}/writers_dict/{split}.json"
    write_json_file(writer_infos, writer_name)


if __name__ == "__main__":
    raw_data_path = sys.argv[1]
    OUTPUT_DIR = Path("./data/preprocess/IMGUR5K")
    OUTPUT_DIR.joinpath("writers_dict").mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.joinpath("transcriptions").mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.joinpath("images").mkdir(exist_ok=True, parents=True)

    process(Path(raw_data_path), "train")
    process(Path(raw_data_path), "test")
