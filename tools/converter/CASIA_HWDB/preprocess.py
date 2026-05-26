import sys
import struct
import codecs
import numpy as np
import cv2
import os
from PIL import Image
from pathlib import Path
import json
import random


def decode_GNT_to_imgs(gnt):
    """
    :param gnt: a writer's encoded ground truth file.
    :return:    samples list, each sample with format (charname, img)
    """
    samples = []
    with codecs.open(gnt, mode="rb") as fin:
        while True:
            left_cache = fin.read(4)
            if len(left_cache) < 4:
                break
            sample_size = struct.unpack("I", left_cache)[0]
            tag_code = str(fin.read(2), "gbk")
            width = struct.unpack("H", fin.read(2))[0]
            height = struct.unpack("H", fin.read(2))[0]

            img = np.zeros(shape=[height, width], dtype=np.uint8)
            for r in range(height):
                for c in range(width):
                    img[r, c] = struct.unpack("B", fin.read(1))[0]
            if width * height + 10 != sample_size:
                break
            samples.append((tag_code[0], img))

    return samples


def process(w_ids, raw_data_path, split):
    """
    解析并保存HWDB-1的gnt格式数据到对应类别的图像文件夹
    :param gnt_root_dir:
    :param img_save_dir:
    :param labelmap_ori: 原始字码表（若是gbk编码的，需要先利用transform_labelmap_to_utf8
                         转换到utf-8编码）, 其中的‘.’和‘/’尚未以dot和slash表示
    :return:
    """
    index_dict = {}
    transcriptions = {}
    writer_ids = {}
    transcript_idx = 0
    writer_idx = 0

    for writer_id in w_ids:
        save_dir = OUTPUT_DIR.joinpath("images").joinpath(writer_id)

        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        gnt_path = os.path.join(raw_data_path, writer_id)
        samples = decode_GNT_to_imgs(gnt_path)

        for sample in samples:
            char_name = sample[0]
            if char_name == ".":
                char_name = "dot"  # 避免文件夹路径冲突
            if char_name == "/":
                char_name = "slash"  # 避免文件夹路径冲突

            if char_name not in index_dict:
                index_dict.update({char_name: 0})
            else:
                index_dict[char_name] += 1

            save_img_path = save_dir.joinpath(char_name)
            if not save_img_path.exists():
                save_img_path.mkdir(parents=True)

            img_save_path = save_img_path.joinpath(str(index_dict[char_name]) + ".jpg")
            img_name = img_save_path.relative_to(OUTPUT_DIR.joinpath("images"))
            transcriptions[transcript_idx] = {
                "image": img_name.as_posix(),
                "s_id": writer_id,
                "label": img_name.parent.name,
            }
            transcript_idx += 1

            if writer_id not in writer_ids:
                writer_ids[writer_id] = writer_idx
                writer_idx += 1

            resize_img = resize_image(sample[1])
            cv2.imwrite(
                os.path.join(str(save_img_path), str(index_dict[char_name]) + ".jpg"),
                resize_img,
            )

        print("processed gnt file %s." % gnt_path)

    # create json object from dictionary if you want to save writer ids
    json_dict = json.dumps(writer_ids)
    f = open(f"{OUTPUT_DIR}/writers_dict/{split}.json", "w")
    f.write(json_dict)
    f.close()

    json_dict = json.dumps(transcriptions)
    f = open(f"{OUTPUT_DIR}/transcriptions/{split}.json", "w")
    f.write(json_dict)
    f.close()


def resize_image(img):
    # pad to square
    pad_size = int(abs(img.shape[0] - img.shape[1]) / 2)
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    img = np.lib.pad(img, pad_dims, mode="constant", constant_values=255)
    # resize
    img = Image.fromarray(img)
    img = img.resize((64, 64), Image.Resampling.BILINEAR)
    assert img.size == (64, 64)
    img = np.asarray(img).astype("uint8")
    resize_ = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return resize_


if __name__ == "__main__":
    raw_data_path = sys.argv[1]
    OUTPUT_DIR = Path("./data/preprocess/CASIA_HWDB")
    OUTPUT_DIR.joinpath("writers_dict").mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.joinpath("transcriptions").mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.joinpath("images").mkdir(exist_ok=True, parents=True)

    total_w_ids = list(Path(raw_data_path).glob("*"))
    total_w_ids = list(map(lambda f: f.name, total_w_ids))

    random.shuffle(total_w_ids)

    train_wid = total_w_ids[: int(0.8 * len(total_w_ids))]
    test_wid = total_w_ids[int(0.8 * len(total_w_ids)) :]

    process(train_wid, raw_data_path, "train")
    process(test_wid, raw_data_path, "test")
