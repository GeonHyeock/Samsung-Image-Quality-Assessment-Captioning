import pandas as pd
import numpy as np
import json
import os
from PIL import Image
from collections import Counter


def train_valid_split():
    data = pd.read_csv("data/raw_train.csv")
    img_names = data.img_name.unique()
    val_img_name = data.img_name.value_counts()[data.img_name.value_counts() == 1].index
    valid_img = np.random.choice(val_img_name, 10000, replace=False)
    train_img = list(set(img_names) - set(valid_img))
    print(f"train : {len(train_img)} valid : {len(valid_img)}")

    if "type" in data.columns:
        data = data.drop("type", axis=1)

    pd.merge(
        data,
        pd.DataFrame(
            {
                "img_name": train_img + list(valid_img),
                "type": ["train"] * len(train_img) + ["valid"] * len(valid_img),
            }
        ),
    ).to_csv("data/train.csv", index=False)


def make_coco_json(data_folder="data", make_type=["valid"]):
    raw_data = pd.read_csv(os.path.join(data_folder, "train.csv"))
    raw_data["image_id"] = pd.Series()

    images_dict = {}
    for idx, value in enumerate(sorted(raw_data.img_path.unique())):
        W, H = Image.open(f"{os.path.join(data_folder,value)}").size
        file_name = value.split("/")[-1].split(".")[0]
        images_dict.update(
            {
                file_name: {
                    "file_name": file_name,
                    "width": W,
                    "height": H,
                    "id": idx + 1,
                }
            }
        )
        raw_data.loc[raw_data.img_path == value, "image_id"] = idx + 1

    for t in make_type:
        data = raw_data[raw_data["type"] == t]
        images = [images_dict[i] for i in data.img_name.unique()]

        annotations = []
        for idx in range(len(data)):
            img_name, _, _, caption, _, _ = data.iloc[idx]
            annotations.append(
                {
                    "image_id": images_dict[img_name]["id"],
                    "id": idx + 1,
                    "caption": caption,
                }
            )

        my_json = {"images": images, "annotations": annotations, "type": "captions"}

        with open(f"{os.path.join(data_folder,t)}.json", "w") as f:
            json.dump(my_json, f, indent=4)
        raw_data.to_csv("data/train.csv", index=False)


def find_duplicate_ngrams(sentences, n):
    ngrams = []
    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            ngrams.append(ngram)
    ngram_counts = Counter(ngrams)
    return ngram_counts


def make_weight(sentence, my_count):
    weight = 0
    for n in [2, 3, 4]:
        words = sentence.split()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            weight += my_count.get(ngram, 2)
    return len(words) / (np.log2(weight))


def make_weight_col():
    raw_data = pd.read_csv("data/train.csv")
    data = raw_data[raw_data.type == "train"]
    data.loc[:, "comments"] = data.comments.apply(
        lambda x: x.replace(". ", ".").replace(".", ". ")
    )
    ngram_counts_2 = find_duplicate_ngrams(data.comments.unique(), 2)
    ngram_counts_3 = find_duplicate_ngrams(data.comments.unique(), 3)
    ngram_counts_4 = find_duplicate_ngrams(data.comments.unique(), 4)

    my_count = {k: v for k, v in ngram_counts_2.items() if v > 1}
    my_count.update({k: v for k, v in ngram_counts_3.items() if v > 1})
    my_count.update({k: v for k, v in ngram_counts_4.items() if v > 1})

    comments = raw_data.comments.apply(
        lambda x: x.replace(". ", ".").replace(".", ". ")
    )
    raw_data["weight"] = [make_weight(x, my_count) for x in comments]
    raw_data.to_csv("data/train_weight.csv", index=False)
    pass


if __name__ == "__main__":
    np.random.seed(42)
    train_valid_split()
    make_coco_json()
    make_weight_col()
