import pandas as pd
import numpy as np
import json
import os
from PIL import Image


def train_valid_split():
    data = pd.read_csv("data/raw_train.csv")
    img_names = data.img_name.unique()
    val_img_name = data.img_name.value_counts()[data.img_name.value_counts() == 1].index
    valid_img = np.random.choice(val_img_name, 15000, replace=False)
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


if __name__ == "__main__":
    np.random.seed(42)
    train_valid_split()
    make_coco_json()
