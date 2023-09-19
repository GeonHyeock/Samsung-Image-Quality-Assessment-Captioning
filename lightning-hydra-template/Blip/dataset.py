from torch.utils.data import Dataset
import nlpaug.augmenter.word as naw
import random
import pandas as pd
import os


class ImageCaptioningDataset(Dataset):
    def __init__(self, data, data_type):
        self.data_path = data
        self.type = data_type
        if data_type == "test":
            csv_path = os.path.join(data, "test.csv")
            self.data = pd.read_csv(csv_path)

        else:
            csv_path = os.path.join(data, "train.csv")
            data = pd.read_csv(csv_path)
            self.data = data[data["type"] == data_type].reset_index(drop=True)

            if data_type == "train":
                diffusion = self.data[
                    self.data.img_name.apply(lambda x: "diffusion" in x)
                ].sample(5968 + 4096)

                self.data = pd.concat(
                    [
                        self.data[
                            self.data.img_name.apply(lambda x: "diffusion" not in x)
                        ],
                        diffusion,
                    ]
                ).reset_index(drop=True)

            if data_type == "valid":
                self.data = self.data.iloc[
                    self.data.img_name.drop_duplicates().index
                ].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = dict(self.data.iloc[idx])
        img = os.path.join(self.data_path, d["img_path"])

        if self.type == "train":
            return {"img": img, "img_id": d["image_id"], "text": d["comments"]}
        elif self.type == "valid":
            return {"img": img, "img_id": d["image_id"]}
        elif self.type == "test":
            return {"img": img}
