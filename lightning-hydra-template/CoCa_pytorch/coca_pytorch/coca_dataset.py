import os
import cv2
import pandas as pd
import albumentations as A

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class CocaDataset(Dataset):
    def __init__(self, data_dir, data_type, trs):
        self.data_dir = data_dir
        self.transform = trs
        self.data_type = data_type

        csv_path = os.path.join(data_dir, "pipe_test.csv")
        data = pd.read_csv(csv_path)
        self.data = data[data["type"] == data_type].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, img_path, mos, comments, _ = self.data.iloc[idx]
        img = cv2.imread(os.path.join(self.data_dir, img_path))
        img = self.transform(image=img)["image"]

        if self.data_type == "train":
            comments = comments + "[MASK]"

        elif self.data_type == "valid":
            comments = img_name

        return {"img": img, "mos": mos, "text": comments}


def trs_train():
    trs = [A.HorizontalFlip(), ToTensorV2()]
    return A.Compose([A.Resize(256, 256), A.Normalize()] + trs, p=1)


def trs_valid():
    trs = [ToTensorV2()]
    return A.Compose([A.Resize(256, 256), A.Normalize()] + trs, p=1)


if __name__ == "__main__":
    dataset = CocaDataset("data", "valid", trs_train())
    dataset[2]
