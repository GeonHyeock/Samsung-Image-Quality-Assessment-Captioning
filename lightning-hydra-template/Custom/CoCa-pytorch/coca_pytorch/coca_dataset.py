import os
import cv2
import pandas as pd
import albumentations as A

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class CocaDataset(Dataset):
    def __init__(self, data_path, type, *trs):
        self.data_path = data_path
        self.transform = A.Compose(list(*trs), p=0.5)

        csv_path = os.path.join(data_path, "train.csv")
        data = pd.read_csv(csv_path)
        self.data = data[data["type"] == type]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, img_path, mos, comments, _ = self.data.iloc[idx]
        img = cv2.imread(os.path.join(self.data_path, img_path))
        img = self.transform(image=img)["image"]

        return {"img": img, "mos": mos, "text": comments}


if __name__ == "__main__":
    trs = [A.HorizontalFlip(p=0.5), ToTensorV2()]
    dataset = CocaDataset("data", "train", trs)
    dataset[2]
