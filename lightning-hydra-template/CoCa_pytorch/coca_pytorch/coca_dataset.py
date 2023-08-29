import os
import cv2
import pandas as pd
import albumentations as A

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class CocaDataset(Dataset):
    def __init__(self, data_dir, data_type, *trs):
        self.data_dir = data_dir
        self.transform = A.Compose(list(*trs), p=0.5)

        csv_path = os.path.join(data_dir, "train.csv")
        data = pd.read_csv(csv_path)
        self.data = data[data["type"] == data_type].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, img_path, mos, comments, _ = self.data.iloc[idx]
        img = cv2.imread(os.path.join(self.data_dir, img_path))
        img = self.transform(image=img)["image"]

        return {"img": img, "mos": mos, "text": comments}


if __name__ == "__main__":
    trs = [A.HorizontalFlip(p=0.5), ToTensorV2()]
    dataset = CocaDataset("data", "train", trs)
    dataset[2]
