from torch.utils.data import Dataset
from transformers import AutoProcessor
from PIL import Image
import pandas as pd
import os


class ImageCaptioningDataset(Dataset):
    def __init__(self, data, processor, data_type):
        self.data_path = data
        self.type = data_type
        self.processor = AutoProcessor.from_pretrained(processor)

        csv_path = os.path.join(data, "train.csv")
        data = pd.read_csv(csv_path)
        self.data = data[data["type"] == data_type].reset_index(drop=True)
        if data_type == "valid":
            self.data = self.data.iloc[
                self.data.img_name.drop_duplicates().index
            ].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, img_path, _, comments, _, img_id = self.data.iloc[idx]
        img = os.path.join(self.data_path, img_path)

        return {"img": img, "text": comments, "img_id": img_id}
