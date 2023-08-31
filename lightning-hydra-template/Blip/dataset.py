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

        csv_path = os.path.join(data, "pipe_test.csv")
        data = pd.read_csv(csv_path)
        self.data = data[data["type"] == data_type].reset_index(drop=True)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name ,img_path,_,comments,type = self.data.iloc[idx]
        img = Image.open(os.path.join(self.data_path,img_path))

        encoding = self.processor(
            images=img,
            text=comments,
            padding="max_length",
            return_tensors="pt",
        )
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        eocoding = encoding.update({"img_name" : img_name})
        return encoding
