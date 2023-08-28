import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    np.random.seed(42)

    data = pd.read_csv("data/train.csv")
    data["type"] = pd.Series()

    idx = np.array(data.index.array)
    train_idx, valid_idx = train_test_split(idx, test_size=0.5, random_state=42)

    data.loc[train_idx, "type"] = "train"
    data.loc[valid_idx, "type"] = "valid"
    data.to_csv("data/train.csv")
