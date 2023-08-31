import pandas as pd
import numpy as np
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    np.random.seed(42)

    data = pd.read_csv("data/train.csv")
    train_img = np.random.choice(data.img_name.unique(), int(len(data) * 0.9))
    train_idx = np.where(np.isin(data.img_name, train_img) == True)[0]
    valid_idx = np.array(list(set(range(len(data))) - set(train_idx)))
    print(f"train : {len(train_idx)} valid : {len(valid_idx)}")
    print(f"ratio = {len(train_idx) / len(data)}")

    data["type"] = pd.Series()
    data.loc[train_idx, "type"] = "train"
    data.loc[valid_idx, "type"] = "valid"

    comments_dict = defaultdict(list)
    for i in range(len(data)):
        img_name, _, _, comments, _ = data.iloc[i]
        comments_dict[img_name] += [comments]
    with open("./data/comments_dict.json", "w") as f:
        json.dump(comments_dict, f, indent=4)
    data.to_csv("data/train.csv", index=False)
