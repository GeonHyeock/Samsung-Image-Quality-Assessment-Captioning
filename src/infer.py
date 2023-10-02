import pandas as pd
import numpy as np

data = pd.read_csv("test_caption.csv", index_col=False)
data = data.drop("img_path", axis=1)
data["mos"] = np.random.rand(len(data)) * 10
data = data[["img_name", "mos", "comments"]]
data.to_csv("./gh_submission.csv", index=False)
