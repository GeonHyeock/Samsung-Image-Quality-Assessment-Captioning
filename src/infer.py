import pandas as pd
import numpy as np

data = pd.read_csv("test_caption.csv", index_col=False)
mos_data = pd.read_csv("submit_maniqa_384_base_fold0_0921.csv")
median = mos_data.mos.median()
mos_data["mos"] = ((mos_data.mos - median) * 1.5) + median
data = data.drop("img_path", axis=1)

data = pd.merge(data[["img_name", "comments"]], mos_data[["img_name", "mos"]])
data = data[["img_name", "mos", "comments"]]
data.to_csv("./gh_submission.csv", index=False)
