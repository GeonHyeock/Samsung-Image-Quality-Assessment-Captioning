import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def main(score_dict):
    data = []
    for path, score in score_dict.items():
        d = pd.read_csv(path)
        d["score"] = score
        data.append(d)
    data = pd.concat(data)

    img_names, comment, mos, i = [], [], [], 0
    for img_name in tqdm(data.img_name.unique()):
        d = data[data.img_name == img_name].loc[:, ["comments", "score"]]

        result = defaultdict(int)
        for idx in range(len(d)):
            c, s = d.iloc[idx]
            result[c] += s
        img_names += [img_name]
        comment += [max(result, key=result.get)]
        mos += [-1]

        if len(result) < len(score_dict):
            i += 1
    print("ensemble N : ", i)

    return pd.DataFrame({"img_name": img_names, "comments": comment, "mos": mos})


if __name__ == "__main__":
    score_dict = {
        "ensemble_csv/base_diffusion.csv": 1.276742618,
        "ensemble_csv/large_beamserch.csv": 1.273379889,
        "ensemble_csv/base_diffusion_lora4.csv": 1.256808295,
        "ensemble_csv/large_lora4.csv": 1.247990337,
        "ensemble_csv/large_lora_4_diffusion.csv": 1.24555302,
        "ensemble_csv/base_diffusion_lora_8.csv": 1.234590897,
        # "ensemble_csv/base_fp16all.csv": 1.2036627085,
        # "ensemble_csv/base - diffusion + text train.csv": 1.183253507,
        # "ensemble_csv/large-diffusion-lora_4-train_text_decoder.csv": 1.153261951,
    }

    df = main(score_dict)
    df.to_csv("voting.csv", index=False)

    pass
