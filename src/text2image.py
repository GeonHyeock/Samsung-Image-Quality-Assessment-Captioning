from diffusers import StableDiffusionPipeline
from collections import defaultdict
from tqdm import tqdm
import random
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import argparse


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def make_pipe(model_id="runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    return pipe


def main(args):
    train_data = pd.read_csv("data/train_weight.csv")
    img_id = train_data.image_id.max() + 1

    pipe = make_pipe()
    difussion_df = defaultdict(list)
    for _ in tqdm(range(int(args.diffusion_N))):
        data = train_data.sample(8, weights="weight")
        comments = list(data["comments"].values)
        image = pipe(comments).images
        for img, comment in zip(image, comments):
            if img.convert("RGB").resize((1, 1)).getpixel((0, 0)) != (0, 0, 0):
                img_name = f"diffusion_{str(img_id).zfill(6)}"
                img_path = f"./train/diffusion/{img_name}.jpg"

                difussion_df["img_name"] += [img_name]
                difussion_df["img_path"] += [img_path]
                difussion_df["mos"] += [-1]
                difussion_df["comments"] += [comment]
                difussion_df["type"] += ["train"]
                difussion_df["image_id"] += [img_id]

                img_id += 1
                img.save(os.path.join("data", img_path))

    difussion_df = pd.DataFrame(difussion_df)
    difussion_df.to_csv("data/difussion_df.csv", index=False)


def make_df():
    train_df = pd.read_csv("data/train_weight.csv")
    difussion_df = pd.read_csv("data/difussion_df.csv")
    pd.concat([train_df, difussion_df]).to_csv("data/difussion_train.csv", index=False)


if __name__ == "__main__":
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    createDirectory("data/train/diffusion")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diffusion_N", default=8000, help="Diffusion 이미지 개수 : 최대 N * 8"
    )
    args = parser.parse_args()

    main(args)
    make_df()
    pass
