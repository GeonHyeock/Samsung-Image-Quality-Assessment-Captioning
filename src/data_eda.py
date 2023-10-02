import pandas as pd
import streamlit as st
import numpy as np
import os
import glob
import random
from PIL import Image


def visual_img(d, img_name):
    data = d[d.img_name == img_name]
    img = Image.open(os.path.join("data", data.img_path.unique()[0]))
    st.image(img)
    if "mos" in data.columns:
        st.write(f"mos : {data.mos.unique()[0]}")

    if "comments" in data.columns:
        for idx, value in enumerate(data.comments):
            st.write(f"comments {idx} : {value}")


def main():
    st.title("Samsung Challenge Data EDA")

    with st.sidebar:
        csv = st.selectbox("시각화 할 csv", glob.glob("data/*.csv"))
        
        raw_data = pd.read_csv(csv)
        unique_img = raw_data.img_name.unique()

        img_name = st.selectbox("이미지 이름 선택", unique_img)

        button = st.button("Random_choice", type="primary")
        if button:
            img_name = random.choice(unique_img)

    visual_img(raw_data, img_name)


if __name__ == "__main__":
    main()
