# 2023 Samsung AI Challenge : Image Quality Assessment - Captioning

- 카메라 영상 화질 정량 평가 및 자연어 정성 평가를 동시 생성하는 알고리즘 개발

- 종합 등수 : 6th
- [화질 정량 평가 code]()

## Member🔥
| [박주용](https://github.com/lliee1)| [허건혁](https://github.com/GeonHyeock) |
| :-: | :-: |
| <img src="https://avatars.githubusercontent.com/OZOOOOOH" width="100"> | <img src="https://avatars.githubusercontent.com/GeonHyeock" width="100"> |
***


## Index
* [Competition imformation](#competition-imformation)
* [Code reproduction](#code-reproduction)
***

### Competition imformation

- 주관 : 삼성전자 SAIT
- 운영 : 데이콘
- 대회 : [link](https://dacon.io/competitions/official/236134/overview/description)

대회 기간 : 2023-08-21-11:00 ~ 2023-10-02-10:00

Input : 사진 \
Output : 자연어 기반 정성 평가 캡셔닝(Text output, 영어)

평가 산식 : $CIDEr-D * 4 + METEOR * 3 + ((BLEU-4 + BLEU-3) / 2) * 2 + ROUGE-L * 1$

---

### Code reproduction

1. [raw_data](https://dacon.io/competitions/official/236134/data)를 [data folder](data)에 저장 

~~~md
# Data preprocessing
mv data/train.csv data/raw_train.csv
python src/data_processing.py 

# Make Diffusion image
python src/text2image.py --diffusion_N={int}

# Data EDA
streamlit run src/data_eda.py
~~~

~~~md
# Model Train
cd lightning-hydra-template
python src/train.py {args}

# Model Inference
cd lightning-hydra-template
python src eval.py {agrs}
~~~

---

### Score Ranking
|Type|score|Rank|
| :---: | :---: | :---: |
| Public | 1.35946 | 4 |
| Private | 0.8592 | 11 |
---

