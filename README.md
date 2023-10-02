# 2023 Samsung AI Challenge : Image Quality Assessment - Captioning

## 배경
카메라로 촬영된 영상의 화질에 대한 결과를 자연어로 상세하게 표현하는 알고리즘을 개발합니다.

화질 평가는 모두가 동의하는 절대적인 기준이 없고, 영상의 선명도, 노이즈 정도, 색감, 선호도 등 다양한 인지 화질 요소를 종합적으로 고려 해야 하는 Challenging 한 문제입니다.

다양한 인지 화질 요소에 대한 평가를 단일 점수로 나타낼 순 있으나 많은 의미가 생략되게 됩니다. 따라서, 새로운 화질 평가 연구의 한 방향으로 자연어로 상세히 영상의 화질을 설명할 수 있는 기술이 필요합니다.

이 기술은 향후 스마트폰 카메라에서 개인별, 상황별, 국가 별로 특성화되어 사용자에게 최고의 화질을 제공할 수 있는 AI 영상 처리 기술 개발에 활용될 예정입니다.

## 주제
사진 입력에 대해서 자연어 기반 정성 평가 캡셔닝을 생성하는 AI 모델을 개발

대회 기간 : 2023년 08월 21일 11:00 ~ 2023년 10월 02일 10:00

Input : 사진 \
Output : 자연어 기반 정성 평가 캡셔닝(Text output, 영어)

평가 산식 : $CIDEr-D * 4 + METEOR * 3 + ((BLEU-4 + BLEU-3) / 2) * 2 + ROUGE-L * 1$

## Code reproduction
1. [raw_data](https://dacon.io/competitions/official/236134/data)를 [data folder](data)에 저장 

~~~md
# Data preprocessing
mv data/train.csv data/raw_train.csv
python src/data_processing.py 

# Make Diffusion image
python src/text2image.py
~~~

~~~md
# Data EDA
streamlit run src/data_eda.py
~~~

~~~md
# Model Train
cd lightning-hydra-template
python src/train.py {args}
~~~

~~~md
# Model Inference
cd lightning-hydra-template
python src eval.py {agrs}
~~~


