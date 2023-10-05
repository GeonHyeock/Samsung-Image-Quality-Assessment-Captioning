# 2023 Samsung AI Challenge : Image Quality Assessment - Captioning

- 카메라 영상 화질 정량 평가 및 자연어 정성 평가를 동시 생성하는 알고리즘 개발

- 종합 등수 : 6th
- [화질 정량 평가 code](https://github.com/lliee1/Samsung-Image-Quality-Assessment-Mos)

## Member🔥
| [박주용](https://github.com/lliee1)| [허건혁](https://github.com/GeonHyeock) |
| :-: | :-: |
| <img src="https://avatars.githubusercontent.com/lliee1" width="100"> | <img src="https://avatars.githubusercontent.com/GeonHyeock" width="100"> |
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

---

2. 환경설정\
- [docker file](Dockerfile) OR [Conda]() 실행 \
docker file은 다음과 같은 설정으로 작성되어 있습니다. \
GeForce RTX 3080 Ti : 2EA

    |구분|env|
    |:---:|:---:|
    |OS|Linux-5.11.0-43-generic-x86_64-with-glibc2.31|
    |CUDA|11.2.2|
    |CUDNN|8|

~~~md
# 1-1 use Dokcer
docker build -t caption .
docker run --gpus all --ipc=host -it caption /bin/bash

# 1-2  use Conda
conda create -n captioning python=3.10.9
~~~

~~~md
# 2 requirement install
sh requirement.sh
~~~

# 
3. 명령문

~~~md
# Data preprocessing
mv data/train.csv data/raw_train.csv
python src/data_processing.py 

# Make Diffusion image
python src/text2image.py --diffusion_N={int - default : 8000}

# Data EDA
streamlit run src/data_eda.py
~~~


~~~md
# Model Train - ckpt 저장 경로 : lightning-hydra-template/logs 
cd lightning-hydra-template
python src/train.py {args}

# Model Inference : test-caption.csv 생성
cd lightning-hydra-template
git checkout feat/infer
python src eval.py ckpt_path={ckpt} {agrs}
~~~
- args는 [config](lightning-hydra-template/configs)에서 설정할 수 있습니다 : [args_example](Reproduct.lua)
- data.use_diffusion=true를 사용하기 위해서는 [Diffusion image](src/text2image.py)를 train size의 15%이상 생성해야함
 : 참조 : [Dataset](lightning-hydra-template/Blip/dataset.py)

~~~md
# gh-submission 생성
python src/infer.py

# gh-submission 결과들로부터 voting.csv 생성
python src/ensemble.py
~~~


---
### Score Ranking
|Type|score|Rank|
| :---: | :---: | :---: |
| Public | 1.35946 | 4 |
| Private | 0.8592 | 11 |
---


### [Report](Report.pdf)