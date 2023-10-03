pip install --upgrade pip \
 && pip install -r ./lightning-hydra-template/requirements.txt \
 && pip install -r ./requirements.txt \
 && pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

apt-get update \
 && apt-get -y install libgl1-mesa-glx \
 && apt-get -y install openjdk-11-jdk \
 && apt-get install libglib2.0-0
