pip install --upgrade pip \
 && pip install -r ./lightning-hydra-template/requirements.txt \
 && pip install -r ./requirements.txt \
 && pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

sudo apt-get update \
 && sudo apt-get -y install openjdk-11-jdk \
 && sudo apt-get -y install libglib2.0-0 \
 && sudo apt-get -y install libgl1-mesa-glx