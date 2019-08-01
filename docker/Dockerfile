FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN pip install tqdm
RUN pip install h5py==2.7.1
RUN pip install boto3
RUN pip install -U scikit-learn
## install pytorch bert
RUN pip install pytorch-pretrained-bert==v0.6.0
RUN pip install sentencepiece

## GLUE baseline dependencies
RUN pip install nltk
RUN pip install ipdb
RUN pip install tensorboardX

## others
RUN pip install tensorboard
RUN pip install future
RUN pip install regex

RUN apt-get update && apt-get install -y wget unzip
