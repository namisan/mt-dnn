# Base image must at least have pytorch and CUDA installed.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:20.10-py3
FROM $BASE_IMAGE
ARG BASE_IMAGE
ENV DEBIAN_FRONTEND=noninteractive
RUN echo "Installing Apex on top of ${BASE_IMAGE}"
RUN apt-get update && apt-get install -y \
sudo \
make \
vim \
jq \
locate \
wget \
tar \
bzip2 \
sudo \
environment-modules \
libhwloc-dev \
hwloc \
libhwloc-common \
libhwloc-plugins \
openssh-server \
binutils \
tcl \
curl \
ipmitool \
rename \
libibverbs-dev

RUN pip install --upgrade pip
RUN pip install tensorboard_logger
RUN pip install tqdm
RUN pip install h5py==2.7.1
RUN pip install -U scikit-learn
RUN pip install nltk>=3.4
RUN pip install sentencepiece
RUN python -m nltk.downloader punkt
RUN pip install numpy>=1.15.4
RUN pip install pandas>=0.24.0
RUN pip install numpy
RUN pip install colorlog
RUN pip install regex
RUN pip install pyyaml
RUN pip install pytest
RUN pip install boto3
RUN pip install tb-nightly
RUN pip install seqeval==0.0.12
RUN pip install transformers==4.6.0
RUN pip install tensorboardX
RUN pip install pytorch-pretrained-bert==v0.6.0
RUN pip install more_itertools
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pwd
WORKDIR /workspace
