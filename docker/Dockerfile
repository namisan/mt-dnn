FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
RUN apt-get clean && apt-get update && apt-get install -y locales
ENV  LANG="en_US.UTF-8" LC_ALL="en_US.UTF-8" LANGUAGE="en_US.UTF-8" LC_TYPE="en_US.UTF-8" TERM=xterm-256color
RUN locale-gen en_US en_US.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         zip \
         wget \
         unzip \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*


ENV PYTHON_VERSION=3.6

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy=1.14.5 scipy ipython mkl&& \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH

RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION cuda90 pytorch=0.4.1 torchvision -c pytorch && \
    /opt/conda/bin/conda clean -ya
RUN pip install --upgrade pip
RUN pip install tensorboard_logger
RUN pip install tqdm
RUN pip install h5py==2.7.1
RUN pip install boto3
RUN pip install -U scikit-learn
# install pytorch bert
RUN pip install pytorch-pretrained-bert==v0.6.0

# GLUE baseline dependencies
RUN pip install nltk
RUN pip install allennlp==0.4
RUN pip install ipdb
RUN pip install tensorboardX

WORKDIR /root
#COPY requirements.txt /root/
#RUN pip install -r requirements.txt
