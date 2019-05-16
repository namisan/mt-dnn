# Multi-Task Deep Neural Networks for Natural Language Understanding

This PyTorch package implements the Multi-Task Deep Neural Networks (MT-DNN) for Natural Language Understanding, as described in:

Xiaodong Liu\*, Pengcheng He\*, Weizhu Chen and Jianfeng Gao<br/>
Multi-Task Deep Neural Networks for Natural Language Understanding<br/>
[arXiv version](https://arxiv.org/abs/1901.11504) <br/>
\*: Equal contribution <br/>

Xiaodong Liu, Pengcheng He, Weizhu Chen and Jianfeng Gao<br/>
Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding <br/>
[arXiv version](https://arxiv.org/abs/1904.09482) <br/>


## Quickstart 

### Setup Environment
#### Install via pip:
1. python3.6

2. install requirements </br>
   ```> pip install -r requirements.txt```

#### Use docker:
1. Pull docker </br>
   ```> docker pull allenlao/pytorch-mt-dnn:v0.1```

2. Run docker </br>
   ```> docker run -it --rm --runtime nvidia  allenlao/pytorch-mt-dnn:v0.1 bash``` </br>
    Please refere the following link if you first use docker: https://docs.docker.com/

### Train a toy MT-DNN model
1. Download data </br>
   ```> sh download.sh``` </br>
   Please refer to download GLUE dataset: https://gluebenchmark.com/

2. Preprocess data </br>
   ```> python prepro.py```

3. Training </br>
   ```> python train.py```

**Note that we ran experiments on 4 V100 GPUs for base MT-DNN models. You may need to reduce batch size for other GPUs.** <br/>

### GLUE Result reproduce
1. MTL refinement: refine MT-DNN (shared layers), initialized with the pre-trained BERT model, via MTL using all GLUE tasks excluding WNLI to learn a new shared representation. </br>
**Note that we ran this experiment on 8 V100 GPUs (32G) with a batch size of 32.**
   + Preprocess GLUE data via the aforementioned script
   + Training: </br>
   ```>scripts\run_mt_dnn.sh```

2. Finetuning: finetune MT-DNN to each of the GLUE tasks to get task-specific models. </br>
Here, we provide two examples, STS-B and RTE. You can use similar scripts to finetune all the GLUE tasks. </br>
   + Finetune on the STS-B task </br>
   ```> scripts\run_stsb.sh``` </br>
   You should get about 90.5/90.4 on STS-B dev in terms of Pearson/Spearman correlation. </br>
   + Finetune on the RTE task  </br>
   ```> scripts\run_rte.sh``` </br>
   You should get about 83.8 on RTE dev in terms of accuracy. </br>

### SciTail & SNIL Result reproduce (Domain Adaptation)
1. Domain Adaptation on SciTail  </br>
   ```>scripts\scitail_domain_adaptation_bash.sh```

2. Domain Adaptation on SNLI </br>
  ```>scripts\snli_domain_adaptation_bash.sh```

### TODO
[ ] Release codes/models MT-DNN with Knowledge Distillation. </br>
[ ] Publish pretrained Tensorflow checkpoints.

## FAQ

### Do you shared the pretrained mt-dnn models?
Yes, we released the pretrained shared embedings via MTL which are aligned to BERT base/large models: ```mt_dnn_base.pt``` and ```mt_dnn_large.pt```. </br>
To obtain the similar models:
1. run the ```>sh scripts\run_mt_dnn.sh```, and then pick the best checkpoint based on the average dev preformance of MNLI/RTE. </br>
2. strip the task-specific layers via ```scritps\strip_model.py```. </br>

### Why SciTail/SNLI do not enable SAN?
For SciTail/SNLI tasks, the purpose is to test generalization of the learned embedding and how easy it is adapted to a new domain instead of complicated model structures for a direct comparison with BERT. Thus, we use a linear projection on the all **domain adaptation** settings.

### What is the difference between V1 and V2
The difference is in the QNLI dataset. Please refere to the GLUE official homepage for more details. 

### Do you fine-tune single task for your GLUE leaderboard submission? 
We can use the multi-task refinement model to run the prediction and produce a reasonable result. But to achieve a better result, it requires a fine-tuneing on each task. It is worthing noting the paper in arxiv is a littled out-dated and on the old GLUE dataset. We will update the paper as we mentioned below. 


## Notes and Acknowledgments
BERT pytorch is from: https://github.com/huggingface/pytorch-pretrained-BERT <br/>
BERT: https://github.com/google-research/bert <br/>
We also used some code from: https://github.com/kevinduh/san_mrc <br/>

### How do I cite MT-DNN?

For now, please cite [arXiv version](https://arxiv.org/abs/1901.11504):

```
@article{liu2019mt-dnn,
  title={Multi-Task Deep Neural Networks for Natural Language Understanding},
  author={Liu, Xiaodong and He, Pengcheng and Chen, Weizhu and Gao, Jianfeng},
  journal={arXiv preprint arXiv:1901.11504},
  year={2019}
}

and a new version of the paper will be shared later.

@article{liu2019mt-dnn-kd,
  title={Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding},
  author={Liu, Xiaodong and He, Pengcheng and Chen, Weizhu and Gao, Jianfeng},
  journal={arXiv preprint arXiv:1904.09482},
  year={2019}
}
```
***Typo:*** there is no activation fuction in Equation 2. 
### Contact Information

For help or issues using MT-DNN, please submit a GitHub issue.

For personal communication related to MT-DNN, please contact Xiaodong Liu (`xiaodl@microsoft.com`), Pengcheng He (`penhe@microsoft.com`), Weizhu Chen (`wzchen@microsoft.com`) or Jianfeng Gao (`jfgao@microsoft.com`).
