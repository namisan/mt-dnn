**New Release ** <br/>
We are working on the v0.2 version which is easier to adapt it to new tasks. <br/>
You are welcomed to test our new version. <br/>
If you want to use the old version, please use following cmd to clone the code: <br/>
```git clone -b v0.1 https://github.com/namisan/mt-dnn.git ```


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
1. python3.6 </br>
   Reference to download and install : https://www.python.org/downloads/release/python-360/

2. install requirements </br>
   ```> pip install -r requirements.txt```

#### Use docker:
1. Pull docker </br>
   ```> docker pull allenlao/pytorch-mt-dnn:v0.1```

2. Run docker </br>
   ```> docker run -it --rm --runtime nvidia  allenlao/pytorch-mt-dnn:v0.1 bash``` </br>
   Please refer to the following link if you first use docker: https://docs.docker.com/

### Train a toy MT-DNN model
1. Download data </br>
   ```> sh download.sh``` </br>
   Please refer to download GLUE dataset: https://gluebenchmark.com/

2. Preprocess data </br>
   ```> sh experiments/glue/prepro.sh```

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


### Extract embeddings
1. Extracting embeddings of a pair text example </br>
   ```>python extractor.py --do_lower_case --finput input_examples\pair-input.txt --foutput input_examples\pair-output.json --bert_model bert-base-uncased --checkpoint mt_dnn_models\mt_dnn_base.pt``` </br>
   Note that the pair of text is split by a special token ```|||```. You may refer ``` input_examples\pair-output.json``` as example. </br>

2. Extracting embeddings of a single sentence example </br>
   ```>python extractor.py  --do_lower_case --finput input_examples\single-input.txt --foutput input_examples\single-output.json --bert_model bert-base-uncased --checkpoint mt_dnn_models\mt_dnn_base.pt``` </br>


### Convert Tensorflow BERT model to the MT-DNN format
Here, we go through how to convert a Chinese Tensorflow BERT model into mt-dnn format. <br/>
1. Download BERT model from the Google bert web: https://github.com/google-research/bert <br/>

2. Run the following script for MT-DNN format</br>
   ```python scripts\convert_tf_to_pt.py --tf_checkpoint_root chinese_L-12_H-768_A-12\ --pytorch_checkpoint_path chinese_L-12_H-768_A-12\bert_base_chinese.pt```

### TODO
- [x] MT-DNN with Knowledge Distillation code and model. <br/>
- [ ] Publish pretrained Tensorflow checkpoints.

## FAQ

### Did you share the pretrained mt-dnn models?
Yes, we released the pretrained shared embedings via MTL which are aligned to BERT base/large models: ```mt_dnn_base.pt``` and ```mt_dnn_large.pt```. </br>
To obtain the similar models:
1. run the ```>sh scripts\run_mt_dnn.sh```, and then pick the best checkpoint based on the average dev preformance of MNLI/RTE. </br>
2. strip the task-specific layers via ```scritps\strip_model.py```. </br>

### Why SciTail/SNLI do not enable SAN?
For SciTail/SNLI tasks, the purpose is to test generalization of the learned embedding and how easy it is adapted to a new domain instead of complicated model structures for a direct comparison with BERT. Thus, we use a linear projection on the all **domain adaptation** settings.

### What is the difference between V1 and V2
The difference is in the QNLI dataset. Please refere to the GLUE official homepage for more details. If you want to formulate QNLI as pair-wise ranking task as our paper, make sure that you use the old QNLI data. </br>
Then run the prepro script with flags:   ```> sh experiments/glue/prepro.sh --old_glue``` </br>
If you have issues to access the old version of the data, please contact the GLUE team.

### Did you fine-tune single task for your GLUE leaderboard submission? 
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


@article{liu2019mt-dnn-kd,
  title={Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding},
  author={Liu, Xiaodong and He, Pengcheng and Chen, Weizhu and Gao, Jianfeng},
  journal={arXiv preprint arXiv:1904.09482},
  year={2019}
}
```
### Contact Information

For help or issues using MT-DNN, please submit a GitHub issue.

For personal communication related to MT-DNN, please contact Xiaodong Liu (`xiaodl@microsoft.com`), Pengcheng He (`penhe@microsoft.com`), Weizhu Chen (`wzchen@microsoft.com`), Jianfeng Gao (`jfgao@microsoft.com`) or Yu Wang (`yuwan@microsoft.com`).
