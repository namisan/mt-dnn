[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Travis-CI](https://travis-ci.org/namisan/mt-dnn.svg?branch=master)](https://github.com/namisan/mt-dnn)

**New Release** <br/>
We released Adversarial training for both LM pre-training/finetuning and f-divergence.

Large-scale Adversarial training for LMs: [ALUM code](https://github.com/namisan/mt-dnn/blob/master/alum/README.md). <br/>
If you want to use the old version, please use following cmd to clone the code: <br/>
```git clone -b v0.1 https://github.com/namisan/mt-dnn.git ```



# Multi-Task Deep Neural Networks for Natural Language Understanding

This PyTorch package implements the Multi-Task Deep Neural Networks (MT-DNN) for Natural Language Understanding, as described in:

Xiaodong Liu\*, Pengcheng He\*, Weizhu Chen and Jianfeng Gao<br/>
Multi-Task Deep Neural Networks for Natural Language Understanding<br/>
[ACL 2019](https://aclweb.org/anthology/papers/P/P19/P19-1441/) <br/>
\*: Equal contribution <br/>

Xiaodong Liu, Pengcheng He, Weizhu Chen and Jianfeng Gao<br/>
Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding <br/>
[arXiv version](https://arxiv.org/abs/1904.09482) <br/>


Pengcheng He, Xiaodong Liu, Weizhu Chen and Jianfeng Gao<br/>
Hybrid Neural Network Model for Commonsense Reasoning <br/>
[arXiv version](https://arxiv.org/abs/1907.11983) <br/>


Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao and Jiawei Han <br/>
On the Variance of the Adaptive Learning Rate and Beyond <br/>
[arXiv version](https://arxiv.org/abs/1908.03265) <br/>

Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao and Tuo Zhao <br/>
SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization <br/>
[arXiv version](https://arxiv.org/abs/1911.03437) <br/>

Xiaodong Liu, Yu Wang, Jianshu Ji, Hao Cheng, Xueyun Zhu, Emmanuel Awa, Pengcheng He, Weizhu Chen, Hoifung Poon, Guihong Cao, Jianfeng Gao<br/>
The Microsoft Toolkit of Multi-Task Deep Neural Networks for Natural Language Understanding <br/>
[arXiv version](https://arxiv.org/abs/2002.07972) <br/>

Xiaodong Liu, Hao Cheng, Pengcheng He, Weizhu Chen, Yu Wang, Hoifung Poon and Jianfeng Gao<br/>
Adversarial Training for Large Neural Language Models <br/>
[arXiv version](https://arxiv.org/abs/2004.08994) <br/>

Hao Cheng and Xiaodong Liu and Lis Pereira and Yaoliang Yu and Jianfeng Gao<br/>
Posterior Differential Regularization with f-divergence for Improving Model Robustness <br/>
[arXiv version](https://arxiv.org/abs/2010.12638) <br/>


## Quickstart

### Setup Environment
#### Install via pip:
1. python3.6 </br>
   Reference to download and install : https://www.python.org/downloads/release/python-360/

2. install requirements </br>
   ```> pip install -r requirements.txt```

#### Use docker:
1. Pull docker </br>
   ```> docker pull allenlao/pytorch-mt-dnn:v0.5```

2. Run docker </br>
   ```> docker run -it --rm --runtime nvidia  allenlao/pytorch-mt-dnn:v0.5 bash``` </br>
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

### Sequence Labeling Task
1. Preprocess data </br>
   a) Download NER data to data/ner including: {train/valid/test}.txt </br>
   b) Convert NER data to the canonical format: ```> python experiments\ner\prepro.py --data data\ner --output_dir data\canonical_data``` </br>
   c) Preprocess the canonical data to the MT-DNN format: ```> python prepro_std.py --do_lower_case --root_dir data\canonical_data --task_def experiments\ner\ner_task_def.yml --model bert-base-uncased``` </br>

2. Training </br>
   ```> python train.py --data_dir <data-path> --init_checkpoint <bert/ner-model> --train_dataset ner --test_dataset ner --task_def experiments\ner\ner_task_def.yml```

### SMART
Adv training at the fine-tuning stages:
   ```> python train.py --data_dir <data-path> --init_checkpoint <bert/mt-dnn-model> --train_dataset mnli --test_dataset mnli_matched,mnli_mismatched --task_def experiments\glue\glue_task_def.yml --adv_train --adv_opt 1```


### HNN
The code to reproduce HNN is under `hnn` folder, to reproduce the results of HNN, run 

```> hnn/script/hnn_train_large.sh```


### Extract embeddings
1. Extracting embeddings of a pair text example </br>
   ```>python extractor.py --do_lower_case --finput input_examples\pair-input.txt --foutput input_examples\pair-output.json --bert_model bert-base-uncased --checkpoint mt_dnn_models\mt_dnn_base.pt``` </br>
   Note that the pair of text is split by a special token ```|||```. You may refer ``` input_examples\pair-output.json``` as example. </br>

2. Extracting embeddings of a single sentence example </br>
   ```>python extractor.py  --do_lower_case --finput input_examples\single-input.txt --foutput input_examples\single-output.json --bert_model bert-base-uncased --checkpoint mt_dnn_models\mt_dnn_base.pt``` </br>


### Speed up Training
1. Gradient Accumulation </br>
   If you have small GPUs, you may need to use the gradient accumulation to make training stable. </br>
   For example, if you use the flag: ```--grad_accumulation_step 4 ``` during the training, the actual batch size will be ``` batch_size * 4 ```. </br>

2. FP16
   The current version of MT-DNN also supports FP16 training, and please install apex. </br>
   You just need to turn on the flag during the training: ```--fp16 ```  </br>
Please refer the script: ``` scripts\run_mt_dnn_gc_fp16.sh```



### Convert Tensorflow BERT model to the MT-DNN format
Here, we go through how to convert a Chinese Tensorflow BERT model into mt-dnn format. <br/>
1. Download BERT model from the Google bert web: https://github.com/google-research/bert <br/>

2. Run the following script for MT-DNN format</br>
   ```python scripts\convert_tf_to_pt.py --tf_checkpoint_root chinese_L-12_H-768_A-12\ --pytorch_checkpoint_path chinese_L-12_H-768_A-12\bert_base_chinese.pt```

### TODO
- [ ] Publish pretrained Tensorflow checkpoints. <br/>


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

## Related Projects/Codebase
1. Pretrained UniLM: https://github.com/microsoft/unilm <br/>
2. Pretrained Response Generation Model: https://github.com/microsoft/DialoGPT <br/>
3. Internal MT-DNN repo: https://github.com/microsoft/mt-dnn <br/>

### How do I cite MT-DNN?

```
@inproceedings{liu2019mt-dnn,
    title = "Multi-Task Deep Neural Networks for Natural Language Understanding",
    author = "Liu, Xiaodong and He, Pengcheng and Chen, Weizhu and Gao, Jianfeng",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1441",
    pages = "4487--4496"
}


@article{liu2019mt-dnn-kd,
  title={Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding},
  author={Liu, Xiaodong and He, Pengcheng and Chen, Weizhu and Gao, Jianfeng},
  journal={arXiv preprint arXiv:1904.09482},
  year={2019}
}


@article{he2019hnn,
  title={A Hybrid Neural Network Model for Commonsense Reasoning},
  author={He, Pengcheng and Liu, Xiaodong and Chen, Weizhu and Gao, Jianfeng},
  journal={arXiv preprint arXiv:1907.11983},
  year={2019}
}


@article{liu2019radam,
  title={On the Variance of the Adaptive Learning Rate and Beyond},
  author={Liu, Liyuan and Jiang, Haoming and He, Pengcheng and Chen, Weizhu and Liu, Xiaodong and Gao, Jianfeng and Han, Jiawei},
  journal={arXiv preprint arXiv:1908.03265},
  year={2019}
}


@article{jiang2019smart,
  title={SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization},
  author={Jiang, Haoming and He, Pengcheng and Chen, Weizhu and Liu, Xiaodong and Gao, Jianfeng and Zhao, Tuo},
  journal={arXiv preprint arXiv:1911.03437},
  year={2019}
}


@article{liu2020mtmtdnn,
  title={The Microsoft Toolkit of Multi-Task Deep Neural Networks for Natural Language Understanding},
  author={Liu, Xiaodong and Wang, Yu and Ji, Jianshu and Cheng, Hao and Zhu, Xueyun and Awa, Emmanuel and He, Pengcheng and Chen, Weizhu and Poon, Hoifung and Cao, Guihong and Jianfeng Gao},
  journal={arXiv preprint arXiv:2002.07972},
  year={2020}
}


@article{liu2020alum,
  title={Adversarial Training for Large Neural Language Models},
  author={Liu, Xiaodong and Cheng, Hao and He, Pengcheng and Chen, Weizhu and Wang, Yu and Poon, Hoifung and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2004.08994},
  year={2020}
}

@article{cheng2020posterior,
  title={Posterior Differential Regularization with f-divergence for Improving Model Robustness},
  author={Cheng, Hao and Liu, Xiaodong and Pereira, Lis and Yu, Yaoliang and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2010.12638},
  year={2020}
}
```
### Contact Information

For help or issues using MT-DNN, please submit a GitHub issue.

For personal communication related to this package, please contact Xiaodong Liu (`xiaodl@microsoft.com`), Yu Wang (`yuwan@microsoft.com`), Pengcheng He (`penhe@microsoft.com`), Weizhu Chen (`wzchen@microsoft.com`), Jianshu Ji (`jianshuj@microsoft.com`), Hao Cheng (`chehao@microsoft.com`) or Jianfeng Gao (`jfgao@microsoft.com`).
