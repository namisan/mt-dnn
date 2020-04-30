[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Travis-CI](https://travis-ci.org/namisan/mt-dnn.svg?branch=master)](https://github.com/namisan/mt-dnn)


# Adversarial Training for Large Neural Language Models (ALUM)

This PyTorch package implements the Adversarial Training for Large Neural Language Models, as described in:

Xiaodong Liu, Hao Cheng, Pengcheng He, Weizhu Chen, Yu Wang, Hoifung Poon and Jianfeng Gao<br/>
Adversarial Training for Large Neural Language Models <br/>
[arXiv version](https://arxiv.org/abs/2004.08994) <br/>


## Quickstart

### Setup Environment
   Refer to [setting] (https://github.com/pytorch/fairseq)


### Train a toy MT-DNN model
1. Download data </br>
   Download the large scale raw texts: [RoBERTa](https://arxiv.org/abs/1907.11692) 

2. Preprocess data/Training </br>
   Please refer: [prepro/train](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md)


## Notes and Acknowledgments
BERT pytorch is from: https://github.com/huggingface/pytorch-pretrained-BERT <br/>
FAIRSEQ: https://github.com/pytorch/fairseq<br/>
We also used some code from: https://github.com/kevinduh/san_mrc <br/>

### How do I cite ALUM?

```

@article{liu2020alum,
  title={Adversarial Training for Large Neural Language Models},
  author={Liu, Xiaodong and Cheng, Hao and He, Pengcheng and Chen, Weizhu and Wang, Yu and Poon, Hoifung and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2004.08994},
  year={2020}
}
```
### Contact Information


For personal communication related to this package, please contact Xiaodong Liu (`xiaodl@microsoft.com`), Hao Cheng (`chehao@microsoft.com`), Pengcheng He (`penhe@microsoft.com`), Weizhu Chen (`wzchen@microsoft.com`), Yu Wang (`yuwan@microsoft.com`), Hoifung Poon (`hoifung@microsoft.com`), or Jianfeng Gao (`jfgao@microsoft.com`).
