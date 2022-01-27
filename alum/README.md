[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Travis-CI](https://travis-ci.org/namisan/mt-dnn.svg?branch=master)](https://github.com/namisan/mt-dnn)


# Adversarial Training for Large Neural Language Models (ALUM)

This PyTorch package implements the Adversarial Training for Large Neural Language Models, as described in:

Xiaodong Liu, Hao Cheng, Pengcheng He, Weizhu Chen, Yu Wang, Hoifung Poon and Jianfeng Gao<br/>
Adversarial Training for Large Neural Language Models <br/>
[arXiv version](https://arxiv.org/abs/2004.08994) <br/>


## Quickstart

### Setup Environment
   [Setup](https://github.com/pytorch/fairseq).


### Pre-train an ALUM model
1. Download data </br>
   [Data Info](https://arxiv.org/abs/1907.11692) 

2. Prepro/train </br>
   [Prepro/train](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md) </br>
   > bash alum_train.sh <data_dir> <LR> <alum_code_path> <RoBERTa_model_path>


## Notes and Acknowledgments
FAIRSEQ: https://github.com/pytorch/fairseq<br/>
Megatron-LM: https://github.com/NVIDIA/Megatron-LM <br/>
SAN: https://github.com/kevinduh/san_mrc <br/>

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
