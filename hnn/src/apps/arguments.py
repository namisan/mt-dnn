import argparse
from utils import *

def build_optimizer_args():
  parser = argparse.ArgumentParser(description='Arguments for optimizer')
  parser.add_argument("--max_grad_norm",
            default=1,
            type=float,
            help="The clip threshold of global gradient norm")
  parser.add_argument("--global_grad_norm",
            default=True,
            type=boolean_string,
            help="wheather to enable global gradient norm clip or use local grad norm with is defualt in v1")
  parser.add_argument("--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.")
  parser.add_argument("--qhadam_v1",
            default=1,
            type=float,
            help="The v1 parameter for QHAdam.")
  parser.add_argument("--qhadam_v2",
            default=1,
            type=float,
            help="The v2 parameter for QHAdam.")
  parser.add_argument("--adam_beta1",
            default=0.9,
            type=float,
            help="The beta1 parameter for Adam.")
  parser.add_argument("--adam_beta2",
            default=0.999,
            type=float,
            help="The beta2 parameter for Adam.")
  parser.add_argument("--epsilon",
            default=1e-6,
            type=float,
            help="The epsilon parameter for Adam.")
  parser.add_argument("--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
              "E.g., 0.1 = 10%% of training.")
  parser.add_argument("--lr_schedule_ends",
            default=0,
            type=float,
            help="The ended learning rate scale for learning rate scheduling")
  parser.add_argument("--lr_schedule",
            default='warmup_linear',
            type=str,
            help="The learning rate scheduler used for traning. "
              "E.g. warmup_linear, warmup_linear_shift, warmup_cosine, warmup_constant. Default, warmup_linear")
  parser.add_argument('--optimize_on_cpu',
            default=False,
            action='store_true',
            help="Whether to perform optimization and keep the optimizer averages on CPU")
  parser.add_argument('--weight_decay',
            type=float,
            default=0.01,
            help="The weight decay rate")
  return parser

def build_training_args():
  parser = argparse.ArgumentParser(description='Arguments for training settings')
  parser.add_argument("--train_batch_size",
            default=32,
            type=int,
            help="Total batch size for training.")
  parser.add_argument('--loss_scale',
            type=float, default=256,
            help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
  parser.add_argument('--scale_steps',
            type=int, default=1000,
            help='The steps to wait to increase the loss scale.')
  parser.add_argument('--no_even_grad',
            default=False,
            action='store_true',
            help="Don't evenly distribute gradient copy amount mutliple devices.")
  parser.add_argument("--num_train_epochs",
            default=3.0,
            type=float,
            help="Total number of training epochs to perform.")
  parser.add_argument("--with_lm_loss",
            default=False,
            type=boolean_string,
            help="Whether to use additional lm loss")
  parser.add_argument('--cls_drop_out',
            type=float,
            default=None,
            help="The config file model initialization and fine tuning.")
  parser.add_argument('--label_smooth',
            type=float,
            default=None,
            help="The parameter for label smoothing.")
  parser.add_argument('--lm_loss_weight',
            type=float,
            default=0.1,
            help="The weight for masked language model loss.")
  parser.add_argument('--cls_loss_weight',
            type=float,
            default=1,
            help="The weight for classification model loss.")
  parser.add_argument('--mask_ratio',
            type=float,
            default=0.1,
            help="The the ratio of masked tokens.")
  parser.add_argument('--mask_keep_ratio',
            type=float,
            default=0.1,
            help="The the ratio of keep the original tokens.")
  parser.add_argument('--mask_other_ratio',
            type=float,
            default=0.1,
            help="The the ratio of replace with other tokens.")
  parser.add_argument('--freeze_lm_weight',
            default=True,
            type=boolean_string,
            help="Whether to stop update embedding in LM loss module")
  parser.add_argument('--reset_ratio',
            type=float,
            default=0,
            help="The the ratio of weight reset.")
  parser.add_argument('--init_spec',
            type=str,
            help="The config file model initialization and fine tuning.")

  parser.add_argument('--ctx_detach',
            default=False,
            type=boolean_string,
            help="Whether to detach context/cls encoding for classification.")
  return parser

def build_runtime_args():
  parser = argparse.ArgumentParser(description='Arguments for runtime settings')
  parser.add_argument("--no_cuda",
            default=False,
            action='store_true',
            help="Whether not to use CUDA when available")

  parser.add_argument("--local_rank",
            type=int,
            default=-1,
            help="local_rank for distributed training on gpus")
  parser.add_argument('--seed',
            type=int,
            default=42,
            help="random seed for initialization")
  parser.add_argument('--gradient_accumulation_steps',
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument('--fp16',
            default=False,
            action='store_true',
            help="Whether to use 16-bit float precision instead of 32-bit")
  return parser

def build_data_args():
  parser = argparse.ArgumentParser(description='Arguments for data cooking and inputs settings.')
  parser.add_argument("--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
              "Sequences longer than this will be truncated, and sequences shorter \n"
              "than this will be padded.")
  parser.add_argument("--do_lower_case",
            default=False,
            action='store_true',
            help="Set this flag if you are using an uncased model.")
  parser.add_argument('--with_spacy',
            action='store_true',
            default=False,
            help="Whether to spacy to re-process the input text")
  parser.add_argument('--with_cache',
            default=True,
            type=boolean_string,
            help="Whether to cache cooked binary features.")
  parser.add_argument('--cache_dir',
            default='torch-cached',
            type=str,
            help="The dir of intermediate cache.")
  parser.add_argument('--worker_num',
            default=0,
            type=int,
            help="The number of workers for data loader.")
  return parser

def build_app_args():
  ## Required parameters
  parser = argparse.ArgumentParser(description='Arguments for application running settings')
  parser.add_argument("--task_name",
            default=None,
            type=str,
            required=True,
            help="The name of the task to train.")

  parser.add_argument("--output_dir",
            default=None,
            type=str,
            required=True,
            help="The output directory where the model checkpoints will be written.")

  ## Other parameters
  parser.add_argument("--do_train",
            default=False,
            action='store_true',
            help="Whether to run training.")
  parser.add_argument("--do_eval",
            default=False,
            action='store_true',
            help="Whether to run eval on the dev set.")
  parser.add_argument("--do_predict",
            default=False,
            action='store_true',
            help="Whether to run prediction on the test set.")
  parser.add_argument("--eval_batch_size",
            default=32,
            type=int,
            help="Total batch size for eval.")
  parser.add_argument("--predict_batch_size",
            default=32,
            type=int,
            help="Total batch size for prediction.")

  parser.add_argument('--tag',
            type=str,
            default='final',
            help="The tag name of current prediction/runs.")
  # BERT arguments
  parser.add_argument('--init_model',
            type=str,
            help="The model state file used to initialize the model weights.")
  parser.add_argument('--bert_config',
            type=str,
            help="The config file of bert model.")
  parser.add_argument('--vocab',
            type=str,
            help="The vocabulary file of bert model.")

  parser.add_argument('--pool_config',
            type=str,
            default=None,
            help="The config file of pooling.")

  parser.add_argument("--attentive_pool",
            default=False,
            type=boolean_string,
            help="Whether to use attention pooling")

  return parser

OptimizerArgs = build_optimizer_args()
TrainingArgs = build_training_args()
RuntimeArgs = build_runtime_args()
DataArgs = build_data_args()
AppArgs = build_app_args()

DefaultArgs = [OptimizerArgs, TrainingArgs, RuntimeArgs, DataArgs, AppArgs]

