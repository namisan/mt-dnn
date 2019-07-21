# This scripts is to convert Google's TF BERT to the pytorch version which is used by mt-dnn.
# It is a supplementary script.
# Note that it relies on tensorflow==1.12.0 which does not support by our released docker. 
# If you want to use this, please install tensorflow==1.12.0 by: pip install tensorflow==1.12.0
# Some codes are adapted from https://github.com/huggingface/pytorch-pretrained-BERT
# by: xiaodl
from __future__ import absolute_import
from __future__ import division
import re
import os
import argparse
import tensorflow as tf
import torch
import numpy as np
from pytorch_pretrained_bert.modeling import BertConfig
from sys import path
path.append(os.getcwd())
from mt_dnn.matcher import SANBertNetwork
from data_utils.log_wrapper import create_logger

logger =  create_logger(__name__, to_disk=False)
def model_config(parser):
    parser.add_argument('--update_bert_opt',  default=0, type=int)
    parser.add_argument('--multi_gpu_on', action='store_true')
    parser.add_argument('--mem_cum_type', type=str, default='simple',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_num_turn', type=int, default=5)
    parser.add_argument('--answer_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--answer_att_hidden_size', type=int, default=128)
    parser.add_argument('--answer_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--answer_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_merge_opt', type=int, default=1)
    parser.add_argument('--answer_mem_type', type=int, default=1)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_weight_norm_on', action='store_true')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=0, help='0,1')
    parser.add_argument('--label_size', type=str, default='3')
    parser.add_argument('--mtl_opt', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--init_ratio', type=float, default=1)
    return parser

def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--tasks_dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.000)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)
    parser.add_argument('--dump_feature', action='store_false')

    # EMA
    parser.add_argument('--ema_opt', type=int, default=0)
    parser.add_argument('--ema_gamma', type=float, default=0.995)

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--freeze_layers', type=int, default=-1)
    parser.add_argument('--embedding_opt', type=int, default=0)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--bert_l2norm', type=float, default=0.0)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_dir', default='checkpoint')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    return parser


def convert(args):
    tf_checkpoint_path = args.tf_checkpoint_root
    bert_config_file = os.path.join(tf_checkpoint_path, 'bert_config.json')
    pytorch_dump_path = args.pytorch_checkpoint_path
    config = BertConfig.from_json_file(bert_config_file)
    opt = vars(args)
    opt.update(config.to_dict())
    model = SANBertNetwork(opt)
    path = os.path.join(tf_checkpoint_path, 'bert_model.ckpt')
    logger.info('Converting TensorFlow checkpoint from {}'.format(path))
    init_vars = tf.train.list_variables(path)
    names = []
    arrays = []

    for name, shape in init_vars:
        logger.info('Loading {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(path, name)
        logger.info('Numpy array shape {}'.format(array.shape))

        # new layer norm var name
        # make sure you use the latest huggingface's new layernorm implementation
        # if you still use beta/gamma, remove line: 48-52
        if name.endswith('LayerNorm/beta'):
            name = name[:-14] + 'LayerNorm/bias'
        if name.endswith('LayerNorm/gamma'):
            name = name[:-15] + 'LayerNorm/weight'

        if name.endswith('bad_steps'):
            print('bad_steps')
            continue
        if name.endswith('steps'):
            print('step')
            continue
        if name.endswith('step'):
            print('step')
            continue
        if name.endswith('adam_m'):
            print('adam_m')
            continue
        if name.endswith('adam_v'):
            print('adam_v')
            continue
        if name.endswith('loss_scale'):
            print('loss_scale')
            continue
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        flag = False
        if name == 'cls/squad/output_bias':
            name = 'out_proj/bias'
            flag = True
        if name == 'cls/squad/output_weights':
            name = 'out_proj/weight'
            flag = True

        logger.info('Loading {}'.format(name))
        name = name.split('/')
        if name[0] in ['redictions', 'eq_relationship', 'cls', 'output']:
            logger.info('Skipping')
            continue
        pointer = model
        for m_name in name:
            if flag: continue
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        elif flag:
            continue
            pointer = getattr(getattr(pointer, name[0]), name[1])
        try:
            assert tuple(pointer.shape) == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pointer.data = torch.from_numpy(array)

    nstate_dict = model.state_dict()
    params = {'state':nstate_dict, 'config': config.to_dict()}
    torch.save(params, pytorch_dump_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint_root', type=str, required=True)
    parser.add_argument('--pytorch_checkpoint_path', type=str, required=True)
    parser = model_config(parser)
    parser = train_config(parser)
    args = parser.parse_args()
    logger.info(args)
    convert(args)
