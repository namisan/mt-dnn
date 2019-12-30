#
# Author: penhe@microsoft.com
# Date: 04/25/2019
#
""" HNN runner
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from concurrent.futures import ThreadPoolExecutor

import csv
import os
import json
import argparse
import random
import time
from tqdm import tqdm, trange

import numpy as np
import math
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from bert.tokenization import BertTokenizer
from bert.modeling import *
from module.pooling import *
from utils.argument_types import *
from utils.logger_util import *

from dataloader import SequentialDataLoader
from training_utils import *

from hnn_dataset import *
from hnn_model import *
import pdb

class SWA:
  def __init__(self, model, mu, start = 0):
    self.shadow_={}
    self.original_={}
    self.model = model
    for n,p in model.named_parameters():
      self.original_[n] = p.data
      self.shadow_[n] = p.data.clone().detach()
    self.mu = mu
    self.start = start
    self.update_cnt = 0
    self.cnt = 0

  def original(self):
    for n,p in self.model.named_parameters():
      p.data = self.original_[n]
    return self.model

  def shadow(self):
    for n,p in self.model.named_parameters():
      p.data = self.shadow_[n]
    return self.model

  def update(self):
    mu = self.mu
    if self.update_cnt<self.start:
      mu = 0
    else:
      mu = self.cnt/(self.cnt+1)
      self.cnt += 1

    self.update_cnt += 1
    for n in self.shadow_:
      with torch.no_grad():
        self.shadow_[n] = self.original_[n]*(1-mu) + self.shadow_[n]*mu

def create_model(args, num_labels, device):
  # Prepare model
  init_spec = None
  if args.init_spec and args.do_train:
    init_spec = InitSpec.load(args.init_spec)
  model = HNNClassifer.load_model(args.init_model, args.bert_config, init_spec,
      drop_out=args.cls_drop_out, alpha = args.alpha, beta = args.beta, gama=args.gama, similarity=args.similarity,
      loss_type=args.loss_type, pooling=args.pooling.lower())

  def partial_reset(module):
    return model.partial_reset_weights(module, args.reset_ratio)
  if args.do_train:
    model.apply(partial_reset)
  if args.fp16:
    model.half()
  model.to(device)

  dcnt = torch.cuda.device_count()
  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                             output_device=args.local_rank)
  elif dcnt > 1:
    model = torch.nn.DataParallel(model)

  return model

def train_model(args, device, n_gpu, model, train_data, eval_data, training_steps):
  if args.local_rank == -1:
    train_sampler = RandomSampler(train_data)
  else:
    train_sampler = DistributedSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, pin_memory=True, num_workers=args.worker_num)
  init_spec = None
  if args.init_spec and args.do_train:
    init_spec = InitSpec.load(args.init_spec)
  optimizer, param_optimizer, t_total = create_optimizer(model, args, init_spec=init_spec,
      num_train_steps=training_steps) #, no_decay=['bias', 'LayerNorm.weight', 'theta'])
  tr_loss = 0
  nb_tr_steps = 0
  n_epoch = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  start = time.time()
  last_step = 0
  global_step = 0
  last_scale_step = 0
  nan_cnt = 0
  dump_parameter_names(model, os.path.join(args.output_dir, 'parameters.txt'))
  best_metric = 0
  best_epoch = 0
  loss_upper = 500
  acc_steps = 0
  if args.with_lm_loss:
    with open(args.vocab) as fs:
      vocab_dict = {l.strip():i for i,l in enumerate(fs)}

  if eval_data:
    results = run_eval(args, model, device, eval_data, \
      prefix='{}-{}'.format(n_epoch, args.num_train_epochs))
    eval_metric = np.mean([v[0] for k,v in results.items()])
    if eval_metric>best_metric:
      best_metric = eval_metric
      best_epoch = n_epoch
  swa = SWA(model, args.swa, args.swa_start)
  
  model.train()
  zero_grad(model, param_optimizer)
  cls_loss_weight = args.cls_loss_weight
  lm_loss_weight = args.lm_loss_weight
  ctx_only = args.ctx_detach
  mask_ratio = args.mask_ratio
  for ep in range(int(args.num_train_epochs)):
    torch.cuda.empty_cache()
    for step, batch in enumerate(train_dataloader):
      if n_gpu <= 1:
        batch = tuple(t.to(device) for t in batch)
      input_ids, tids, label_ids = batch
      acc_steps += 1
      nb_tr_examples += input_ids.size(0)
      go_next=False
      nan_cnt = 0
      model = swa.original()
      while not go_next:
        _,_,_,loss = model(input_ids, tids, label_ids, group_tasks=torch.tensor([args.group_tasks]*len(input_ids)))
        if n_gpu > 1:
          loss = loss.mean() # mean() to average on multi-gpu.
        step_loss = loss.item()
        if args.fp16:
          # rescale loss for fp16 training
          # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
          loss_scale = loss_upper/loss.float().detach().item()

          if step_loss != 0:
            loss_scale = loss_upper/loss.float().detach().item()
          else:
            loss_scale = 1
          loss = loss.float() * loss_scale
        if args.gradient_accumulation_steps > 1:
          loss = loss / args.gradient_accumulation_steps
        loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
          if args.fp16:
            # scale down gradients for fp16 training
            is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
            for n,param in param_optimizer:
              if param.grad is not None:
                param.grad.data = param.grad.data/loss_scale
            if is_nan:
              logger.warn("Hit nan gradient with loss scale: {}:{}".format(loss_scale, loss_upper))
              nan_cnt += 1
              zero_grad(model, param_optimizer)
              args.loss_scale = args.loss_scale*nan_cnt/(nan_cnt+1)
              loss_upper *=  nan_cnt/(nan_cnt+1)
              last_scale_step = global_step
              continue
            elif (global_step - last_scale_step > args.scale_steps):
              args.loss_scale = args.loss_scale*1.5
              loss_upper*=2
              last_scale_step = global_step
            optimizer.step()
            copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
          elif args.optimize_on_cpu:
            set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
            optimizer.step()
            copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
          else:
            optimizer.step()
          zero_grad(model, param_optimizer)
          global_step += 1
          if global_step%100 == 0:
            #swa.update()
            end = time.time()
            logger.info('[{:0.1f}%][{:0.2f}h] Steps={}, loss={}, examples={}, {:0.1f}s'.format(100*global_step/t_total, (t_total-global_step)*(start-end)/((global_step-last_step)*3600), global_step, tr_loss/nb_tr_steps, nb_tr_examples, end-start))
            start = time.time()
            last_step = global_step
          if global_step%10000 == 0:
            swa.update()
            best_metric, best_epoch = make_check_point(args, device, swa.shadow(), param_optimizer, eval_data, n_epoch, global_step, best_metric, best_epoch)
        go_next = True
      nb_tr_steps += 1
      tr_loss += step_loss
    n_epoch += 1

    if global_step> last_step:
      end = time.time()
      logger.info('[{:0.1f}%][{:0.2f}h] Steps={}, loss={}, examples={}, {:0.1f}s'.format(100*global_step/t_total, (t_total-global_step)*(start-end)/((global_step-last_step)*3600), global_step, tr_loss/nb_tr_steps, nb_tr_examples, end-start))
      start = time.time()
      last_step = global_step

    swa.update()
    best_metric, best_epoch = make_check_point(args, device, swa.shadow(), param_optimizer, eval_data, n_epoch, global_step, best_metric, best_epoch)

def make_check_point(args, device, model, param_optimizer, eval_data, n_epoch, steps, best_metric, best_epoch):
  # Save model
  zero_grad(model, param_optimizer)
  save_path= os.path.join(args.output_dir, f'pytorch.model-{n_epoch:02}-{steps:06}.bin')
  state_dict = OrderedDict([(n,p) for n,p in model.module.state_dict().items()])
  if args.fp16:
    state_dict.update([(n[len('module.'):] if n.startswith('module.') else n ,p) for n,p in param_optimizer])
  logger.info('Save model to: {}'.format(save_path))
  torch.save(state_dict, save_path)

  if eval_data:
    results = run_eval(args, model.eval(), device, eval_data, \
        prefix=f'{n_epoch:02}-{steps:06}-{args.num_train_epochs}')
    eval_metric = np.mean([v[0] for k,v in results.items()])
    if eval_metric>best_metric:
      best_metric = eval_metric
      best_epoch = n_epoch
    logger.info('Best metric={}@{}'.format(best_metric, best_epoch))
  model.train()
  return best_metric, best_epoch

def run_eval(args, model, device, eval_data, prefix=None):
  # Run prediction for full data
  eval_results=OrderedDict()
  eval_metric=0
  for eval_item in eval_data:
    torch.cuda.empty_cache()
    name = eval_item.name
    eval_sampler = SequentialSampler(eval_item.data)
    eval_dataloader = SequentialDataLoader(eval_item.data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.worker_num)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    sm_predicts=[]
    lm_predicts=[]
    en_predicts=[]
    labels=[]
    for input_ids, tids, label_ids in tqdm(eval_dataloader, ncols=80, desc='Evaluating: {}'.format(prefix)):
      with torch.no_grad():
        sm_logits, lm_logits, en_logits, tmp_eval_loss = model(input_ids, tids, label_ids)
      label_ids = label_ids.to('cpu').numpy()
      labels.append(label_ids)
      if sm_logits is not None:
        sm_predicts.append(sm_logits.detach().cpu().numpy())
      if lm_logits is not None:
        lm_predicts.append(lm_logits.detach().cpu().numpy())
      if en_logits is not None:
        en_predicts.append(en_logits.detach().cpu().numpy())
  
      eval_loss += tmp_eval_loss.mean().item()
  
      nb_eval_examples += input_ids.size(0)
      nb_eval_steps += 1
  
    eval_loss = eval_loss / nb_eval_steps
    result=OrderedDict()
    labels = np.concatenate(labels, axis=0)
    metrics_fn = eval_item.metrics_fn
    def calc_metric(predicts, mn="", quiet=False):
      if metrics_fn is None:
        eval_metric = metric_accuracy(predicts, labels)
      else:
        metrics = metrics_fn(predicts, labels, quiet=quiet, tag=mn)
      for k in metrics:
        result[mn+k] = metrics[k]
      output_predict_file = os.path.join(args.output_dir, "{}predict_results_{}_{}.txt".format(mn, name, prefix))
      np.savetxt(output_predict_file, predicts, delimiter='\t')
      eval_metric = np.mean(list(metrics.values()))
      eval_results[mn + name]=(eval_metric, predicts, labels)
      return eval_metric
    ensemble_list = []
    if len(lm_predicts)>0:
      lm_predicts = np.concatenate(lm_predicts, axis=0)
      ensemble_list.append(lm_predicts)
      eval_metric=calc_metric(lm_predicts, 'LM-')

    if len(sm_predicts)>0:
      sm_predicts = np.concatenate(sm_predicts, axis=0)
      ensemble_list.append(sm_predicts)
      eval_metric=calc_metric(sm_predicts, 'SIM-')

    if len(en_predicts)>0:
      en_predicts = np.concatenate(en_predicts, axis=0)
      eval_metric=calc_metric(en_predicts, f'EN-Avg-')
      ensemble_list.append(en_predicts)
      if args.ensemble:
        best = 0
        best_i = 0
        best_logits = None
        for i in np.arange(1,-0.1,-0.1):
          en_predicts = i*sm_predicts + (1-i)*lm_predicts
          eval_metric=calc_metric(en_predicts, f'EN-Scan-', quiet=True)
          if eval_metric>best:
            best = eval_metric
            best_logits = en_predicts
            best_i = i
        logger.info(f'Scanned best metric@{best_i:0.02}: {best:0.03}')
        eval_metric=calc_metric(best_logits, f'EN-Scan-', quiet=True)

    result['eval_loss'] = eval_loss
    result['eval_metric'] = eval_metric
    result['eval_samples'] = len(labels)
    output_eval_file = os.path.join(args.output_dir, "eval_results_{}_{}.txt".format(name, prefix))
    with open(output_eval_file, "w") as writer:
      logger.info("***** Eval results-{}-{} *****".format(name, prefix))
      for key in sorted(result.keys()):
        logger.info("  %s = %0.03f", key, result[key])
        writer.write("%s = %s\n" % (key, str(result[key])))
    output_label_file = os.path.join(args.output_dir, "predict_labels_{}_{}.txt".format(name, prefix))
    np.savetxt(output_label_file, labels, delimiter='\t')
  return eval_results

def run_predict(args, model, device, test_data, prefix=None):
  # Run prediction for full data
  eval_results=OrderedDict()
  for test_item in test_data:
    torch.cuda.empty_cache()
    name = test_item.name
    test_sampler = SequentialSampler(test_item.data)
    test_dataloader = SequentialDataLoader(test_item.data, sampler=test_sampler, batch_size=args.predict_batch_size, num_workers = args.worker_num)
    model.eval()
    sm_predicts=[]
    lm_predicts=[]
    en_predicts=[]
    for input_ids,tids,_ in tqdm(test_dataloader, ncols=80, desc='Predicting: {}'.format(prefix)):
      with torch.no_grad():
        sm_logits, lm_logits, en_logits,_ = model(input_ids, tids)
      if sm_logits is not None:
        sm_predicts.append(sm_logits.detach().cpu().numpy())
      if lm_logits is not None:
        lm_predicts.append(lm_logits.detach().cpu().numpy())
      if en_logits is not None:
        en_predicts.append(en_logits.detach().cpu().numpy())
    def pred(predicts, tag):
      output_test_file = os.path.join(args.output_dir, "test_results_{}_{}_{}.txt".format(name, prefix, tag))
      logger.info("***** Dump prediction results-{}-{}-{} *****".format(name, prefix, tag))
      logger.info("Location: {}".format(output_test_file))
      np.savetxt(output_test_file, predicts, delimiter='\t')
      predict_fn = test_item.predict_fn
      if predict_fn:
        predict_fn(predicts, args.output_dir, name, prefix, tag=tag)
    if len(lm_predicts)>0:
      lm_predicts = np.concatenate(lm_predicts, axis=0)
      pred(lm_predicts, 'LM-')

    if len(sm_predicts)>0:
      sm_predicts = np.concatenate(sm_predicts, axis=0)
      pred(sm_predicts, 'SIM-')

    if len(en_predicts)>0:
      en_predicts = np.concatenate(en_predicts, axis=0)
      pred(en_predicts, 'EN-Avg-')

def build_training_data(args, tokenizer, tasks):
  dprd_task = DPRDTask(tokenizer)
  if args.wiki_data:
    wiki_task = WikiWSCRTask(tokenizer)
    train_data = wiki_task.get_train_dataset(args.wiki_data, args.max_seq_length, input_type=tasks)
  else:
    train_data = dprd_task.get_train_dataset(args.data_dir, args.max_seq_length, input_type=tasks)
    if args.dev_train:
      _data = dprd_task.get_dev_dataset(args.data_dir, args.max_seq_length, input_type=tasks)
      _data = [e.data for e in _data if e.name=='DPRD-test'][0]
      train_data = ConcatDataset([train_data, _data])
    if args.gap_data:
      gap_data = gap_task.get_train_dataset(args.gap_data, args.max_seq_length, input_type=tasks)
      train_data = ConcatDataset([train_data, gap_data])
      if args.dev_train:
        gap_data = [e.data for e in gap_task.get_dev_dataset(args.gap_data, args.max_seq_length, input_type=tasks)]
        train_data = ConcatDataset(gap_data + [train_data])
  return train_data

def build_training_data_mt(args, tokenizer):
  if args.group_tasks:
    return build_training_data(args, tokenizer, args.tasks)
  else:
    data = []
    for t in args.tasks:
      data.append(build_training_data(args, tokenizer, [t]))
    return ConcatDataset(data)

def main(args):
  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
  else:
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
    if args.fp16:
      logger.info("16-bits training currently not supported in distributed training")
      args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
  logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

  if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
              args.gradient_accumulation_steps))

  args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

  if not args.do_train and not args.do_eval and not args.do_predict:
    raise ValueError("At least one of `do_train` or `do_eval` or `do_predict` must be True.")

  if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    logger.warn('Output directory {} already exists.'.format(args.output_dir))
  os.makedirs(args.output_dir, exist_ok=True)

  task_name = args.task_name.lower()

  tokenizer = BertTokenizer(args.vocab, do_lower_case=args.do_lower_case)
  dprd_task = DPRDTask(tokenizer)

  eval_data = dprd_task.get_dev_dataset(args.data_dir, 128, input_type=args.tasks)
  if args.wnli_data:
    wnli_task = WNLITask(tokenizer)
    wnli_data = wnli_task.get_dev_dataset(args.wnli_data, 128, input_type=args.tasks)
    eval_data += wnli_data
  if args.wsc273_data:
    wsc273_task = WSC273Task(tokenizer)
    wsc273_data = wsc273_task.get_dev_dataset(args.wsc273_data, 128, input_type=args.tasks)
    eval_data += wsc273_data

  if args.gap_data:
    gap_task = GAPTask(tokenizer)
    gap_data = gap_task.get_dev_dataset(args.gap_data, 384, input_type=args.tasks)
    eval_data += gap_data

  logger.info("  Evaluation batch size = %d", args.eval_batch_size)
  train_examples = None
  num_train_steps = None
  if args.do_train:
    train_data = build_training_data_mt(args, tokenizer)

    total_examples = len(train_data)
    num_train_steps = int(total_examples / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info("  Training batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

  model = create_model(args, 2, device)

  if args.do_train:
    train_model(args, device, n_gpu, model, train_data, eval_data, num_train_steps)

  if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    run_eval(args, model.eval(), device, eval_data, prefix=args.tag)

  if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    test_data = dprd_task.get_test_dataset(args.data_dir, 128, input_type=args.tasks)
    if args.wnli_data:
      wnli_data = wnli_task.get_test_dataset(args.wnli_data, 128, input_type=args.tasks)
      test_data += wnli_data
    if args.wsc273_data:
      wsc273_data = wsc273_task.get_test_dataset(args.wsc273_data, 128, input_type=args.tasks)
      test_data += wsc273_data
    logger.info("  Prediction batch size = %d", args.predict_batch_size)
    run_predict(args, model, device, test_data, prefix=args.tag)

def build_argument_parser():
  from arguments import DefaultArgs
  parser = argparse.ArgumentParser(parents=DefaultArgs, conflict_handler='resolve')
  parser.add_argument('--gama',
      default='1,1',
      type=str,
      help='Hyperparamter of ranking hinge loss')
  parser.add_argument('--alpha',
      default='10,10',
      type=str,
      help='Hyperparamter of ranking hinge loss')

  parser.add_argument('--beta',
      default='0.5,0.5',
      type=str,
      help='Hyperparamter of ranking hinge loss')

  parser.add_argument('--wnli_data',
      default=None,
      type=str,
      help='data dir for wnli data')

  parser.add_argument('--wiki_data',
      default=None,
      type=str,
      help='data dir for wiki data')

  parser.add_argument('--wsrc_data',
      default=None,
      type=str,
      help='data dir for wsrc data')

  parser.add_argument('--gap_data',
      default=None,
      type=str,
      help='data dir for gap data')

  parser.add_argument('--dev_train',
      default=False,
      type=boolean_string,
      help='Whether to merge dev data in to training')

  parser.add_argument('--wsc273_data',
      default=None,
      type=str,
      help='data dir for wsc273 data')

  parser.add_argument('--tasks',
      default=None,
      type=str,
      help='tasks to run, SM,LM')

  parser.add_argument('--group_tasks',
      default=True,
      type=boolean_string,
      help='Where to group different task to one instance.')

  parser.add_argument('--similarity',
      default='cos',
      type=str,
      help='similarity module to be used, cos|neural')
  parser.add_argument('--loss_type',
      default='binary',
      type=str,
      help='similarity module to be used, binary|entropy')

  parser.add_argument('--ensemble',
      default=False,
      type=boolean_string,
      help='Whether to scan the ensemble coefficient')

  parser.add_argument('--pooling',
      default='cap',
      type=str,
      help='Pooling method to produce antencedent encoding, cap, mean, ftp')

  parser.add_argument('--swa',
      default=0,
      type=float,
      help='The parameter mu of model swa')

  parser.add_argument('--swa_start',
      default=0,
      type=float,
      help='The start step of model swa')
  return parser

if __name__ == "__main__":
  parser = build_argument_parser()
  args = parser.parse_args()
  args.alpha = [float(k) for k in args.alpha.split(',')]
  args.beta = [float(k) for k in args.beta.split(',')]
  args.gama = [float(k) for k in args.gama.split(',')]
  args.tasks = args.tasks.split(',') if args.tasks else None
  logger = set_logger(args.task_name, os.path.join(args.output_dir, 'training_{}.log'.format(args.task_name)))
  logger.info(args)
  main(args)
