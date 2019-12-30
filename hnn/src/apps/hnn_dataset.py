from glob import glob
from collections import namedtuple
import pickle
from multiprocessing import Pool, Queue, Process
from tqdm import tqdm
import os
import sys
import random
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from io import BytesIO
from tqdm import tqdm
from array import array
import struct
import gc
import utils
import pdb
from collections import OrderedDict
import numpy as np

logger = utils.get_logger()
from bert.tokenization import BertTokenizer
from bert.tokenization import convert_to_unicode

import spacy
class EvalData:
  def __init__(self, name, examples, metrics_fn=None, predict_fn=None):
    def accuracy_fn(logits, labels):
      return OrderedDict()

    def default_pred_fn(logits, output_dir, name, prefix):
      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      preds = np.argmax(logits, axis=-1)
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('index\tpredictions\n')
        for i,p in enumerate(preds):
          fs.write('{}\t{}\n'.format(i, p))
    self.name = name
    self.data = examples
    self.metrics_fn = metrics_fn if metrics_fn is not None else accuracy_fn
    self.predict_fn = predict_fn if predict_fn is not None else default_pred_fn

def load_tsv(path, columns):
  with open(path) as fs:
    lines = [l.strip().split('\t') for l in fs]
  return [[l[c] for c in columns]  for l in lines[1:]]

def token_index(src, tgt, offset = 0):
  i = offset
  while i < len(src):
    k=0
    while k<len(tgt) and (i+k)<len(src) and src[i+k]==tgt[k]:
      k+=1
    if k==len(tgt):
      return i
    else:
      i+=1
  return -1

class _ABCDataset(Dataset):
  def __init__(self, max_len=256, tid=None):
    self.max_len = max_len
    self.doc_count = 0
    self.tid = tid if tid is not None and len(tid)>0 else ['lm', 'sm']
    self.tid = [t.lower() for t in self.tid]
    assert 'lm' in self.tid or 'sm' in self.tid, 'tids must be lm|sm'

  def __len__(self):
    return self.doc_count

  def __getitem__(self, index):
    input_ids, tids, selected_idx = self.get_example(index)
    for k in input_ids:
      for z in k:
        z.extend([0]*(self.max_len-len(z)))

    inputs = input_ids
    label = torch.tensor(selected_idx, dtype=torch.float)
    tids = torch.tensor(tids, dtype=torch.int)

    return [torch.tensor(inputs, dtype=torch.int), tids, label]

  def get_data(self, index):
    raise NotImplemented('The method must be implmented by sub class')

  def get_example(self, index):
    raise NotImplemented('The method must be implmented by sub class')

  def _make_inputs(self, src_left, src_right, pronoun_tokens, candidates, selected):
    lm_i,lm_s,lm_t = self._make_inputs_lm(src_left, src_right, pronoun_tokens, candidates, selected, 1)
    sm_i,sm_s,sm_t = self._make_inputs_sm(src_left, src_right, pronoun_tokens, candidates, selected, 0)
    input_ids = []
    task_ids = []
    if 'lm' in self.tid:
      input_ids = lm_i
      task_ids = lm_t

    if 'sm' in self.tid:
      input_ids += sm_i
      task_ids += sm_t
    return (input_ids, task_ids, selected)

  # inputs for language modeling
  def _make_inputs_lm(self, src_left, src_right, pronoun_tokens, candidates, selected, tid=1):
    pronoun_idx = len(src_left) + 1
    input_ids = []
    for cand in candidates:
      if cand:
        tokens = ['[CLS]'] + src_left + ['[MASK]' for _ in range(len(cand))] + src_right + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        type_ids = [0]*len(tokens)
        mask_ids = [1]*len(token_ids)
        cand_ids = [0]*len(token_ids)
        cand_ids[pronoun_idx:pronoun_idx+len(cand)]=self.tokenizer.convert_tokens_to_ids(cand)
        input_ids.append([token_ids, mask_ids, type_ids, cand_ids, [0]])
      else:
        input_ids.append([[0], [0], [0], [0], [0]])

    task_ids = [tid for _ in range(len(input_ids))]
    
    return (input_ids, selected, task_ids)

  # inputs for semantic matching
  def _make_inputs_sm(self, src_left, src_right, pronoun_tokens, candidates, selected, tid=0):
    src_tokens = ['[CLS]'] + src_left + pronoun_tokens + src_right + ['[SEP]']
    pronoun_idx = len(src_left) + 1
    pronoun_mask = [0]*len(src_tokens)
    pronoun_mask[pronoun_idx] = 1
    input_ids = []
    for cand in candidates:
      if cand:
        cand_ids = self.tokenizer.convert_tokens_to_ids(cand)
        token_ids = self.tokenizer.convert_tokens_to_ids(src_tokens + cand + ['[SEP]'])
        type_ids = [0]*len(src_tokens) + [1]*(len(cand)+1)
        mask_ids = [1]*len(token_ids)
        cand_mask = [0]*len(src_tokens) + [1]*(len(cand))
        input_ids.append([token_ids, mask_ids, type_ids, cand_mask, pronoun_mask.copy()])
      else:
        input_ids.append([[0], [0], [0], [0], [0]])

    task_ids = [tid for _ in range(len(input_ids))]
    return (input_ids, selected, task_ids)

from difflib import SequenceMatcher, get_close_matches

WSCRecord=namedtuple('WSCRecord', ['sentence', 'pron_idx', 'pron', 'selected', 'candidates'])
class WSCDataset(_ABCDataset):
  """
  Data set for Winograd Schema Challenge task
  """
  def __init__(self, tokenizer, datapaths, max_len=256, tid=None, topn=-1):
    super(WSCDataset, self).__init__(max_len, tid)
    self.datapaths = datapaths
    self.tokenizer = tokenizer
    self.topn = topn
    self.raw_data = []
    self.doc_count = 0
    self._load(datapaths, topn)

  def _load(self, datapaths, topn):
    doc_count = 0
    self.raw_data = []
    for src in datapaths:
      # for DPRD,WikiWSRC
      data = load_tsv(src, [0, 1, 2, 3, 4])
      doc_count += len(data)
      self.raw_data.extend(data)
      if doc_count > topn and topn>0:
        doc_count = topn
        break
    self.doc_count = doc_count

  def get_data(self, index):
    return WSCRecord(*self.raw_data[index])

  def get_example(self, index):
    data = self.get_data(index)
    pronoun_idx = int(data.pron_idx)
    pronoun = data.pron
    src = data.sentence
    left_src = src[:pronoun_idx].strip()
    right_src = src[pronoun_idx+len(pronoun):].strip()
    assert pronoun==src[pronoun_idx:len(pronoun)+pronoun_idx], data

    src_left = self.tokenizer.tokenize(convert_to_unicode(left_src))
    src_right = self.tokenizer.tokenize(convert_to_unicode(right_src))
    pronoun_tokens = self.tokenizer.tokenize(convert_to_unicode(data.pron))

    candidates = [self.tokenizer.tokenize(convert_to_unicode(c)) for c in data.candidates.split(',')]
    selected_idx = [i for i,c in enumerate(data.candidates.split(',')) if c.lower().strip()==data.selected.lower().strip()]
    assert len(selected_idx)==1, data
    assert len(candidates)==2, data
    selected = [0] * len(candidates)
    selected[selected_idx[0]]=1
    return self._make_inputs(src_left, src_right, pronoun_tokens, candidates, selected)

WSC273Record=namedtuple('WSC273Record', ['left', 'pron', 'right', 'candidates', 'selected'])
class WSC273Dataset(_ABCDataset):
  """
  Data set for Winograd Schema Challenge task
  """
  def __init__(self, tokenizer, datapaths, max_len=256, tid=None, topn=-1, max_candidates=2):
    super().__init__(max_len, tid)
    self.datapaths = datapaths
    self.tokenizer = tokenizer
    self.topn = topn
    self.raw_data = []
    self.doc_count = 0
    self.max_candidates=max_candidates
    self._load(datapaths, topn)

  def _load(self, datapaths, topn):
    doc_count = 0
    self.raw_data = []
    for src in datapaths:
      # for DPRD,WikiWSRC
      data = load_tsv(src, [0, 1, 2, 3, 4])
      doc_count += len(data)
      self.raw_data.extend(data)
      if doc_count > topn and topn>0:
        doc_count = topn
        break
    self.doc_count = doc_count

  def get_data(self, index):
    return WSC273Record(*self.raw_data[index])

  def get_example(self, index):
    data = self.get_data(index)
    # left, pron, right, candidates, selected
    src_left = self.tokenizer.tokenize(convert_to_unicode(data.left))
    src_right = self.tokenizer.tokenize(convert_to_unicode(data.right))
    pronoun_tokens = self.tokenizer.tokenize(convert_to_unicode(data.pron))

    candidates = [self.tokenizer.tokenize(convert_to_unicode(c)) for c in data.candidates.split(',')]
    selected_idx = int(data.selected)
    assert len(candidates)<=self.max_candidates, data
    candidates.extend([None]*(self.max_candidates - len(candidates)))
    selected = [0]*len(candidates)
    selected[selected_idx]=1
    return self._make_inputs(src_left, src_right, pronoun_tokens, candidates, selected)

WNLIRecord=namedtuple('WNLIRecord', ['sentence', 'hypersis', 'pron_idx' ,'pron', 'selected', 'candidates', 'label'])
class WNLIDataset(_ABCDataset):
  """
  Data set for Winograd Schema Challenge task
  """
  def __init__(self, tokenizer, datapaths, is_test=False, max_len=256, tid=None, topn=-1):
    super(WNLIDataset, self).__init__(max_len, tid)
    self.datapaths = datapaths
    self.tokenizer = tokenizer
    self.topn = topn
    self.raw_data = []
    self.doc_count = 0
    self._load(datapaths, topn, is_test)
    self.is_test = is_test
    self.nlp = spacy.load('en_core_web_sm')

  def _load(self, datapaths, topn, is_test):
    doc_count = 0
    self.raw_data = []
    # sentence, target, pronoun_idx, pronoun, selected, candidates, label
    fields = [1, 2, 3, 4, 5, 6, 7]
    if is_test:
      fields = [1, 2, 3, 4, 5, 6]
    for src in datapaths:
      data = load_tsv(src, fields)
      doc_count += len(data)
      self.raw_data.extend(data)
      if doc_count > topn and topn>0:
        doc_count = topn
        break
    self.doc_count = doc_count

  def get_data(self, index):
    if self.is_test:
      data = self.raw_data[index] + ['0']
    else:
      data = self.raw_data[index]
    return WNLIRecord(*data)

  def get_example(self, index):
    data = self.get_data(index)
    # source, pronoun_idx, pronoun, selected, candidates, label
    pronoun_idx = int(data.pron_idx)
    pronoun = data.pron
    src = data.sentence
    left_src = src[:pronoun_idx].strip()
    right_src = src[pronoun_idx+len(pronoun):].strip()
    src_left = self.tokenizer.tokenize(convert_to_unicode(left_src))
    src_right = self.tokenizer.tokenize(convert_to_unicode(right_src))

    pronoun_tokens = self.tokenizer.tokenize(convert_to_unicode(data.pron))

    selected = self.tokenizer.tokenize(convert_to_unicode(data.selected).lower().strip())
    selected_idx = 0

    candidates = [selected]
    label = 0
    if (not self.is_test):
      label = int(data.label)
    selected = [label]
    return self._make_inputs(src_left, src_right, pronoun_tokens, candidates, selected)

GAPRecord=namedtuple('GAPRecord', ['sentence', 'pron_idx', 'pron', 'selected', 'candidates'])
class GAPDataset(_ABCDataset):
  """
  Data set for Winograd Schema Challenge task
  """
  def __init__(self, tokenizer, datapaths, max_len=384, tid=None, topn=-1):
    super().__init__(max_len, tid)
    self.datapaths = datapaths
    self.tokenizer = tokenizer
    self.topn = topn
    self.raw_data = []
    self.doc_count = 0
    self._load(datapaths, topn)

  def _load(self, datapaths, topn):
    doc_count = 0
    self.raw_data = []
    for src in datapaths:
      # for DPRD,WikiWSRC
      data = load_tsv(src, [0, 1, 2, 3, 4])
      doc_count += len(data)
      self.raw_data.extend(data)
      if doc_count > topn and topn>0:
        doc_count = topn
        break
    self.doc_count = doc_count

  def get_data(self, index):
    return GAPRecord(*self.raw_data[index])

  def get_example(self, index):
    data = self.get_data(index)
    # source, pronoun_idx, selected, candidates, label
    pidx = int(data.pron_idx)
    
    src_left = self.tokenizer.tokenize(convert_to_unicode(data.sentence[:pidx]))
    src_right = self.tokenizer.tokenize(convert_to_unicode(data.sentence[pidx+len(data.pron):]))
    pronoun_tokens = self.tokenizer.tokenize(data.pron)
    candidates = [self.tokenizer.tokenize(convert_to_unicode(c)) for c in data.candidates.split('|')]
    selected_idx = [i for i,c in enumerate(data.candidates.split('|')) if c.lower().strip()==data.selected.lower().strip()]
    selected = [0]*len(candidates)
    selected[selected_idx[0]]=1
    return self._make_inputs(src_left, src_right, pronoun_tokens, candidates, selected)

def wsc_accuracy(logits, labels):
  # bxc
  count = 0
  for g,l in zip(logits, labels):
    prd = np.argmax(g)
    if l[prd] == 1:
      count+=1
  return count/len(labels)

class _ABCTask(object):
  def __init__(self, tokenizer):
    self.tokenizer = tokenizer
  
  def get_train_dataset(self, data_dir, maxlen, input_type=None):
    raise NotImplemented('This method must be implemented by sub class.')

  def get_test_dataset(self, data_dir, maxlen, input_type=None):
    return []

  def get_dev_dataset(self, data_dir, maxlen, input_type=None):
    return []

  def get_metric_fn(self):
    def metric_fn(logits, labels, *argv, **kwargs):
      return OrderedDict(accuracy=wsc_accuracy(logits, labels))
    return metric_fn

class DPRDTask(_ABCTask):
  def __init__(self, tokenizer):
    super().__init__(tokenizer)

  def get_train_dataset(self, data_dir, maxlen=128, input_type=None):
    paths = glob(os.path.join(data_dir, 'train_annotated.tsv'))
    return WSCDataset(self.tokenizer, paths, max_len=maxlen, tid=input_type)

  def get_dev_dataset(self, data_dir, maxlen, input_type=None):
    eval_set = [
        self._mk_eval('DPRD-test', data_dir, 'test_annotated.tsv', maxlen, input_type=input_type)
        ]
    return eval_set

  def get_test_dataset(self, data_dir, maxlen, input_type=None):
    eval_set = [
        #self._mk_eval('DPRD-test', data_dir, 'test_annotated.tsv', maxlen)
        ]
    return eval_set

  def _mk_eval(self, name, data_dir, data, maxlen, input_type=None):
    paths = glob(os.path.join(data_dir, data))
    dataset = WSCDataset(self.tokenizer, paths, max_len=maxlen, tid=input_type)
    return EvalData(name=name, examples=dataset, \
          metrics_fn=self.get_metric_fn())

class WSC273Task(_ABCTask):
  def __init__(self, tokenizer):
    super().__init__(tokenizer)

  def get_dev_dataset(self, data_dir, maxlen, input_type=None):
    eval_set = [
        self._mk_eval('wsc273-test', data_dir, 'wsc273.tsv', maxlen, input_type=input_type),
        self._mk_eval('pdp60-test', data_dir, 'pdp60.tsv', maxlen, max_candidates=5, input_type=input_type)
        ]
    return eval_set

  def get_test_dataset(self, data_dir, maxlen, input_type=None):
    eval_set = [
        #self._mk_eval('wsc273-test', data_dir, 'wsc273.tsv', maxlen)
        ]
    return eval_set

  def _mk_eval(self, name, data_dir, data, maxlen, max_candidates=2, input_type=None):
    paths = glob(os.path.join(data_dir, data))
    dataset = WSC273Dataset(self.tokenizer, paths, max_len=maxlen, max_candidates=max_candidates, \
        tid=input_type)
    return EvalData(name=name, examples=dataset, \
          metrics_fn=self.get_metric_fn())

class WikiWSCRTask(_ABCTask):
  def __init__(self, tokenizer):
    super().__init__(tokenizer)

  def get_train_dataset(self, data_dir, tokenizer, maxlen=128, input_type=None):
    paths = glob(os.path.join(data_dir, 'train_annotated.tsv'))
    return WSCDataset(self.tokenizer, paths, max_len=maxlen)

class WNLITask(_ABCTask):
  def __init__(self, tokenizer):
    super().__init__(tokenizer)
    self.threshold = 0.0
    self.thred_dict = {}

  def get_train_dataset(self, data_dir, maxlen=128, input_type=None):
    paths = glob(os.path.join(data_dir, 'train_annotated.tsv'))
    return WNLIDataset(self.tokenizer, paths, max_len=maxlen, tid=input_type)

  def get_dev_dataset(self, data_dir, maxlen, input_type=None):
    eval_set = [
        self._mk_eval('wnli-dev', data_dir, 'dev_annotated.tsv', maxlen, scan_thred=True, \
            rank=True, input_type=input_type)]
    return eval_set

  def get_test_dataset(self, data_dir, maxlen, input_type=None):
    eval_set = [
        self._mk_eval('wnli-test', data_dir, 'test_annotated.tsv', maxlen, is_test=True, input_type=input_type)]
    return eval_set

  def _mk_eval(self, name, data_dir, data, maxlen, is_test=False, scan_thred=False, rank=False, \
      input_type=None):
    paths = glob(os.path.join(data_dir, data))
    dataset = WNLIDataset(self.tokenizer, paths, max_len=maxlen, is_test=is_test, tid=input_type)
    if name != 'wnli-rank':
      pred_fn = self._predict_fn(dataset)
    else:
      pred_fn = self._rank_fn(dataset)

    return EvalData(name=name, examples=dataset, \
          metrics_fn=self.get_metric_fn(scan_thred, dataset, rank), predict_fn=pred_fn)

  def _rank_fn(self, datas):
    def predict_fn(logits, output_dir, name, prefix, *argv, **kwargs):
      tag = None
      if 'tag' in kwargs:
        tag = kwargs['tag']
      output=os.path.join(output_dir, 'submit-rank-{}-{}-{}.tsv'.format(name, prefix, tag))
      with open(output, 'w') as fs:
        fs.write('index\tpred\n')
        start = 0
        for i,l in enumerate(logits):
          ids, sel = datas.get_example(i)
          cid = np.argmax(l)
          cands_mask=[ids[j*4+4] for j in range((len(ids)-1)//4)]
          for j,s in enumerate(cands_mask):
            if sum(s)==0:
              break
            pred = 1 if cid==j else 0
            fs.write(f'{start}\t{pred}\n')
            start+=1
    return predict_fn

  def _predict_fn(self, datas):
    def predict_fn(logits, output_dir, name, prefix, *argv, **kwargs):
      logits = np.reshape(logits, [-1])
      tag = None
      if 'tag' in kwargs:
        tag = kwargs['tag']
      th = self.threshold if not tag or tag not in self.thred_dict else self.thred_dict[tag]

      logger.info(f'Predict with [{name}][prefix][{tag}] {th:0.02}')
      output=os.path.join(output_dir, 'submit-{}-{}-{}{:0.02}.tsv'.format(name, prefix, tag, th))
      with open(output, 'w') as fs:
        fs.write('index\tpred\n')
        for i,l in enumerate(logits):
          pred = 1 if l>th else 0
          fs.write(f'{i}\t{pred}\n')

      group = None
      count = 0
      result = []
      for i in range(len(datas)):
        d = datas.get_data(i)
        if group is None:
          group = []
          group.append([d, logits[i], i])
        elif group[-1][0].sentence==d.sentence and group[-1][0].pron_idx==d.pron_idx:
          group.append([d, logits[i], i])
        else:
          ll = [g[1] for g in group]
          m = np.argmax(ll)
          labels = [0]*len(group)
          labels[m] = 1
          result.extend(labels)
          group=[[d,logits[i],i]]
      if len(group)>0:
        ll = [g[1] for g in group]
        m = np.argmax(ll)
        labels = [0]*len(group)
        labels[m] = 1
        result.extend(labels)

      output=os.path.join(output_dir, 'submit-rank-{}-{}-{}.tsv'.format(name, prefix, tag))
      with open(output, 'w') as fs:
        fs.write('index\tpred\n')
        for i,pred in enumerate(result):
          fs.write(f'{i}\t{pred}\n')

    return predict_fn

  def get_metric_fn(self, threshold_scan=False, data=None, rank=False):
    def metric_fn(logits, labels, *argv, **kwargs):
      quiet = False
      tag = None
      if 'quiet' in kwargs:
        quiet = kwargs['quiet']
      if 'tag' in kwargs:
        tag = kwargs['tag']
      if rank:
        return OrderedDict(accuracy=self.wnli_accuracy(logits, labels, threshold_scan, quiet, tag),
            rank_acc=self.wnli_rank_acc(logits, labels, data, quiet, tag)
            )
      else:
        return OrderedDict(accuracy=self.wnli_accuracy(logits, labels, threshold_scan, quiet, tag))

    return metric_fn

  def wnli_rank_acc(self, logits, labels, data, quiet=False, tag=None):
    # bxc
    labels = np.reshape(labels, [-1])
    logits = np.reshape(logits, [-1])
    group = None
    count = 0
    for i in range(len(data)):
      d = data.get_data(i)
      if group is None:
        group = []
        group.append((d, logits[i], labels[i], i))
      elif group[-1][0].sentence==d.sentence and group[-1][0].pron_idx==d.pron_idx:
        group.append((d, logits[i], labels[i], i))
      else:
        ll = [g[1] for g in group]
        m = np.argmax(ll)
        if ll[m]<0:
          count += len(group)-1
        elif group[m][2]==1:
          count += len(group)
        else:
            count += len(group)-2
        group=[(d,logits[i], labels[i], i)]
    if len(group)>0:
      ll = [g[1] for g in group]
      m = np.argmax(ll)
      if ll[m]<0:
        count += len(group)-1
      elif group[m][2]==1:
        count += len(group)
      else:
        count += len(group)-2

    return count/len(labels)

  def wnli_accuracy(self, logits, labels, threshold_scan, quiet=False, tag=None):
    # bxc
    labels = np.reshape(labels, [-1])
    def acc(thred):
      count = 0
      for g,l in zip(logits, labels):
        idx = 0
        if g[idx]>thred:
          count += 1 if l>0 else 0
        else:
          count += 1 if l==0 else 0
      return count/len(labels)

    def thred_scan(reverse=True):
      if reverse:
        steps = np.arange(1,0,-0.01)
      else:
        steps = np.arange(0,1,0.01)

      best_th = 0.95
      best_score = 0
      for th in steps:
        score = acc(th)
        if score > best_score:
          best_score = score
          best_th = th
      return best_score,best_th

    mp = np.mean([l for l,k in zip(logits, labels) if k>0])
    mn = np.mean([l for l,k in zip(logits, labels) if k==0])
    th = self.threshold if not tag or tag not in self.thred_dict else self.thred_dict[tag]
    if not quiet:
      logger.info(f'[{tag}] Mean sim score={np.mean(logits):0.02}|[+]{mp:0.02}|[-]{mn:0.02}; th={th:0.03}; acc@0.5={acc(0.5):0.02}')
    if threshold_scan:
      score, th = thred_scan(reverse=True)
      score2, th2 = thred_scan(reverse=False)
      best_th = (th+th2)/2
      if not quiet:
        logger.info(f'[{tag}] Best score: BWD={score:0.03}@{th:0.02}, FWD={score2:0.03}@{th2:0.02}, Avg={score:0.03}@{best_th:0.02}')
      self.threshold = best_th
      if tag:
        self.thred_dict[tag] = best_th
      return score
    else:
      return acc(th)

class GAPTask(_ABCTask):
  def __init__(self, tokenizer):
    super().__init__(tokenizer)

  # max len 256
  def get_train_dataset(self, data_dir, maxlen=384, input_type=None):
    paths = glob(os.path.join(data_dir, 'train_annotated.tsv'))
    return GAPDataset(self.tokenizer, paths, max_len=maxlen, tid=input_type)

  def get_dev_dataset(self, data_dir, maxlen, input_type=None):
    eval_set = [
        self._mk_eval('GAP-test', data_dir, 'test_annotated.tsv', maxlen, input_type=input_type),
        self._mk_eval('GAP-dev', data_dir, 'dev_annotated.tsv', maxlen, input_type=input_type)
        ]
    return eval_set

  def get_test_dataset(self, data_dir, maxlen, input_type=None):
    eval_set = [
        self._mk_eval('GAP-test', data_dir, 'test_annotated.tsv', maxlen, input_type=input_type)]
    return eval_set

  def _mk_eval(self, name, data_dir, data, maxlen, input_type=None):
    paths = glob(os.path.join(data_dir, data))
    dataset = GAPDataset(self.tokenizer, paths, max_len=maxlen, tid=input_type)
    return EvalData(name=name, examples=dataset, \
          metrics_fn=self.get_metric_fn())
