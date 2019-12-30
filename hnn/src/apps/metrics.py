import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score,f1_score
from scipy.stats import pearsonr, spearmanr
from statistics import *
from scipy.special import softmax

def metric_multi_accuracy(logits, labels, options_num):
  logits = np.reshape(softmax(logits, -1)[:,1], (len(logits)//options_num, options_num))
  labels = np.argmax(np.reshape(labels, (len(labels)//options_num, options_num)),-1)
  return metric_accuracy(logits, labels)

def metric_accuracy(logits, labels):
  predicts = np.argmax(logits, axis=1)
  return accuracy_score(labels, predicts)

def metric_f1(logits, labels):
  predicts = np.argmax(logits, axis=1)
  return f1_score(labels, predicts)

def metric_mcc(logits, labels):
  predicts = np.argmax(logits, axis=1)
  return matthews_corrcoef(labels, predicts)
