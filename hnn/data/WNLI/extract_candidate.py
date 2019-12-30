# coding: utf-8
# %load extract_candidate.py
import spacy
import numpy as np
nlp=spacy.load('en_core_web_lg')

with open('train_annotated.tsv') as fs:
  train = [l.strip().split('\t') for l in fs][1:]

with open('dev_annotated.tsv') as fs:
  dev = [l.strip().split('\t') for l in fs][1:]

merged = dev + train

group={}
for i in merged:
  key=f'{i[1]}${i[3]}${i[4]}'
  if key not in group:
    group[key]=[]
  group[key].append(i)

def token_equal(src, tgt):
  if np.linalg.norm(src.vector)==0 or np.linalg.norm(tgt.vector)==0:
    return src.text.lower()==tgt.text.lower()
  else:
    return src.similarity(tgt)>0.98

def left_matching(src, tgt, src_offset = 0, tgt_offset = 0):
  s = src_offset
  t = tgt_offset
  while s<len(src) and t<len(tgt):
    if token_equal(src[-(s+1)] , tgt[-(t+1)]):
      s+=1
      t+=1
    else:
      break
  return s,t

def right_matching(src, tgt, src_offset = 0, tgt_offset = 0):
  s = src_offset
  t = tgt_offset
  while s<len(src) and t<len(tgt):
    if token_equal(src[s], tgt[t]):
      s+=1
      t+=1
    else:
      break
  return s,t

def extract(src, pidx, p, tgt):
  left = src[:pidx].strip()
  right = src[pidx+len(p):].strip()
  left_tokens = nlp(left)
  right_tokens = nlp(right)
  tgt_tokens = nlp(tgt)
  # left matching
  s = 0
  left_matches = []
  for s in (0,1):
    if s==1 and (left_tokens[-1].text not in ['of']):
      break
    t = 0
    while t<len(tgt_tokens):
      i,j = left_matching(left_tokens, tgt_tokens, s, t)
      i -= s
      if i>0:
        left_matches.append((i, j, t))
      elif t==len(tgt_tokens)-1:
        left_matches.append((i, j, t+1))
      t += 1

  s = 0
  right_matches = []
  for s in [0]:
    t = 0
    while t<len(tgt_tokens):
      i,j = right_matching(right_tokens, tgt_tokens, s, t)
      i -= s
      if i>0 or t==len(tgt_tokens)-1:
        right_matches.append((i, j, t))
      t += 1

  return left_tokens, right_tokens, tgt_tokens, left_matches, right_matches
for i, k in enumerate(group):
    for d in group[k]:
        left_tok, right_tok, tgt_tok, left_m, right_m = extract(d[1], int(d[3]), d[4], d[2])
        matches = []
        for lm in left_m:
            rm = sorted([x for x in right_m if x[-1] - 1>len(tgt_tok)-(lm[-1]+1)], key=lambda y:y[0])
            if len(rm)>0:
                rm = rm[-1]
                matches.append((lm, rm, lm[0]+rm[0]))
        match = sorted(matches, key=lambda x:x[-1])[-1]
        left_text = len(tgt_tok) - (match[0][-1]+1) + 1
        right_text = match[1][-1]
        span = d[2][tgt_tok[left_text].idx:tgt_tok[right_text].idx].strip()
        assert len(span)>0, f'[{i}] {d}'
        rest = list(tgt_tok[0:max(left_text-1, 0)]) + list(tgt_tok[right_text:])
        rest = ''.join([t.text_with_ws for t in rest])
        d.append([span, match, rest])
    assert all((d[-1][-1].lower()==group[k][0][-1][-1].lower() for d in group[k])), f'[{i}], {k}'

with open('eval_annotated_rank.tsv', 'w') as fs:
  fs.write('id\tsrc\ttgt\tpronoun_idx\tpronoun\tselected\tcandidates\tlabel\n')
  for i,k in enumerate(group):
    for v in group[k]:
      fs.write(f'{i}\t{v[1]}\t{v[2]}\t{v[3]}\t{v[4]}\t{v[-1][0]}\t{v[6]}\t{v[7]}\n')
