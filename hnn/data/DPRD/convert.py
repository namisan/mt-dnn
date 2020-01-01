# coding: utf-8
import spacy
import pdb
from difflib import SequenceMatcher
nlp=spacy.load('en_core_web_sm')
def load(path):
    data=[]
    extend_data=[]
    with open(path) as fs:
        lines = [l.strip() for l in fs]
        if lines[-1] !='':
            lines.append('')
        def index(src, tgt):
            s_doc = nlp(src)
            t_doc = nlp(tgt)
            sm = SequenceMatcher(None, [t.text for t in s_doc], [t.text for t in t_doc])
            mb = []
            of = 0
            while of < len(s_doc):
              m = sm.find_longest_match(of, len(s_doc), 0, len(t_doc))
              if m.size==len(t_doc):
                mb.append(m)
                of = m.a + len(t_doc)
              else:
                break
            assert len(mb)>0, src + '|' + tgt
            return [s_doc[m.a].idx for m in mb]
        nid = 0
        for i,l in enumerate(lines):
            if l.strip()=='':
                #sentence, pronoun_idx, pronoun, coref, candidates
                try:                                       
                    indexs = index(lines[i-4], lines[i-3])
                    for idx in indexs[:1]:
                        data.append([lines[i-4], idx, lines[i-3], lines[i-1], lines[i-2]])
                    if len(indexs)>1:
                      for idx in indexs[1:]:
                          extend_data.append([nid, lines[i-4], idx, lines[i-3], lines[i-1], lines[i-2]])
                    nid += 1
                except ValueError as ex:                    
                    print(f'{i}:{lines[i-4]}:{lines[i-3]}')
    return data, extend_data

def convert(src, output):
  data, ex = load(src)
  with open(output, 'w') as fs:
    fs.write(f'sentence\tpronoun_idx\tpronoun\tcoref\tcandidates\n')
    for d in data:
      l='\t'.join([str(i) for i in d])
      fs.write(f'{l}\n')

  # there are some sentences with multiple pronouns, need human review to fix the correct antecedant
  with open(output + ".ext", 'w') as fs:
    fs.write(f'id\tsentence\tpronoun_idx\tpronoun\tcoref\tcandidates\n')
    for d in ex:
      l='\t'.join([str(i) for i in d])
      fs.write(f'{l}\n')


convert('train.c.txt', 'train_annotated.tsv')
convert('test.c.txt', 'test_annotated.tsv')

