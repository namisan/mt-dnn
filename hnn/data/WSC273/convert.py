# coding: utf-8
import xmltodict
with open('WSCollection.xml') as fs:
    doc=xmltodict.parse(fs.read())
    
def extract(schema):
    left=schema['text']['txt1']
    right=schema['text']['txt2']
    pron=schema['text']['pron']
    candidates=schema['answers']['answer']
    answer=schema['correctAnswer'].strip(' .')
    selected=0 if answer=='A' else 1
    assert answer in ['A', 'B'], answer
    return (left, pron, right, candidates, selected)
    
wsc=[]
for p in doc['collection']['schema']:
    wsc.append(extract(p))

with open('wsc273.tsv', 'w') as fs:
  fs.write('left\tpron\tright\tcandidates\tselected\n')
  for left,pron,right,candidates,selected in wsc:
    left=left.replace('\n', ' ').replace('\r', ' ')
    right=right.replace('\n', ' ').replace('\r', ' ')
    candidates=[c.replace('\n', ' ').replace('\r', ' ') for c in candidates]
    assert '\t' not in left
    assert '\t' not in right
    assert all(['\t' not in c for c in candidates])
    assert all(['\n' not in c for c in candidates])
    assert all([',' not in c for c in candidates])
    fs.write(f"{left}\t{pron}\t{right}\t{','.join(candidates)}\t{selected}\n")
