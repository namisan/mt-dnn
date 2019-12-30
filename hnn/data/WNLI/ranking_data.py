# coding: utf-8
# %load ranking_data.py
with open('test_annotated.tsv') as fs:
    data=[l.strip().split('\t') for l in fs]
    test=data[1:]
    
collections=[]
group=[]
prev=None
for t in test:
    if prev is None:
        prev=t
        group = [t]
    elif t[1]==prev[1] and t[3]==prev[3] and t[4]==prev[4]:
        group.append(t)        
        prev=t
    else:
        collections.append(group)
        group = [t]
        prev=t
if len(group)>0:
    collections.append(group)        
        
for g in collections:
    assert len(set(u[5] for u in g))==len(g), g
    cands='|'.join([u[5] for u in g])
    for u in g:
        u[-1]=cands
              
    
for g in collections:
    for i,u in enumerate(g):
        assert i==u[-1].split('|').index(u[5]),g
        
reformat=[]
for g in collections:
    indexes='|'.join([u[0] for u in g])
    
    for i,u in enumerate(g):
        assert i==u[-1].split('|').index(u[5]),g
    reformat.append(g[0] + [indexes])
    
len(reformat)
with open('test_ranking.tsv', 'w') as fs:
    fs.write('\t'.join(data[0])+'\tindex\n')
    for r in reformat:
        fs.write('\t'.join(r)+'\n')
        
l=[len(k[-2].split('|')) for k in reformat]
assert sum(l)==len(test)
