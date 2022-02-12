import collections
import json


def load_xnli(file, header=True):
    lang_dict = collections.defaultdict(list)
    label_dict = {}
    cnt = 0
    label_map = {"contradiction": 0, "neutral": 1, "entailment": 2}
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            # if blocks[1] == '-': continue
            lab = label_map[blocks[1]]
            if lab is None:
                import pdb

                pdb.set_trace()
            uid = str(cnt)
            label_dict[uid] = lab
            lang_dict[blocks[0]].append(uid)
            cnt += 1
        print(cnt)
    return lang_dict, label_dict


fin = "data/XNLI/xnli.dev.tsv"

fout = "data/XNLI/xnli_dev_cat.json"

lang_dict, label_dict = load_xnli(fin)
data = {"lang_map": lang_dict, "label_map": label_dict}
with open(fout, "w") as f:
    json.dump(data, f)
# cnt = 0
# for key, val in lang_dict.items():
#    cnt += len(val)
#    for uid in val:
#        assert uid in label_dict
# print(cnt)
