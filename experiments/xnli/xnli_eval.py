import json
from sklearn.metrics import accuracy_score
import argparse


def compute_acc(predicts, labels):
    return 100.0 * accuracy_score(labels, predicts)


def load(path):
    with open(path, "r") as f:
        return json.load(f)


def compute(scores, labels):
    lang_map = labels["lang_map"]
    label_map = labels["label_map"]
    uids = scores["uids"]
    predictions = scores["predictions"]
    grounds = []
    machines = []
    predictions_map = {}
    for uid, pred in enumerate(predictions):
        uid = str(uid)
        grounds.append(pred)
        machines.append(label_map[uid])
        predictions_map[uid] = pred
    metrics = {"all": compute_acc(machines, grounds)}
    print("total size: {}".format(len(machines)))
    for lan, subuids in lang_map.items():
        sub_machine = [predictions_map[i] for i in subuids]
        sub_ground = [label_map[i] for i in subuids]
        metrics[lan] = compute_acc(sub_machine, sub_ground)
        print("size of {}: {}".format(lan, len(sub_machine)))
    print(metrics)


parser = argparse.ArgumentParser()
parser.add_argument("--fscore", type=str, required=True)
parser.add_argument("--fcat", type=str, required=True)
args = parser.parse_args()

# score_path = "models/xnli_dev_scores_0.json"
# label_path = "data/XNLI/xnli_dev_cat.json"
score_path = args.fscore
label_path = args.fcat
scores = load(score_path)
labels = load(label_path)
compute(scores, labels)
