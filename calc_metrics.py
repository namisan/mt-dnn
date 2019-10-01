import argparse

from data_utils import load_data, load_score_file
from data_utils.metrics import calc_metrics
from experiments.exp_def import TaskDefs

parser = argparse.ArgumentParser()
parser.add_argument("--task_def", type=str, default="experiments/glue/glue_task_def.yml")
parser.add_argument("--task", type=str)
parser.add_argument("--std_input", type=str)
parser.add_argument("--score", type=str)


def generate_golds_predictions_scores(sample_id_2_pred_score_seg_dic, sample_objs):
    sample_id_2_label_dic = {}

    for sample_obj in sample_objs:
        sample_id, label = sample_obj["uid"], sample_obj["label"]
        sample_id_2_label_dic[sample_id] = label

    assert set(sample_id_2_label_dic.keys()) == set(sample_id_2_pred_score_seg_dic.keys())

    golds = []
    predictions = []
    scores = []
    for sample_id, label in sample_id_2_label_dic.items():
        golds.append(label)
        pred, score_seg = sample_id_2_pred_score_seg_dic[sample_id]
        predictions.append(pred)
        scores.extend(score_seg)
    return golds, predictions, scores

args = parser.parse_args()

task_def_path = args.task_def
task = args.task
task_defs = TaskDefs(task_def_path)

n_class = task_defs.n_class_map[task]
sample_id_2_pred_score_seg_dic = load_score_file(args.score, n_class)

data_type = task_defs.data_type_map[task]
task_type = task_defs.task_type_map[task]
label_mapper = task_defs.global_map.get(task, None)
sample_objs = load_data(args.std_input, data_type, task_type, label_mapper)

golds, predictions, scores = generate_golds_predictions_scores(sample_id_2_pred_score_seg_dic, sample_objs)

metrics = calc_metrics(task_defs.metric_meta_map[task], golds, predictions, scores)
print(metrics)
