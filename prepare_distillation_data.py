import argparse

from data_utils import load_score_file
from experiments.exp_def import TaskDefs

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_def", type=str, default="experiments/glue/glue_task_def.yml"
)
parser.add_argument("--task", type=str)
parser.add_argument(
    "--add_soft_label",
    action="store_true",
    help="without this option, we replace hard label with soft label",
)

parser.add_argument("--std_input", type=str)
parser.add_argument("--score", type=str)
parser.add_argument("--std_output", type=str)

args = parser.parse_args()

task_def_path = args.task_def
task = args.task
task_defs = TaskDefs(task_def_path)

n_class = task_defs.get_task_def(task).n_class
sample_id_2_pred_score_seg_dic = load_score_file(args.score, n_class)

with open(args.std_output, "w", encoding="utf-8") as out_f:
    for line in open(args.std_input, encoding="utf-8"):
        fields = line.strip("\n").split("\t")
        sample_id = fields[0]
        target_score_idx = 1  # TODO: here we assume binary classification task
        score = sample_id_2_pred_score_seg_dic[sample_id][1][target_score_idx]
        if args.add_soft_label:
            fields = fields[:2] + [str(score)] + fields[2:]
        else:
            fields[1] = str(score)
        out_f.write("\t".join(fields))
        out_f.write("\n")
