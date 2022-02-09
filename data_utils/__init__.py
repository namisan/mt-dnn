import json
import numpy as np

from data_utils.task_def import TaskType, DataFormat
import tasks


def load_data(file_path, task_def):
    data_format = task_def.data_type
    task_type = task_def.task_type
    label_dict = task_def.label_vocab
    if task_type == TaskType.Ranking:
        assert data_format == DataFormat.PremiseAndMultiHypothesis

    rows = []
    for line in open(file_path, encoding="utf-8"):
        fields = line.strip("\n").split("\t")
        if data_format == DataFormat.PremiseOnly:
            assert len(fields) == 3
            row = {"uid": fields[0], "label": fields[1], "premise": fields[2]}
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            assert len(fields) == 4
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3],
            }
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            assert len(fields) > 5
            row = {
                "uid": fields[0],
                "ruid": fields[1].split(","),
                "label": fields[2],
                "premise": fields[3],
                "hypothesis": fields[4:],
            }
        elif data_format == DataFormat.Seqence:
            row = {
                "uid": fields[0],
                "label": eval(fields[1]),
                "premise": eval(fields[2]),
            }

        elif data_format == DataFormat.MRC:
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3],
            }
        else:
            raise ValueError(data_format)

        task_obj = tasks.get_task_obj(task_def)
        if task_obj is not None:
            row["label"] = task_obj.input_parse_label(row["label"])
        elif task_type == TaskType.Ranking:
            labels = row["label"].split(",")
            if label_dict is not None:
                labels = [label_dict[label] for label in labels]
            else:
                labels = [float(label) for label in labels]
            row["label"] = int(np.argmax(labels))
            row["olabel"] = labels
        elif task_type == TaskType.Span:
            pass  # don't process row label
        elif task_type == TaskType.SeqenceLabeling:
            assert type(row["label"]) is list
            row["label"] = [label_dict[label] for label in row["label"]]

        rows.append(row)
    return rows


def load_score_file(score_path, n_class):
    sample_id_2_pred_score_seg_dic = {}
    score_obj = json.loads(open(score_path, encoding="utf-8").read())
    assert (len(score_obj["scores"]) % len(score_obj["uids"]) == 0) and (
        len(score_obj["scores"]) / len(score_obj["uids"]) == n_class
    ), "scores column size should equal to sample count or multiple of sample count (for classification problem)"

    scores = score_obj["scores"]
    score_segs = [
        scores[i * n_class : (i + 1) * n_class] for i in range(len(score_obj["uids"]))
    ]
    for sample_id, pred, score_seg in zip(
        score_obj["uids"], score_obj["predictions"], score_segs
    ):
        sample_id_2_pred_score_seg_dic[sample_id] = (pred, score_seg)
    return sample_id_2_pred_score_seg_dic
