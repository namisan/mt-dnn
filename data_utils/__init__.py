import json

import numpy as np

from data_utils.task_def import TaskType, DataFormat


def load_data(file_path, data_format, task_type, label_dict=None):
    """
    :param file_path:
    :param data_format:
    :param task_type:
    :param label_dict: map string label to numbers.
        only valid for Classification task or ranking task.
        For ranking task, better label should have large number
    :return:
    """
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
            row = {"uid": fields[0], "label": fields[1], "premise": fields[2], "hypothesis": fields[3]}
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            assert len(fields) > 5
            row = {"uid": fields[0], "ruid": fields[1].split(","), "label": fields[2], "premise": fields[3],
                   "hypothesis": fields[4:]}
        else:
            raise ValueError(data_format)

        if task_type == TaskType.Classification:
            if label_dict is not None:
                row["label"] = label_dict[row["label"]]
            else:
                row["label"] = int(row["label"])
        elif task_type == TaskType.Regression:
            row["label"] = float(row["label"])
        elif task_type == TaskType.Ranking:
            labels = row["label"].split(",")
            if label_dict is not None:
                labels = [label_dict[label] for label in labels]
            else:
                labels = [float(label) for label in labels]
            row["label"] = int(np.argmax(labels))
            row["olabel"] = labels

        rows.append(row)
    return rows


def load_score_file(score_path, n_class):
    sample_id_2_pred_score_seg_dic = {}
    score_obj = json.loads(open(score_path, encoding="utf-8").read())
    assert (len(score_obj["scores"]) % len(score_obj["uids"]) == 0) and \
           (len(score_obj["scores"]) / len(score_obj["uids"]) == n_class), \
        "scores column size should equal to sample count or multiple of sample count (for classification problem)"

    scores = score_obj["scores"]
    score_segs = [scores[i * n_class: (i+1) * n_class] for i in range(len(score_obj["uids"]))]
    for sample_id, pred, score_seg in zip(score_obj["uids"], score_obj["predictions"], score_segs):
        sample_id_2_pred_score_seg_dic[sample_id] = (pred, score_seg)
    return sample_id_2_pred_score_seg_dic