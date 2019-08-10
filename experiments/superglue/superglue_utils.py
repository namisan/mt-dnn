# Copyright (c) Microsoft. All rights reserved.
import os
import json
from sys import path
from random import shuffle
path.append(os.getcwd())
from data_utils.metrics import Metric, METRIC_FUNC
from data_utils.task_def import DataFormat

def load_cb(file, label_dict):
    """Loading data of CB/RTE
    """
    rows = []
    with open(file, encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            label = label_dict[data['label']] if 'label' in data else 0
            uid = data['idx']
            sample = {'uid': uid, 'premise': data['premise'], 'hypothesis': data['hypothesis'], 'label': label}
            rows.append(sample)
    return rows

def load_boolq(file):
    # load boolq
    rows = []
    with open(file, encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            label = data['label'] if 'label' in data else False
            label = 1 if label else 0
            uid = data['idx']
            sample = {'uid': uid, 'premise': data['passage'], 'hypothesis': data['question'], 'label': label}
            rows.append(sample)
    return rows

def load_copa(file):
    rows = []
    with open(file, encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            label = data['label'] if 'label' in data else 0
            uid = data['idx']
            hyp1 = '{} {}'.format(data['question'], data['choice1'])
            hyp2 = '{} {}'.format(data['question'], data['choice2'])
            sample = {'uid': uid, 'ruid':'{},{}'.format(uid,uid), 'premise': data['premise'], 'hypothesis': [hyp1, hyp2], 'label': label}
            rows.append(sample)
    return rows


def load_record(file):
    pass

def load_wsc(file):
    pass

def load_multirc(file):
    rows = []
    with open(file, encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            pidx = data['idx']
            passage = data['passage']['text']
            questionts = data['passage']['questions']
            assert type(questionts) is list
            for question in questionts:
                q = question['question']
                qidx = question['idx']
                answers = question['answers']
                for answer in answers:
                    a = answer['text']
                    hyp = '{} ||| {}'.format(q, a)
                    aidx = answer['idx']
                    label = answer['label'] if 'label' in answer else 0
                    uid = str((pidx, qidx, aidx))
                    sample = {'uid': uid, 'premise': passage, 'hypothesis': hyp, 'label': label}
                    rows.append(sample)
        return rows


def dump_rows(rows, out_path):
    """
    output files should have following format
    :param rows:
    :param out_path:
    :return:
    """

    def detect_format(row):
        data_format = DataFormat.PremiseOnly
        if "hypothesis" in row:
            hypo = row["hypothesis"]
            if isinstance(hypo, str):
                data_format = DataFormat.PremiseAndOneHypothesis
            else:
                assert isinstance(hypo, list)
                data_format = DataFormat.PremiseAndMultiHypothesis
        return data_format

    with open(out_path, "w", encoding="utf-8") as out_f:
        row0 = rows[0]
        data_format = detect_format(row0)
        for row in rows:
            assert data_format == detect_format(row), row
            if data_format == DataFormat.PremiseOnly:
                for col in ["uid", "label", "premise"]:
                    if "\t" in str(row[col]):
                        import pdb; pdb.set_trace()
                out_f.write("%s\t%s\t%s\n" % (row["uid"], row["label"], row["premise"]))
            elif data_format == DataFormat.PremiseAndOneHypothesis:
                for col in ["uid", "label", "premise", "hypothesis"]:
                    if "\t" in str(row[col]):
                        import pdb; pdb.set_trace()
                out_f.write("%s\t%s\t%s\t%s\n" % (row["uid"], row["label"], row["premise"], row["hypothesis"]))
            elif data_format == DataFormat.PremiseAndMultiHypothesis:
                for col in ["uid", "label", "premise"]:
                    if "\t" in str(row[col]):
                        import pdb; pdb.set_trace()
                hypothesis = row["hypothesis"]
                for one_hypo in hypothesis:
                    if "\t" in str(one_hypo):
                        import pdb; pdb.set_trace()
                hypothesis = "\t".join(hypothesis)
                out_f.write("%s\t%s\t%s\t%s\t%s\n" % (row["uid"], row["ruid"], row["label"], row["premise"], hypothesis))
            else:
                raise ValueError(data_format)

def submit(path, data, label_dict=None):
    header = 'index\tprediction'
    with open(path ,'w') as writer:
        predictions, uids = data['predictions'], data['uids']
        writer.write('{}\n'.format(header))
        assert len(predictions) == len(uids)
        # sort label
        paired = [(int(uid), predictions[idx]) for idx, uid in enumerate(uids)]
        paired = sorted(paired, key=lambda item: item[0])
        for uid, pred in paired:
            if label_dict is None:
                writer.write('{}\t{}\n'.format(uid, pred))
            else:
                assert type(pred) is int
                writer.write('{}\t{}\n'.format(uid, label_dict[pred]))

def eval_model(model, data, metric_meta, use_cuda=True, with_label=True):
    data.reset()
    if use_cuda:
        model.cuda()
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for batch_meta, batch_data in data:
        score, pred, gold = model.predict(batch_meta, batch_data)
        predictions.extend(pred)
        golds.extend(gold)
        scores.extend(score)
        ids.extend(batch_meta['uids'])
    if with_label:
        for mm in metric_meta:
            metric_name = mm.name
            metric_func = METRIC_FUNC[mm]
            if mm in (Metric.ACC, Metric.F1, Metric.MCC):
                metric = metric_func(predictions, golds)
            else:
                metric = metric_func(scores, golds)
            metrics[metric_name] = metric
    return metrics, predictions, scores, golds, ids

path = 'D:/data/superglue_data/COPA/train.jsonl'

load_copa(path)
