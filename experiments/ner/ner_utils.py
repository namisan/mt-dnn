import os
from sys import path

path.append(os.getcwd())
from data_utils.task_def import DataFormat


def load_conll_ner(file, is_train=True):
    rows = []
    cnt = 0
    sentence = []
    label = []
    with open(file, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
                if len(sentence) > 0:
                    sample = {"uid": cnt, "premise": sentence, "label": label}
                    rows.append(sample)
                    sentence = []
                    label = []
                    cnt += 1
                continue
            splits = line.split(" ")
            sentence.append(splits[0])
            label.append(splits[-1])
        if len(sentence) > 0:
            sample = {"uid": cnt, "premise": sentence, "label": label}
    return rows


def load_conll_pos(file, is_train=True):
    rows = []
    cnt = 0
    sentence = []
    label = []
    with open(file, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
                if len(sentence) > 0:
                    sample = {"uid": cnt, "premise": sentence, "label": label}
                    rows.append(sample)
                    sentence = []
                    label = []
                    cnt += 1
                continue
            splits = line.split(" ")
            sentence.append(splits[0])
            label.append(splits[1])
        if len(sentence) > 0:
            sample = {"uid": cnt, "premise": sentence, "label": label}
    return rows


def load_conll_chunk(file, is_train=True):
    rows = []
    cnt = 0
    sentence = []
    label = []
    with open(file, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
                if len(sentence) > 0:
                    sample = {"uid": cnt, "premise": sentence, "label": label}
                    rows.append(sample)
                    sentence = []
                    label = []
                    cnt += 1
                continue
            splits = line.split(" ")
            sentence.append(splits[0])
            label.append(splits[2])
        if len(sentence) > 0:
            sample = {"uid": cnt, "premise": sentence, "label": label}
    return rows
