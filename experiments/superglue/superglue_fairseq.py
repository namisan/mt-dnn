import os
import argparse
import random
from sys import path

path.append(os.getcwd())
from experiments.superglue.superglue_utils import save, TASKS, LOAD_FUNCS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data directory')
    parser.add_argument('--task', type=str, default='CB')
    args = parser.parse_args()
    return args

def main(args):
    task = args.task.lower()
    assert task in TASKS, "don't support {}".format(task)
    files = TASKS[task]
    laod_function = LOAD_FUNCS[task]
    for file in files:
        data = laod_function(os.path.join(args.data_dir, file))
        assert len(data) > 0
        filename, extension = os.path.splitext(file)
        fin = os.path.join(args.data_dir, file)
        prefix = os.path.join(args.data_dir, "{}".format(filename))

        columns = len(data[0])
        labels = [str(sample["label"]) for sample in data]
        input0 = [sample["premise"] for sample in data]
        input1 = [sample["hypothesis"] for sample in data] if columns > 3 else None
        input2 = [sample["hypothesis_extra"] for sample in data] if "hypothesis_extra" in data[0] else None
        has_answer = "answer" in data[0]
        answers = [sample["answer"] for sample in data] if has_answer  else None

        flabel = "{}.label".format(prefix)
        save(labels, flabel)
        finput0 = "{}.raw.input0".format(prefix)
        save(input0, finput0)
        if input1:
            finput1 = "{}.raw.input1".format(prefix)
            save(input1, finput1)
        if input2:
            finput2 = "{}.raw.input2".format(prefix)
            save(input2, finput2)
        if answers:
            fanswers = "{}.answer".format(prefix)
            save(answers, fanswers)            
        uids = [str(sample["uid"]) for sample in data]
        fuid = "{}.id".format(prefix)
        save(uids, fuid)


if __name__ == '__main__':
    args = parse_args()
    main(args)
