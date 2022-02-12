import os
import argparse
import random
from sys import path

path.append(os.getcwd())
from experiments.common_utils import dump_rows
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
from experiments.glue.glue_utils import *

logger = create_logger(__name__, to_disk=True, log_file="xnli_prepro.log")


def load_xnli(file, header=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            if blocks[1] == "-":
                continue
            lab = blocks[1]
            if lab is None:
                import pdb

                pdb.set_trace()
            sample = {
                "uid": blocks[9],
                "premise": blocks[6],
                "hypothesis": blocks[7],
                "label": lab,
                "lang": blocks[0],
            }
            rows.append(sample)
            cnt += 1
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing XNLI dataset.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--root_dir", type=str, default="data")
    args = parser.parse_args()
    return args


def main(args):
    root = args.root_dir
    assert os.path.exists(root)

    ######################################
    # XNLI/SciTail Tasks
    ######################################

    xnli_dev_path = os.path.join(root, "XNLI/xnli.dev.tsv")
    xnli_test_path = os.path.join(root, "XNLI/xnli.test.tsv")

    ######################################
    # Loading DATA
    ######################################

    xnli_dev_data = load_xnli(xnli_dev_path)
    xnli_test_data = load_xnli(xnli_test_path)
    logger.info("Loaded {} XNLI train samples".format(len(xnli_dev_data)))
    logger.info("Loaded {} XNLI test samples".format(len(xnli_test_data)))

    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)

    # BUILD XNLI
    xnli_dev_fout = os.path.join(canonical_data_root, "xnli_dev.tsv")
    xnli_test_fout = os.path.join(canonical_data_root, "xnli_test.tsv")
    dump_rows(xnli_dev_data, xnli_dev_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(xnli_test_data, xnli_test_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info("done with XNLI")


if __name__ == "__main__":
    args = parse_args()
    main(args)
