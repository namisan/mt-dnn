import os
import argparse
import random
from sys import path

path.append(os.getcwd())
from experiments.common_utils import dump_rows
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger

logger = create_logger(__name__, to_disk=True, log_file="domain_prepro.log")

def load_scitail(file):
    """Loading data of scitail"""
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            blocks = line.strip().split("\t")
            assert len(blocks) > 2
            if blocks[0] == "-":
                continue
            sample = {
                "uid": str(cnt),
                "premise": blocks[0],
                "hypothesis": blocks[1],
                "label": blocks[2],
            }
            rows.append(sample)
            cnt += 1
    return rows


def load_snli(file, header=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            assert len(blocks) > 10
            if blocks[-1] == "-":
                continue
            lab = blocks[-1]
            if lab is None:
                import pdb

                pdb.set_trace()
            sample = {
                "uid": blocks[0],
                "premise": blocks[7],
                "hypothesis": blocks[8],
                "label": lab,
            }
            rows.append(sample)
            cnt += 1
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing GLUE/SNLI/SciTail dataset."
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument(
        "--old_glue",
        action="store_true",
        help="whether it is old GLUE, refer official GLUE webpage for details",
    )
    args = parser.parse_args()
    return args


def main(args):
    is_old_glue = args.old_glue
    root = args.root_dir
    assert os.path.exists(root)

    ######################################
    # SNLI/SciTail Tasks
    ######################################
    scitail_train_path = os.path.join(root, "SciTail/tsv_format/scitail_1.0_train.tsv")
    scitail_dev_path = os.path.join(root, "SciTail/tsv_format/scitail_1.0_dev.tsv")
    scitail_test_path = os.path.join(root, "SciTail/tsv_format/scitail_1.0_test.tsv")

    snli_train_path = os.path.join(root, "SNLI/train.tsv")
    snli_dev_path = os.path.join(root, "SNLI/dev.tsv")
    snli_test_path = os.path.join(root, "SNLI/test.tsv")

    ######################################
    # Loading DATA
    ######################################
    scitail_train_data = load_scitail(scitail_train_path)
    scitail_dev_data = load_scitail(scitail_dev_path)
    scitail_test_data = load_scitail(scitail_test_path)
    logger.info("Loaded {} SciTail train samples".format(len(scitail_train_data)))
    logger.info("Loaded {} SciTail dev samples".format(len(scitail_dev_data)))
    logger.info("Loaded {} SciTail test samples".format(len(scitail_test_data)))

    snli_train_data = load_snli(snli_train_path)
    snli_dev_data = load_snli(snli_dev_path)
    snli_test_data = load_snli(snli_test_path)
    logger.info("Loaded {} SNLI train samples".format(len(snli_train_data)))
    logger.info("Loaded {} SNLI dev samples".format(len(snli_dev_data)))
    logger.info("Loaded {} SNLI test samples".format(len(snli_test_data)))

    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)

    # BUILD SciTail
    scitail_train_fout = os.path.join(canonical_data_root, "scitail_train.tsv")
    scitail_dev_fout = os.path.join(canonical_data_root, "scitail_dev.tsv")
    scitail_test_fout = os.path.join(canonical_data_root, "scitail_test.tsv")
    dump_rows(
        scitail_train_data, scitail_train_fout, DataFormat.PremiseAndOneHypothesis
    )
    dump_rows(scitail_dev_data, scitail_dev_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(scitail_test_data, scitail_test_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info("done with scitail")

    # BUILD SNLI
    snli_train_fout = os.path.join(canonical_data_root, "snli_train.tsv")
    snli_dev_fout = os.path.join(canonical_data_root, "snli_dev.tsv")
    snli_test_fout = os.path.join(canonical_data_root, "snli_test.tsv")
    dump_rows(snli_train_data, snli_train_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(snli_dev_data, snli_dev_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(snli_test_data, snli_test_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info("done with snli")

if __name__ == "__main__":
    args = parse_args()
    main(args)
