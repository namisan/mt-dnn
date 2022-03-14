import os
import argparse
import random
from sys import path

path.append(os.getcwd())
from experiments.common_utils import dump_rows
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
from experiments.superglue.superglue_utils import *

logger = create_logger(__name__, to_disk=True, log_file="superglue_prepro.log")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing SuperGLUE dataset."
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--root_dir", type=str, default="data")
    args = parser.parse_args()
    return args


def main(args):
    root = args.root_dir
    assert os.path.exists(root)

    ######################################
    # SuperGLUE tasks
    ######################################

    cb_train_path = os.path.join(root, "CB/train.jsonl")
    cb_dev_path = os.path.join(root, "CB/val.jsonl")
    cb_test_path = os.path.join(root, "CB/test.jsonl")

    boolq_train_path = os.path.join(root, "BoolQ/train.jsonl")
    boolq_dev_path = os.path.join(root, "BoolQ/val.jsonl")
    boolq_test_path = os.path.join(root, "BoolQ/test.jsonl")

    copa_train_path = os.path.join(root, "COPA/train.jsonl")
    copa_dev_path = os.path.join(root, "COPA/val.jsonl")
    copa_test_path = os.path.join(root, "COPA/test.jsonl")

    record_train_path = os.path.join(root, "ReCoRD/train.jsonl")
    record_dev_path = os.path.join(root, "ReCoRD/val.jsonl")
    record_test_path = os.path.join(root, "ReCoRD/test.jsonl")

    wic_train_path = os.path.join(root, "WiC/train.jsonl")
    wic_dev_path = os.path.join(root, "WiC/val.jsonl")
    wic_test_path = os.path.join(root, "WiC/test.jsonl")

    multirc_train_path = os.path.join(root, "MultiRC/train.jsonl")
    multirc_dev_path = os.path.join(root, "MultiRC/val.jsonl")
    multirc_test_path = os.path.join(root, "MultiRC/test.jsonl")

    ######################################
    # Loading DATA
    ######################################

    cb_train_data = load_cb(cb_train_path)
    cb_dev_data = load_cb(cb_dev_path)
    cb_test_data = load_cb(cb_test_path)
    logger.info("Loaded {} CB train samples".format(len(cb_train_data)))
    logger.info("Loaded {} CB dev samples".format(len(cb_dev_data)))
    logger.info("Loaded {} CB test samples".format(len(cb_test_data)))

    boolq_train_data = load_boolq(boolq_train_path)
    boolq_dev_data = load_boolq(boolq_dev_path)
    boolq_test_data = load_boolq(boolq_test_path)
    logger.info("Loaded {} BoolQ train samples".format(len(boolq_train_data)))
    logger.info("Loaded {} BoolQ dev samples".format(len(boolq_dev_data)))
    logger.info("Loaded {} BoolQ test samples".format(len(boolq_test_data)))

    copa_train_data = load_copa_mtdnn(copa_train_path)
    copa_dev_data = load_copa_mtdnn(copa_dev_path)
    copa_test_data = load_copa_mtdnn(copa_test_path)
    logger.info("Loaded {} COPA train samples".format(len(copa_train_data)))
    logger.info("Loaded {} COPA dev samples".format(len(copa_dev_data)))
    logger.info("Loaded {} COPA test samples".format(len(copa_test_data)))

    record_train_data = load_record_mtdnn(record_train_path)
    record_dev_data = load_record_mtdnn(record_dev_path)
    record_test_data = load_record_mtdnn(record_test_path)
    logger.info("Loaded {} Record train samples".format(len(record_train_data)))
    logger.info("Loaded {} Record dev samples".format(len(record_dev_data)))
    logger.info("Loaded {} Record test samples".format(len(record_test_data)))

    wic_train_data = load_wic_mtdnn(wic_train_path)
    wic_dev_data = load_wic_mtdnn(wic_dev_path)
    wic_test_data = load_wic_mtdnn(wic_test_path)
    logger.info("Loaded {} WiC train samples".format(len(wic_train_data)))
    logger.info("Loaded {} WiC dev samples".format(len(wic_dev_data)))
    logger.info("Loaded {} WiC test samples".format(len(wic_test_data)))

    multirc_train_data = load_multirc_mtdnn(multirc_train_path)
    multirc_dev_data = load_multirc_mtdnn(multirc_dev_path)
    multirc_test_data = load_multirc_mtdnn(multirc_test_path)
    logger.info("Loaded {} MultiRC train samples".format(len(multirc_train_data)))
    logger.info("Loaded {} MultiRC dev samples".format(len(multirc_dev_data)))
    logger.info("Loaded {} MultiRC test samples".format(len(multirc_test_data)))

    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)

    cb_train_fout = os.path.join(canonical_data_root, "cb_train.tsv")
    cb_dev_fout = os.path.join(canonical_data_root, "cb_dev.tsv")
    cb_test_fout = os.path.join(canonical_data_root, "cb_test.tsv")
    dump_rows(cb_train_data, cb_train_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(cb_dev_data, cb_dev_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(cb_test_data, cb_test_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info("done with CB")

    boolq_train_fout = os.path.join(canonical_data_root, "boolq_train.tsv")
    boolq_dev_fout = os.path.join(canonical_data_root, "boolq_dev.tsv")
    boolq_test_fout = os.path.join(canonical_data_root, "boolq_test.tsv")
    dump_rows(boolq_train_data, boolq_train_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(boolq_dev_data, boolq_dev_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(boolq_test_data, boolq_test_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info("done with boolq")

    copa_train_fout = os.path.join(canonical_data_root, "copa_train.tsv")
    copa_dev_fout = os.path.join(canonical_data_root, "copa_dev.tsv")
    copa_test_fout = os.path.join(canonical_data_root, "copa_test.tsv")
    dump_rows(copa_train_data, copa_train_fout, DataFormat.PremiseAndMultiHypothesis)
    dump_rows(copa_dev_data, copa_dev_fout, DataFormat.PremiseAndMultiHypothesis)
    dump_rows(copa_test_data, copa_test_fout, DataFormat.PremiseAndMultiHypothesis)
    logger.info("done with record")

    record_train_fout = os.path.join(canonical_data_root, "record_train.tsv")
    record_dev_fout = os.path.join(canonical_data_root, "record_dev.tsv")
    record_test_fout = os.path.join(canonical_data_root, "record_test.tsv")
    dump_rows(record_train_data, record_train_fout, DataFormat.ClozeChoice)
    dump_rows(record_dev_data, record_dev_fout, DataFormat.ClozeChoice)
    dump_rows(record_test_data, record_test_fout, DataFormat.ClozeChoice)
    logger.info("done with record")

    wic_train_fout = os.path.join(canonical_data_root, "wic_train.tsv")
    wic_dev_fout = os.path.join(canonical_data_root, "wic_dev.tsv")
    wic_test_fout = os.path.join(canonical_data_root, "wic_test.tsv")
    dump_rows(wic_train_data, wic_train_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(wic_dev_data, wic_dev_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(wic_test_data, wic_test_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info("done with WiC")

    multirc_train_fout = os.path.join(canonical_data_root, "multirc_train.tsv")
    multirc_dev_fout = os.path.join(canonical_data_root, "multirc_dev.tsv")
    multirc_test_fout = os.path.join(canonical_data_root, "multirc_test.tsv")
    dump_rows(multirc_train_data, multirc_train_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(multirc_dev_data, multirc_dev_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(multirc_test_data, multirc_test_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info("done with MultiRC")

if __name__ == "__main__":
    args = parse_args()
    main(args)
