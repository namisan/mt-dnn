import os
import argparse
from sys import path

path.append(os.getcwd())
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
from experiments.ner.ner_utils import load_conll_chunk, load_conll_ner, load_conll_pos
from experiments.common_utils import dump_rows

logger = create_logger(
    __name__, to_disk=True, log_file="bert_ner_data_proc_512_cased.log"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing English NER dataset.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    data_dir = args.data_dir
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    train_path = os.path.join(data_dir, "train.txt")
    dev_path = os.path.join(data_dir, "valid.txt")
    test_path = os.path.join(data_dir, "test.txt")

    train_data = load_conll_ner(train_path)
    dev_data = load_conll_ner(dev_path)
    test_data = load_conll_ner(test_path)
    logger.info("Loaded {} NER train samples".format(len(train_data)))
    logger.info("Loaded {} NER dev samples".format(len(dev_data)))
    logger.info("Loaded {} NER test samples".format(len(test_data)))

    pos_train_data = load_conll_pos(train_path)
    pos_dev_data = load_conll_pos(dev_path)
    pos_test_data = load_conll_pos(test_path)
    logger.info("Loaded {} POS train samples".format(len(pos_train_data)))
    logger.info("Loaded {} POS dev samples".format(len(pos_train_data)))
    logger.info("Loaded {} POS test samples".format(len(pos_train_data)))

    chunk_train_data = load_conll_chunk(train_path)
    chunk_dev_data = load_conll_chunk(dev_path)
    chunk_test_data = load_conll_chunk(test_path)
    logger.info("Loaded {} POS train samples".format(len(chunk_train_data)))
    logger.info("Loaded {} POS dev samples".format(len(chunk_dev_data)))
    logger.info("Loaded {} POS test samples".format(len(chunk_test_data)))

    bert_root = args.output_dir
    if not os.path.isdir(bert_root):
        os.mkdir(bert_root)
    train_fout = os.path.join(bert_root, "ner_train.tsv")
    dev_fout = os.path.join(bert_root, "ner_dev.tsv")
    test_fout = os.path.join(bert_root, "ner_test.tsv")

    dump_rows(train_data, train_fout, DataFormat.Seqence)
    dump_rows(dev_data, dev_fout, DataFormat.Seqence)
    dump_rows(test_data, test_fout, DataFormat.Seqence)
    logger.info("done with NER")

    train_fout = os.path.join(bert_root, "pos_train.tsv")
    dev_fout = os.path.join(bert_root, "pos_dev.tsv")
    test_fout = os.path.join(bert_root, "pos_test.tsv")
    dump_rows(pos_train_data, train_fout, DataFormat.Seqence)
    dump_rows(pos_dev_data, dev_fout, DataFormat.Seqence)
    dump_rows(pos_test_data, test_fout, DataFormat.Seqence)
    logger.info("done with POS")

    train_fout = os.path.join(bert_root, "chunk_train.tsv")
    dev_fout = os.path.join(bert_root, "chunk_dev.tsv")
    test_fout = os.path.join(bert_root, "chunk_test.tsv")
    dump_rows(chunk_train_data, train_fout, DataFormat.Seqence)
    dump_rows(chunk_dev_data, dev_fout, DataFormat.Seqence)
    dump_rows(chunk_test_data, test_fout, DataFormat.Seqence)
    logger.info("done with chunk")


if __name__ == "__main__":
    args = parse_args()
    main(args)
