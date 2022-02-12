# because we don't specify exact software version in Dockerfile,
# the train loss could be different when you rebuild the Dockerfile
# so we hide this test. But it still useful for developer when you constantly working on exact same environment
# (Docker, hardware)
import os
import shutil
import subprocess

import re
TRAIN_LOSS_RE = re.compile("train loss\[[\d\.]+\]")

def assert_file_equal(output_file, expected_file):
    output = open(output_file).read()
    expected = open(expected_file).read()
    assert output == expected, "file diff: %s != %s" % (output_file, expected_file)

def compare_output(output_dir, expected_dir):
    config = open(os.path.join(output_dir, "config.json")).read()
    expected_config = open(os.path.join(expected_dir, "config.json")).read()
    assert config == expected_config, "Config diff"

    train_loss = TRAIN_LOSS_RE.findall(open(os.path.join(output_dir, "log.txt")).read())
    expected_train_loss = TRAIN_LOSS_RE.findall(open(os.path.join(expected_dir, "log.txt")).read())
    assert train_loss == expected_train_loss, "Train loss diff:\n\ttrain_loss is %s\n\texpected_train_loss is %s\n" % (
        train_loss, expected_train_loss
    )

    for file_name in ("mnli_matched_dev_scores_0.json", "mnli_matched_test_scores_0.json",
                      "mnli_mismatched_dev_scores_0.json", "mnli_mismatched_test_scores_0.json"):
        assert_file_equal(os.path.join(output_dir, file_name), os.path.join(expected_dir, file_name))

def test_train():
    OUTPUT_DIR = r"run_test/checkpoint"
    EXPECTED_DIR = r"tests/sample_data/checkpoint"

    if os.access("./run_test", os.F_OK):
        shutil.rmtree("./run_test")
    os.mkdir("./run_test")
    shutil.copytree("./sample_data", "./run_test/sample_data")
    os.mkdir("./run_test/checkpoint")
    subprocess.check_output("python train.py --epoch 1 --log_per_updates 1 --data_dir run_test/sample_data/output --output_dir %(OUTPUT_DIR)s 2>&1 > %(OUTPUT_DIR)s/log.txt"
                                     % {"OUTPUT_DIR": OUTPUT_DIR}, stderr=subprocess.STDOUT, shell=True)

    compare_output(OUTPUT_DIR, EXPECTED_DIR)
