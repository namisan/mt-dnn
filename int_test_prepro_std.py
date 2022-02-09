import os
import os.path as path
import filecmp
import subprocess
import glob


def assert_dir_equal(dir0, dir1):
    for file_path0 in glob.glob(path.join(dir0, "*")):
        file_name = path.split(file_path0)[1]
        file_path1 = path.join(dir1, file_name)
        assert filecmp.cmp(
            file_path0, file_path1, shallow=False
        ), '%s diff in two directories "%s" and "%s"' % (file_name, dir0, dir1)


def test_prepro_std(cmd, src_dir, task_def_path, target_dir, expected_dir):
    subprocess.call("rm -rf %s" % target_dir, shell=True)
    assert not os.access(
        target_dir, os.F_OK
    ), "preprocessed target directory already exist"
    subprocess.call(
        cmd % (src_dir, task_def_path), shell=True, stdout=subprocess.DEVNULL
    )
    assert_dir_equal(target_dir, expected_dir)


BERT_CMD = "python prepro_std.py --model bert-base-uncased --root_dir %s --task_def %s --do_lower_case "
ROBERTA_CMD = "python prepro_std.py --model roberta-base --root_dir %s --task_def %s"
SRC_DIR = "int_test_data/glue/input/prepro_std"
TASK_DEF_PATH = "int_test_data/glue/input/prepro_std/glue_task_def.yml"

test_prepro_std(
    BERT_CMD,
    SRC_DIR,
    TASK_DEF_PATH,
    "int_test_data/glue/input/prepro_std/bert_base_uncased_lower",
    "int_test_data/glue/expected/prepro_std/bert_base_uncased_lower",
)

test_prepro_std(
    ROBERTA_CMD,
    SRC_DIR,
    TASK_DEF_PATH,
    "int_test_data/glue/input/prepro_std/roberta_base_cased",
    "int_test_data/glue/expected/prepro_std/roberta_base_cased",
)
