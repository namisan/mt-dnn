# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import shutil
import os
import subprocess
import filecmp
import os.path


def compare_files(dir1, dir2, common_files, text_mode=False):
    same_files = []
    diff_files = []
    for common_file in common_files:
        path0 = os.path.join(dir1, common_file)
        path1 = os.path.join(dir2, common_file)
        open_mode = "r" if text_mode else "rb"
        s0 = open(path0, open_mode).read()
        s1 = open(path1, open_mode).read()
        if s0 == s1:
            same_files.append(common_file)
        else:
            diff_files.append(common_file)
    return same_files, diff_files


def are_dir_trees_equal(dir1, dir2):
    """
    Compare two directories recursively. Files in each directory are
    assumed to be equal if their names and contents are equal.

    @param dir1: First directory path
    @param dir2: Second directory path

    @return: True if the directory trees are the same and
        there were no errors while accessing the directories or files,
        False otherwise.
   """

    dirs_cmp = filecmp.dircmp(dir1, dir2)
    
    if len(dirs_cmp.left_only)>0 or len(dirs_cmp.right_only)>0 or \
        len(dirs_cmp.funny_files)>0:
        return False
    _, diff_files = compare_files(dir1, dir2, dirs_cmp.common_files, text_mode=True)
    if len(diff_files) > 0:
        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = os.path.join(dir1, common_dir)
        new_dir2 = os.path.join(dir2, common_dir)
        if not are_dir_trees_equal(new_dir1, new_dir2):
            return False
    return True

def test_prepro():
    if os.access("./run_test", os.F_OK):
        shutil.rmtree("./run_test")
    os.mkdir("./run_test")
    shutil.copytree("./tests/sample_data/input", "./run_test/sample_data")

    result = subprocess.check_output("python experiments/glue/glue_prepro.py --root_dir run_test/sample_data", stderr=subprocess.STDOUT, shell=True)
    result = subprocess.check_output("python prepro_std.py --model bert-base-uncased --root_dir run_test/sample_data/canonical_data --task_def experiments/glue/glue_task_def.yml", stderr=subprocess.STDOUT, shell=True)
    assert are_dir_trees_equal("./run_test/sample_data/canonical_data/bert-base-uncased", "./tests/sample_data/output")

