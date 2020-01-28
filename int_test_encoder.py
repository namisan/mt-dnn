import os
import os.path as path
import filecmp
import subprocess
import glob
import torch

ENCODE_CMD = """python train.py --train_datasets cola --test_datasets cola \
--encode_mode \
--data_dir %s \
--init_checkpoint %s \
--encoder_type %s \
--output_dir %s """

def test_encoder(src_dir, checkpoint_path, encoder_type, target_dir, expected_dir):
    subprocess.call("rm -rf %s" % target_dir, shell=True)
    assert not os.access(target_dir, os.F_OK), "preprocessed target directory already exist"
    cmd = ENCODE_CMD % (src_dir, checkpoint_path, encoder_type, target_dir)
    print("execute cmd:")
    print(cmd)
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)

    encoding_0 = torch.load(os.path.join(target_dir, r"cola_encoding.pt"))
    encoding_1 = torch.load(os.path.join(expected_dir, r"cola_encoding.pt"))
    abs_diff = (encoding_0 - encoding_1).abs() 
    abs_avg = ((encoding_0 + encoding_1)/2).abs()
    assert (abs_diff/abs_avg).mean() < 1e-4, "relative diff: %s" % (abs_diff/abs_avg).mean()

src_dir = "int_test_data/glue/input/encoder/bert_uncased_lower"
checkpoint_path = "mt_dnn_models/bert_model_base_uncased.pt"
encoder_type = 1
target_dir = "int_test_data/glue/test_output"
expected_dir = "int_test_data/glue/expected/encoder/bert_uncased_lower"
test_encoder(src_dir, checkpoint_path, encoder_type, target_dir, expected_dir)

src_dir = "int_test_data/glue/input/encoder/roberta_cased_lower"
checkpoint_path = "mt_dnn_models/roberta.base"
encoder_type = 2
target_dir = "int_test_data/glue/test_output"
expected_dir = "int_test_data/glue/expected/encoder/roberta_cased_lower"
test_encoder(src_dir, checkpoint_path, encoder_type, target_dir, expected_dir)
