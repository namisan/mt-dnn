import os
import os.path as path
import filecmp
import subprocess
import glob
import torch
import json
import numpy.ma as ma

ENCODE_CMD = """python train.py --train_datasets cola --test_datasets cola \
--encode_mode \
--data_dir %s \
--init_checkpoint %s \
--encoder_type %s \
--output_dir %s """


def test_encoder(src_dir, checkpoint_path, encoder_type, target_dir, expected_dir):
    subprocess.call("rm -rf %s" % target_dir, shell=True)
    assert not os.access(
        target_dir, os.F_OK
    ), "preprocessed target directory already exist"
    cmd = ENCODE_CMD % (src_dir, checkpoint_path, encoder_type, target_dir)
    print("execute cmd:")
    print(cmd)
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)

    # get masks from unit_test input file (cola_test.json)
    input_lengths = []
    with open(os.path.join(src_dir, "cola_test.json"), "r", encoding="utf-8") as fin:
        for line in fin:
            json_line = json.loads(line)
            assert (
                "token_id" in json_line
            ), "token_id is not in current test file, please check name of token ids in input json"
            input_lengths.append(len(json_line["token_id"]))

    encoding_0 = torch.load(os.path.join(target_dir, r"cola_encoding.pt"))
    encoding_1 = torch.load(os.path.join(expected_dir, r"cola_encoding.pt"))

    abs_diff = (encoding_0 - encoding_1).abs()
    abs_avg = ((encoding_0 + encoding_1) / 2).abs()

    tensor_mask = torch.LongTensor(
        len(input_lengths), max(input_lengths), encoding_0.size()[-1]
    ).fill_(1)
    for index, input_length in enumerate(input_lengths):
        tensor_mask[index, :input_length, :] = torch.LongTensor(
            input_length, encoding_0.size()[-1]
        ).fill_(0)

    relative_diff = abs_diff / abs_avg
    masked_array = ma.masked_array(relative_diff, mask=tensor_mask)

    assert masked_array.mean() < 1e-4, "relative diff: %s" % masked_array.mean()


src_dir = "int_test_data/glue/input/encoder/bert_uncased_lower"
checkpoint_path = "mt_dnn_models/bert_model_base_uncased.pt"
encoder_type = 1
target_dir = "int_test_data/glue/test_output"
expected_dir = "int_test_data/glue/expected/encoder/bert_uncased_lower"
test_encoder(src_dir, checkpoint_path, encoder_type, target_dir, expected_dir)

# test again using downloading
checkpoint_path = "bert-base-uncased"
test_encoder(src_dir, checkpoint_path, encoder_type, target_dir, expected_dir)

src_dir = "int_test_data/glue/input/encoder/roberta_cased_lower"
checkpoint_path = "roberta-base"
encoder_type = 2
target_dir = "int_test_data/glue/test_output"
expected_dir = "int_test_data/glue/expected/encoder/roberta_cased_lower"
test_encoder(src_dir, checkpoint_path, encoder_type, target_dir, expected_dir)
