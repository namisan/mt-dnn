# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
"""Extract feature vectors.
"""
import os
import argparse
import torch
import json
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from data_utils.log_wrapper import create_logger
from data_utils.utils import set_environment
from mt_dnn.batcher import Collater, SingleTaskDataset
from mt_dnn.model import MTDNNModel
from data_utils.task_def import DataFormat, EncoderModelType

logger = create_logger(__name__, to_disk=True, log_file="mt_dnn_feature_extractor.log")


def load_data(file):
    rows = []
    cnt = 0
    is_single_sentence = False
    with open(file, encoding="utf8") as f:
        for line in f:
            blocks = line.strip().split("|||")
            if len(blocks) == 2:
                sample = {
                    "uid": str(cnt),
                    "premise": blocks[0],
                    "hypothesis": blocks[1],
                    "label": 0,
                }
            else:
                is_single_sentence = True
                sample = {"uid": str(cnt), "premise": blocks[0], "label": 0}
            rows.append(sample)
            cnt += 1
    return rows, is_single_sentence


def build_data(data, max_seq_len, is_train=True, tokenizer=None):
    """Build data of sentence pair tasks"""
    rows = []
    for idx, sample in enumerate(data):
        ids = sample["uid"]
        label = sample["label"]
        inputs = tokenizer(
            sample["premise"],
            sample["hypothesis"],
            add_special_tokens=True,
            max_length=max_seq_len,
            truncation=True,
            padding=False,
        )
        input_ids = inputs["input_ids"]
        token_type_ids = (
            inputs["token_type_ids"] if "token_type_ids" in inputs else [0] * len(input_ids)
        )

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = inputs["attention_mask"]
        feature = {
                "uid": ids,
                "label": label,
                "token_id": input_ids,
                "type_id": token_type_ids,
                #"tokens": ["[CLS]"] + sample["premise"] + ["[SEP]"] + sample["hypothesis"] + ["[SEP]"],
            }
        rows.append(feature)
    return rows


def build_data_single(data, max_seq_len, tokenizer=None):
    """Build data of single sentence tasks"""
    rows = []
    for idx, sample in enumerate(data):
        ids = sample["uid"]
        label = sample["label"]
        inputs = tokenizer(
            sample["premise"],
            None,
            add_special_tokens=True,
            max_length=max_seq_len,
            truncation=True,
            padding=False,
        )
        input_ids = inputs["input_ids"]
        token_type_ids = (
            inputs["token_type_ids"] if "token_type_ids" in inputs else [0] * len(input_ids)
        )

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = inputs["attention_mask"]

        features = {
            "uid": ids,
            "label": label,
            "token_id": input_ids,
            "type_id": token_type_ids,
        }
        rows.append(features)
    return rows


def model_config(parser):
    parser.add_argument("--update_bert_opt", default=0, type=int)
    parser.add_argument("--multi_gpu_on", action="store_true")
    parser.add_argument(
        "--mem_cum_type", type=str, default="simple", help="bilinear/simple/defualt"
    )
    parser.add_argument("--answer_num_turn", type=int, default=5)
    parser.add_argument("--answer_mem_drop_p", type=float, default=0.1)
    parser.add_argument("--answer_att_hidden_size", type=int, default=128)
    parser.add_argument(
        "--answer_att_type",
        type=str,
        default="bilinear",
        help="bilinear/simple/defualt",
    )
    parser.add_argument(
        "--answer_rnn_type", type=str, default="gru", help="rnn/gru/lstm"
    )
    parser.add_argument(
        "--answer_sum_att_type",
        type=str,
        default="bilinear",
        help="bilinear/simple/defualt",
    )
    parser.add_argument("--answer_merge_opt", type=int, default=1)
    parser.add_argument("--answer_mem_type", type=int, default=1)
    parser.add_argument("--answer_dropout_p", type=float, default=0.1)
    parser.add_argument("--answer_weight_norm_on", action="store_true")
    parser.add_argument("--dump_state_on", action="store_true")
    parser.add_argument("--answer_opt", type=int, default=0, help="0,1")
    parser.add_argument("--label_size", type=str, default="3")
    parser.add_argument("--mtl_opt", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0)
    parser.add_argument("--mix_opt", type=int, default=0)
    parser.add_argument("--init_ratio", type=float, default=1)
    return parser


def train_config(parser):
    parser.add_argument(
        "--cuda",
        type=bool,
        default=torch.cuda.is_available(),
        help="whether to use GPU acceleration.",
    )
    parser.add_argument(
        "--optimizer",
        default="adamax",
        help="supported optimizer: adamax, sgd, adadelta, adam",
    )
    parser.add_argument("--grad_clipping", type=float, default=0)
    parser.add_argument("--global_grad_clipping", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--warmup_schedule", type=str, default="warmup_linear")
    parser.add_argument("--vb_dropout", action="store_false")
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--dropout_w", type=float, default=0.000)
    parser.add_argument("--bert_dropout_p", type=float, default=0.1)
    parser.add_argument("--ema_opt", type=int, default=0)
    parser.add_argument("--ema_gamma", type=float, default=0.995)
    # scheduler
    parser.add_argument(
        "--have_lr_scheduler", dest="have_lr_scheduler", action="store_false"
    )
    parser.add_argument("--multi_step_lr", type=str, default="10,20,30")
    parser.add_argument("--freeze_layers", type=int, default=-1)
    parser.add_argument("--embedding_opt", type=int, default=0)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--bert_l2norm", type=float, default=0.0)
    parser.add_argument("--scheduler_type", type=str, default="ms", help="ms/rop/exp")
    parser.add_argument("--output_dir", default="checkpoint")
    parser.add_argument(
        "--seed",
        type=int,
        default=2018,
        help="random seed for data shuffling, embedding init, etc.",
    )
    parser.add_argument("--encoder_type", type=int, default=EncoderModelType.BERT)
    # fp 16
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    return parser


def set_config(parser):
    parser.add_argument("--finput", default=None, type=str, required=True)
    parser.add_argument("--foutput", default=None, type=str, required=True)
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert model: bert-base-uncased",
    )
    parser.add_argument(
        "--checkpoint", default=None, type=str, required=True, help="model parameters"
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument("--layers", default="10,11", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int, help="")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--transformer_cache", default=".cache", type=str)


def process_data(args):

    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_model, cache_dir=args.transformer_cache
    )

    path = args.finput
    data, is_single_sentence = load_data(path)
    if is_single_sentence:
        tokened_data = build_data_single(
            data, max_seq_len=args.max_seq_length, tokenizer=tokenizer
        )
    else:
        tokened_data = build_data(
            data, max_seq_len=args.max_seq_length, tokenizer=tokenizer
        )
    return tokened_data, is_single_sentence


def dump_data(data, path):
    with open(path, "w", encoding="utf-8") as writer:
        for sample in data:
            writer.write("{}\n".format(json.dumps(sample)))


def main():
    parser = argparse.ArgumentParser()
    model_config(parser)
    set_config(parser)
    train_config(parser)
    args = parser.parse_args()
    encoder_type = args.encoder_type
    layer_indexes = [int(x) for x in args.layers.split(",")]
    set_environment(args.seed)
    # process data
    data, is_single_sentence = process_data(args)
    data_type = (
        DataFormat.PremiseOnly
        if is_single_sentence
        else DataFormat.PremiseAndOneHypothesis
    )
    fout_temp = "{}.tmp".format(args.finput)
    dump_data(data, fout_temp)
    collater = Collater(is_train=False, encoder_type=encoder_type)
    dataset = SingleTaskDataset(
        fout_temp, False, maxlen=args.max_seq_length,
    )
    batcher = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collater.collate_fn,
        pin_memory=args.cuda,
    )
    opt = vars(args)
    # load model
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint)
        config = state_dict["config"]
        config["dump_feature"] = True
        opt.update(config)
    else:
        logger.error("#" * 20)
        logger.error(
            "Could not find the init model!\n The parameters will be initialized randomly!"
        )
        logger.error("#" * 20)
        return
    num_all_batches = len(batcher)
    model = MTDNNModel(opt, state_dict=state_dict, num_train_step=num_all_batches)
    if args.cuda:
        model.cuda()

    features_dict = {}
    for batch_meta, batch_data in batcher:
        batch_meta, batch_data = Collater.patch_data(args.cuda, batch_meta, batch_data)
        all_encoder_layers, _ = model.extract(batch_meta, batch_data)
        embeddings = [
            all_encoder_layers[idx].detach().cpu().numpy() for idx in layer_indexes
        ]

        uids = batch_meta["uids"]
        masks = batch_data[batch_meta["mask"]].detach().cpu().numpy().tolist()
        for idx, uid in enumerate(uids):
            slen = sum(masks[idx])
            features = {}
            for yidx, layer in enumerate(layer_indexes):
                features[layer] = str(embeddings[yidx][idx][:slen].tolist())
            features_dict[uid] = features

    # save features
    with open(args.foutput, "w", encoding="utf-8") as writer:
        for sample in data:
            uid = sample["uid"]
            feature = features_dict[uid]
            feature["uid"] = uid
            writer.write("{}\n".format(json.dumps(feature)))


if __name__ == "__main__":
    main()
