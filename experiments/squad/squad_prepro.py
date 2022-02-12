# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import yaml
import os
import numpy as np
import argparse
import json
import sys
from tqdm.auto import tqdm
from data_utils.task_def import TaskType, DataFormat
from data_utils.log_wrapper import create_logger
from experiments.exp_def import TaskDefs, EncoderModelType
from transformers import AutoTokenizer


DEBUG_MODE = False
MAX_SEQ_LEN = 384
DOC_STRIDE = 128
MAX_QUERY_LEN = 64

logger = create_logger(
    __name__, to_disk=True, log_file="mt_dnn_clues_data_proc_{}.log".format(MAX_SEQ_LEN)
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_squad(path):
    input_data = load_json(path)
    print("version: {}".format(input_data["version"]))
    version = input_data["version"]
    input_data = input_data["data"]
    examples = []
    for entry in tqdm(input_data):
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question = qa["question"]
                answer = qa["answers"]
                is_impossible = qa.get("is_impossible", False)
                example = {
                    "id": qas_id,
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "is_impossible": is_impossible,
                }
                examples.append(example)
    return examples


def flat_squad(path, is_training=True):
    def qa_sample(uid, context, question, answers, is_impossible, answer_start=None):
        if len(answers) > 0:
            answer_text = [answer["text"].strip() for answer in answers]
            answer_start = [answer["answer_start"] for answer in answers]
        else:
            answer_text = []
            answer_start = []
        answers = {"text": answer_text, "answer_start": answer_start}

        example = {
            "id": uid,
            "context": context.strip(),  # fix whitespace issues
            "question": question.strip(),  # fix whitespace issues
            "answer": answers,
            "is_impossible": is_impossible,
        }
        return example

    input_data = load_squad(path)
    examples = []
    for entry in tqdm(input_data):
        context = entry["context"]
        if "qas" in entry:
            for qa_entry in entry["qas"]:
                question = qa_entry["question"]
                if isinstance(question, list):
                    assert len(question) == 1
                    question = question[0]
                uid = qa_entry["id"]
                # remove dumplicated answers
                answers = list(set(qa_entry.get("answer", [])))
                if type(context) is dict:
                    entities = context["entities"]
                    context_text = context["text"]
                    assert type(answers) is list
                    temp_answers = []
                    ent_strs = []
                    for ent in entities:
                        ent_strs.append(context_text[ent["start"] : ent["end"] + 1])
                    for answer in answers:
                        positions = []
                        for ent in entities:
                            if context_text[ent["start"] : ent["end"] + 1] == answer:
                                positions.append(ent["start"])
                        temp_answers.append({"text": answer, "answer_start": positions})
                    answers = temp_answers
                is_impossible = qa_entry.get("is_impossible", None)
                example = qa_sample(uid, context_text, question, answers, is_impossible)
                examples.append(example)
        else:
            question = entry["question"]
            uid = entry["id"]
            answers = entry.get("answer", [])
            is_impossible = entry.get("is_impossible", None)
            example = qa_sample(uid, context, question, answers, is_impossible)
            examples.append(example)
    return examples


def search_index(
    input_ids,
    sequence_ids,
    offsets,
    cls_index,
    start_char,
    end_char,
    pad_on_right=False,
):
    start_position, end_position = cls_index, cls_index
    # Start token index of the current span in the text.
    token_start_index = 0
    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
        token_start_index += 1
    # End token index of the current span in the text.
    token_end_index = len(input_ids) - 1
    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
        token_end_index -= 1

    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
    if not (
        offsets[token_start_index][0] <= start_char
        and offsets[token_end_index][1] >= end_char
    ):
        start_position = cls_index
        end_position = cls_index
    else:
        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
        # Note: we could go after the last offset if the answer is the last word (edge case).
        while (
            token_start_index < len(offsets)
            and offsets[token_start_index][0] <= start_char
        ):
            token_start_index += 1
        start_position = token_start_index - 1
        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_position = token_end_index + 1
    return start_position, end_position


def prepare_train_feature(
    tokenizer,
    samples,
    output_path,
    data_type=DataFormat.CLUE_CLASSIFICATION,
    max_seq_length=384,
    doc_stride=128,
    pad_on_right=True,
    pad_to_max_length=True,
    label_mapper=None,
):
    if not tokenizer.cls_token:
        # cls_tok_id = tokenizer.eos_token_id
        cls_tok_id = tokenizer.pad_token_id
        prefix_pad = True
    else:
        cls_tok_id = tokenizer.cls_token_id
        prefix_pad = False

    if not tokenizer.sep_token:
        sep_tok_id = tokenizer.eos_token_id
    else:
        sep_tok_id = tokenizer.sep_token_id

    with open(output_path, "w", encoding="utf-8") as writer:
        for sample in samples:
            context = sample["context"]
            question = sample["question"]

            if pad_on_right and prefix_pad:
                question = tokenizer.pad_token + question

            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = tokenizer(
                question if pad_on_right else context,
                context if pad_on_right else question,
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if pad_to_max_length else False,
                verbose=False,
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            # Let's label those examples!
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []
            tokenized_examples["id"] = []
            tokenized_examples["label"] = []
            for i, offsets in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                cls_index = input_ids.index(cls_tok_id)

                # One example can give several spans, this is the index of the example containing this span of text.
                # sample_index = sample_mapping[i]
                answers = sample["answer"]
                tokenized_examples["id"].append(sample["id"])
                label = None
                if sample["is_impossible"] is not None:
                    label = 1 if sample["is_impossible"] else 0
                tokenized_examples["label"].append(label)
                # If no answers are given, set the cls_index as answer.
                if len(answers) == 0 or len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    if type(start_char) is list:
                        start_position, end_position = [], []
                        for sc in start_char:
                            end_char = sc + len(answers["text"][0])
                            sp, ep = search_index(
                                input_ids,
                                sequence_ids,
                                offsets,
                                cls_index,
                                sc,
                                end_char,
                                pad_on_right=pad_on_right,
                            )
                            start_position.append(sp)
                            end_position.append(ep)
                        tokenized_examples["start_positions"].append(start_position)
                        tokenized_examples["end_positions"].append(end_position)
                    else:
                        end_char = start_char + len(answers["text"][0])
                        start_position, end_position = search_index(
                            input_ids,
                            sequence_ids,
                            offsets,
                            cls_index,
                            start_char,
                            end_char,
                            pad_on_right=pad_on_right,
                        )
                        tokenized_examples["start_positions"].append(start_position)
                        tokenized_examples["end_positions"].append(end_position)
                    tokenized_examples["label"].append(0)

            for i in range(0, len(tokenized_examples["input_ids"])):
                sample = {
                    "uid": tokenized_examples["id"][i],
                    "token_id": tokenized_examples["input_ids"][i],
                    "mask": tokenized_examples["attention_mask"][i],
                    "type_id": tokenized_examples["token_type_ids"][i]
                    if "token_type_ids" in tokenized_examples
                    else len(tokenized_examples["input_ids"][i]) * [0],
                    "start_position": tokenized_examples["start_positions"][i],
                    "end_position": tokenized_examples["end_positions"][i],
                    "label": tokenized_examples["label"][i],
                }
                writer.write("{}\n".format(json.dumps(sample)))


# Validation preprocessing
def prepare_validation_features(
    tokenizer,
    samples,
    output_path,
    max_seq_length=384,
    doc_stride=128,
    pad_on_right=True,
    pad_to_max_length=True,
    label_mapper=None,
):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    if not tokenizer.cls_token:
        # cls_tok_id = tokenizer.eos_token_id
        cls_tok_id = tokenizer.pad_token_id
        prefix_pad = True
    else:
        cls_tok_id = tokenizer.cls_token_id
        prefix_pad = False

    if not tokenizer.sep_token:
        sep_tok_id = tokenizer.eos_token_id
    else:
        sep_tok_id = tokenizer.sep_token_id

    with open(output_path, "w", encoding="utf-8") as writer:
        for sample in samples:
            context = sample["context"]
            question = sample["question"]
            answer = sample.get("answer")

            if pad_on_right and prefix_pad:
                question = tokenizer.pad_token + question

            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = tokenizer(
                question if pad_on_right else context,
                context if pad_on_right else question,
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if pad_to_max_length else False,
                verbose=False,
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []
            tokenized_examples["id"] = []
            tokenized_examples["label"] = []
            tokenized_examples["null_ans_index"] = []
            label = None
            offset_mapping = tokenized_examples.pop("offset_mapping")

            for i in range(len(tokenized_examples["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(cls_tok_id)
                sep_index = input_ids.index(sep_tok_id)

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                tokenized_examples["id"].append(sample["id"])
                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.

                tokenized_examples["id"].append(sample["id"])
                if sample["is_impossible"] is not None:
                    label = 1 if sample["is_impossible"] else 0
                    answer["is_impossible"] = sample["is_impossible"]
                tokenized_examples["label"].append(label)
                tokenized_examples["null_ans_index"].append(cls_index)

            # tokenized_examples["offset_mapping"] = offset_mapping
            for i in range(0, len(tokenized_examples["input_ids"])):
                sample = {
                    "uid": tokenized_examples["id"][i],
                    "token_id": tokenized_examples["input_ids"][i],
                    "mask": tokenized_examples["attention_mask"][i],
                    "type_id": tokenized_examples["token_type_ids"][i]
                    if "token_type_ids" in tokenized_examples
                    else len(tokenized_examples["input_ids"][i]) * [0],
                    "offset_mapping": offset_mapping[i],
                    "null_ans_index": tokenized_examples["null_ans_index"][i],
                    "context": context,
                    "answer": answer,
                    "label": tokenized_examples["label"][i],
                }
                writer.write("{}\n".format(json.dumps(sample)))


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing MRC dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="support all BERT, XLNET and ROBERTA family supported by HuggingFace Transformers",
    )
    parser.add_argument("--root_dir", type=str, default="data/canonical_data")
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument(
        "--task_def", type=str, default="experiments/squad/squad_task_def.yml"
    )
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--doc_stride", type=int, default=DOC_STRIDE)
    args = parser.parse_args()
    return args


def main(args):
    # hyper param
    root = args.root_dir
    assert os.path.exists(root)
    suffix = args.model.split("/")[-1]
    literal_model_type = suffix.split("-")[0].upper()

    encoder_model = EncoderModelType[literal_model_type]
    literal_model_type = literal_model_type.lower()
    mt_dnn_suffix = literal_model_type
    if "base" in args.model:
        mt_dnn_suffix += "_base"
    elif "large" in args.model:
        mt_dnn_suffix += "_large"

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        use_fast=True,
        from_slow=True,
        revision=args.model_revision,
    )
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if "uncased" in args.model:
        mt_dnn_suffix = "{}_uncased".format(mt_dnn_suffix)
    else:
        mt_dnn_suffix = "{}_cased".format(mt_dnn_suffix)

    mt_dnn_root = os.path.join(root, mt_dnn_suffix)
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    task_defs = TaskDefs(args.task_def)

    for task in task_defs.get_task_names():
        task_def = task_defs.get_task_def(task)
        logger.info("Task %s" % task)
        for split_name in task_def.split_names:
            print(root)
            file_path = os.path.join(root, "%s_%s.json" % (task, split_name))
            print(file_path)

            if not os.path.exists(file_path):
                logger.warning("File %s doesnot exit")
                sys.exit(1)
            logger.warning("processing %s" % file_path)
            is_training = True
            if not "train" in split_name:
                is_training = False

            rows = flat_squad(file_path, is_training)
            dump_path = os.path.join(mt_dnn_root, "%s_%s.json" % (task, split_name))
            logger.info(dump_path)
            if is_training:
                prepare_train_feature(
                    tokenizer,
                    rows,
                    dump_path,
                    pad_on_right=pad_on_right,
                    label_mapper=task_def.label_vocab,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                )
            else:
                prepare_validation_features(
                    tokenizer,
                    rows,
                    dump_path,
                    pad_on_right=pad_on_right,
                    label_mapper=task_def.label_vocab,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                )


if __name__ == "__main__":
    args = parse_args()
    main(args)
