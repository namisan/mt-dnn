# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
from copy import deepcopy
import sys
import json
import torch
import random
import numpy as np
from shutil import copyfile
from data_utils.task_def import TaskType, DataFormat
from data_utils.task_def import EncoderModelType
import tasks
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler
from experiments.exp_def import TaskDef
from experiments.mlm.mlm_utils import truncate_seq_pair, load_loose_json
from experiments.mlm.mlm_utils import (
    create_instances_from_document,
    create_masked_lm_predictions,
)

UNK_ID = 100
BOS_ID = 101


def search_bin(bins, size):
    idx = len(bins) - 1
    for i, bin in enumerate(bins):
        if size <= bin:
            idx = i
            break
    return idx


def create_bins(bin_size, maxlen):
    return [min(i + bin_size, maxlen) for i in range(0, maxlen, bin_size)]


class DistMultiTaskBatchSampler(Sampler):
    def __init__(
        self,
        datasets,
        batch_size,
        mix_opt,
        extra_task_ratio,
        rank=0,
        world_size=1,
        drop_last=False,
    ):
        self.rank = rank
        self.world_size = world_size
        self._datasets = datasets
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        self.drop_last = drop_last
        train_data_list = []
        for dataset in datasets:
            train_data_list.append(
                self._get_shuffled_index_batches(len(dataset), batch_size)
            )
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [
            list(range(i, min(i + batch_size, dataset_len)))
            for i in range(0, dataset_len, batch_size)
        ]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(
            self._train_data_list, self._mix_opt, self._extra_task_ratio
        )
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].get_task_id()
            batch = next(all_iters[local_task_idx])
            batch = [(task_id, sample_id) for sample_id in batch]
            if len(batch) % self.world_size != 0:
                if self.drop_last:
                    break
                else:
                    batch.extend(
                        [
                            batch[0]
                            for _ in range(
                                self.world_size - len(batch) % self.world_size
                            )
                        ]
                    )
            chunk_size = len(batch) // self.world_size
            yield batch[self.rank * chunk_size : (self.rank + 1) * chunk_size]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(
                min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices))
            )
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices


class DistSingleTaskBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, rank=0, world_size=1, drop_last=False):
        self.rank = rank
        self.world_size = world_size
        self._dataset = dataset
        self.drop_last = drop_last
        self._data = self._get_index_batches(len(dataset), batch_size)

    @staticmethod
    def _get_index_batches(dataset_len, batch_size):
        index_batches = [
            list(range(i, min(i + batch_size, dataset_len)))
            for i in range(0, dataset_len, batch_size)
        ]
        return index_batches

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        indices = iter(self._data)
        for batch in indices:
            task_id = self._dataset.get_task_id()
            batch = [(task_id, sample_id) for sample_id in batch]
            yield batch


class MultiTaskBatchSampler(BatchSampler):
    def __init__(
        self,
        datasets,
        batch_size,
        mix_opt,
        extra_task_ratio,
        bin_size=64,
        bin_on=False,
        bin_grow_ratio=0.5,
    ):
        self._datasets = datasets
        self._batch_size = batch_size
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        self.bin_size = bin_size
        self.bin_on = bin_on
        self.bin_grow_ratio = bin_grow_ratio
        train_data_list = []
        for dataset in datasets:
            if bin_on:
                train_data_list.append(
                    self._get_shuffled_index_batches_bin(
                        dataset,
                        batch_size,
                        bin_size=bin_size,
                        bin_grow_ratio=bin_grow_ratio,
                    )
                )
            else:
                train_data_list.append(
                    self._get_shuffled_index_batches(len(dataset), batch_size)
                )
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [
            list(range(i, min(i + batch_size, dataset_len)))
            for i in range(0, dataset_len, batch_size)
        ]
        random.shuffle(index_batches)
        return index_batches

    @staticmethod
    def _get_shuffled_index_batches_bin(dataset, batch_size, bin_size, bin_grow_ratio):
        maxlen = dataset.maxlen
        bins = create_bins(bin_size, maxlen)
        data = [[] for i in range(0, len(bins))]

        for idx, sample in enumerate(dataset):
            bin_idx = search_bin(bins, len(sample["sample"]["token_id"]))
            data[bin_idx].append(idx)
        index_batches = []

        for idx, sub_data in enumerate(data):
            if len(sub_data) < 1:
                continue
            batch_size = 1 if batch_size < 1 else batch_size
            sub_dataset_len = len(sub_data)
            sub_batches = [
                list(range(i, min(i + batch_size, sub_dataset_len)))
                for i in range(0, sub_dataset_len, batch_size)
            ]
            index_batches.extend(sub_batches)
            batch_size = int(batch_size * bin_grow_ratio)
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(
            self._train_data_list, self._mix_opt, self._extra_task_ratio
        )
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].get_task_id()
            batch = next(all_iters[local_task_idx])
            yield [(task_id, sample_id) for sample_id in batch]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(
                min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices))
            )
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices


class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for dataset in datasets:
            task_id = dataset.get_task_id()
            assert task_id not in task_id_2_data_set_dic, (
                "Duplicate task_id %s" % task_id
            )
            task_id_2_data_set_dic[task_id] = dataset

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return self._task_id_2_data_set_dic[task_id][sample_id]


class DistTaskDataset(Dataset):
    def __init__(self, dataset, task_id):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        _, sample_id = idx
        return self._dataset[sample_id]

    def get_task_id(self):
        return self._dataset.get_task_id()


class SingleTaskDataset(Dataset):
    def __init__(
        self,
        path,
        is_train=True,
        maxlen=512,
        factor=1.0,
        task_id=0,
        task_def: TaskDef = None,
        bert_model="bert-base-uncased",
        do_lower_case=True,
        masked_lm_prob=0.15,
        seed=13,
        short_seq_prob=0.1,
        max_seq_length=512,
        max_predictions_per_seq=80,
        printable=True,
    ):
        data, tokenizer = self.load(
            path,
            is_train,
            maxlen,
            factor,
            task_def,
            bert_model,
            do_lower_case,
            printable=printable,
        )
        self._data = data
        self._tokenizer = tokenizer
        self._task_id = task_id
        self._task_def = task_def
        # below is for MLM
        if self._task_def.task_type is TaskType.MaskLM:
            assert tokenizer is not None
        # init vocab words
        self._vocab_words = (
            None if tokenizer is None else list(self._tokenizer.vocab.keys())
        )
        self._masked_lm_prob = masked_lm_prob
        self._seed = seed
        self._short_seq_prob = short_seq_prob
        self._max_seq_length = max_seq_length
        self._max_predictions_per_seq = max_predictions_per_seq
        self._rng = random.Random(seed)
        self.maxlen = maxlen

    def get_task_id(self):
        return self._task_id

    @staticmethod
    def load(
        path,
        is_train=True,
        maxlen=512,
        factor=1.0,
        task_def=None,
        bert_model="bert-base-uncased",
        do_lower_case=True,
        printable=True,
    ):
        task_type = task_def.task_type
        assert task_type is not None

        if task_type == TaskType.MaskLM:

            def load_mlm_data(path):
                from transformers import AutoTokenizer
                tokenizer =  AutoTokenizer.from_pretrained(bert_model, cache_dir=".cache")
                vocab_words = list(tokenizer.vocab.keys())
                data = load_loose_json(path)
                docs = []
                for doc in data:
                    paras = doc["text"].split("\n\n")
                    paras = [para.strip() for para in paras if len(para.strip()) > 0]
                    tokens = [tokenizer.tokenize(para) for para in paras]
                    docs.append(tokens)
                return docs, tokenizer

            return load_mlm_data(path)

        with open(path, "r", encoding="utf-8") as reader:
            data = []
            cnt = 0
            for line in reader:
                sample = json.loads(line)
                sample["factor"] = factor
                cnt += 1
                if is_train:
                    task_obj = tasks.get_task_obj(task_def)
                    if task_obj is not None and not task_obj.input_is_valid_sample(
                        sample, maxlen
                    ):
                        continue
                    if (task_type == TaskType.Ranking) and (
                        len(sample["token_id"][0]) > maxlen
                        or len(sample["token_id"][1]) > maxlen
                    ):
                        continue
                    if (task_type != TaskType.Ranking) and (
                        len(sample["token_id"]) > maxlen
                    ):
                        continue
                data.append(sample)
            if printable:
                print("Loaded {} samples out of {}".format(len(data), cnt))
        return data, None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self._task_def.task_type == TaskType.MaskLM:
            # create a MLM instance
            instances = create_instances_from_document(
                self._data,
                idx,
                self._max_seq_length,
                self._short_seq_prob,
                self._masked_lm_prob,
                self._max_predictions_per_seq,
                self._vocab_words,
                self._rng,
            )
            instance_ids = list(range(0, len(instances)))
            choice = np.random.choice(instance_ids, 1)[0]
            instance = instances[choice]
            labels = self._tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
            position = instance.masked_lm_positions
            labels = [lab if idx in position else -1 for idx, lab in enumerate(labels)]
            sample = {
                "token_id": self._tokenizer.convert_tokens_to_ids(instance.tokens),
                "type_id": instance.segment_ids,
                "nsp_lab": 1 if instance.is_random_next else 0,
                "position": instance.masked_lm_positions,
                "label": labels,
                "uid": idx,
            }
            return {
                "task": {"task_id": self._task_id, "task_def": self._task_def},
                "sample": sample,
            }
        else:
            return {
                "task": {"task_id": self._task_id, "task_def": self._task_def},
                "sample": self._data[idx],
            }


class Collater:
    def __init__(
        self,
        is_train=True,
        dropout_w=0.005,
        soft_label=False,
        encoder_type=EncoderModelType.BERT,
        max_seq_len=512,
        do_padding=False,
    ):
        self.is_train = is_train
        self.dropout_w = dropout_w
        self.soft_label_on = soft_label
        self.encoder_type = encoder_type
        self.pairwise_size = 1
        self.max_seq_len = max_seq_len
        self.do_padding = do_padding

    def __random_select__(self, arr):
        if self.dropout_w > 0:
            return [UNK_ID if random.uniform(0, 1) < self.dropout_w else e for e in arr]
        else:
            return arr

    @staticmethod
    def patch_data(device, batch_info, batch_data):
        if str(device) != "cpu":
            for i, part in enumerate(batch_data):
                if part is None:
                    continue
                if isinstance(part, torch.Tensor):
                    batch_data[i] = part.pin_memory().to(device)
                elif isinstance(part, tuple):
                    batch_data[i] = tuple(
                        sub_part.pin_memory().to(device) for sub_part in part
                    )
                elif isinstance(part, list):
                    batch_data[i] = [
                        sub_part.pin_memory().to(device) for sub_part in part
                    ]
                else:
                    raise TypeError("unknown batch data type at %s: %s" % (i, part))

                if "soft_label" in batch_info:
                    batch_info["soft_label"] = (
                        batch_info["soft_label"].pin_memory().to(device)
                    )
        return batch_info, batch_data

    def rebatch(self, batch):
        newbatch = []
        sizes = []
        for sample in batch:
            size = len(sample["token_id"])
            sizes.append(size)
            self.pairwise_size = size
            assert size == len(sample["type_id"])
            for idx in range(0, size):
                token_id = sample["token_id"][idx]
                type_id = sample["type_id"][idx]
                attention_mask = sample["attention_mask"][idx]
                uid = sample["ruid"][idx] if "ruid" in sample else sample["uid"]
                olab = sample["olabel"][idx]
                new_sample = deepcopy(sample)
                new_sample["uid"] = uid
                new_sample["token_id"] = token_id
                new_sample["type_id"] = type_id
                new_sample["attention_mask"] = attention_mask
                new_sample["true_label"] = olab
                newbatch.append(new_sample)
        return newbatch, sizes

    def __if_pair__(self, data_type):
        return data_type in [
            DataFormat.PremiseAndOneHypothesis,
            DataFormat.PremiseAndMultiHypothesis,
        ]

    def collate_fn(self, batch):
        task_id = batch[0]["task"]["task_id"]
        task_def = batch[0]["task"]["task_def"]
        new_batch = []
        for sample in batch:
            assert sample["task"]["task_id"] == task_id
            assert sample["task"]["task_def"] == task_def
            new_batch.append(sample["sample"])
        task_type = task_def.task_type
        data_type = task_def.data_type
        batch = new_batch

        if task_type == TaskType.Ranking or task_type == TaskType.ClozeChoice:
            batch, chunk_sizes = self.rebatch(batch)

        # prepare model input
        batch_info, batch_data = self._prepare_model_input(batch, data_type)
        batch_info["task_id"] = task_id  # used for select correct decoding head
        batch_info["input_len"] = len(batch_data)  # used to select model inputs
        # select different loss function and other difference in training and testing
        # DataLoader will convert any unknown type objects to dict,
        # the conversion logic also convert Enum to repr(Enum), which is a string and undesirable
        # If we convert object to dict in advance, DataLoader will do nothing
        batch_info["task_def"] = task_def.__dict__
        batch_info["pairwise_size"] = self.pairwise_size  # need for ranking task
        # add label
        labels = [sample["label"] if "label" in sample else None for sample in batch]
        task_obj = tasks.get_task_obj(task_def)
        if self.is_train:
            # in training model, label is used by Pytorch, so would be tensor
            if task_obj is not None:
                batch_data.append(task_obj.train_prepare_label(labels))
                batch_info["label"] = len(batch_data) - 1
            elif task_type == TaskType.Ranking or task_type == TaskType.ClozeChoice:
                batch_data.append(torch.LongTensor(labels))
                batch_info["label"] = len(batch_data) - 1
            elif task_type == TaskType.Span:
                # support multi positions
                start, end = [], []
                for sample in batch:
                    if type(sample["start_position"]) is list and type(
                        sample["end_position"]
                    ):
                        idx = random.choice(range(0, len(sample["start_position"])))
                        start.append(sample["start_position"][idx])
                        end.append(sample["end_position"][idx])
                    else:
                        start.append(sample["start_position"])
                        end.append(sample["end_position"])
                batch_data.append((torch.LongTensor(start), torch.LongTensor(end)))
                # unify to one type of label
                batch_info["label"] = len(batch_data) - 1
            elif task_type == TaskType.SpanYN:
                # start = [sample['start_position'] for sample in batch]
                # end = [sample['end_position'] for sample in batch]
                start, end = [], []
                for sample in batch:
                    if type(sample["start_position"]) is list and type(
                        sample["end_position"]
                    ):
                        idx = random.choice(range(0, len(sample["start_position"])))
                        start.append(sample["start_position"][idx])
                        end.append(sample["end_position"][idx])
                    else:
                        start.append(sample["start_position"])
                        end.append(sample["end_position"])
                # start, end, yes/no
                batch_data.append(
                    (
                        torch.LongTensor(start),
                        torch.LongTensor(end),
                        torch.LongTensor(labels),
                    )
                )
                # unify to one type of label
                batch_info["label"] = len(batch_data) - 1
            elif task_type == TaskType.SeqenceLabeling:
                batch_size = self._get_batch_size(batch)
                tok_len = self._get_max_len(batch, key="token_id")
                tlab = torch.LongTensor(batch_size, tok_len).fill_(-1)
                for i, label in enumerate(labels):
                    ll = len(label)
                    tlab[i, :ll] = torch.LongTensor(label)
                batch_data.append(tlab)
                batch_info["label"] = len(batch_data) - 1
            elif task_type == TaskType.MaskLM:
                batch_size = self._get_batch_size(batch)
                tok_len = self._get_max_len(batch, key="token_id")
                tlab = torch.LongTensor(batch_size, tok_len).fill_(-1)
                for i, label in enumerate(labels):
                    ll = len(label)
                    tlab[i, :ll] = torch.LongTensor(label)
                labels = torch.LongTensor([sample["nsp_lab"] for sample in batch])
                batch_data.append((tlab, labels))
                batch_info["label"] = len(batch_data) - 1
            elif task_type == TaskType.SeqenceGeneration:
                batch_size = self._get_batch_size(batch)
                y_idxs = torch.LongTensor([sample["label"][:-1] for sample in batch])
                label = torch.LongTensor([sample["label"][1:] for sample in batch])
                label.masked_fill_(label == 0, -1)
                batch_data.append(y_idxs)
                batch_info["y_token_id"] = len(batch_data) - 1
                batch_data.append(label)
                batch_info["label"] = len(batch_data) - 1

            # soft label generated by ensemble models for knowledge distillation
            if self.soft_label_on and "softlabel" in batch[0]:
                sortlabels = [sample["softlabel"] for sample in batch]
                sortlabels = task_obj.train_prepare_soft_labels(sortlabels)
                batch_info["soft_label"] = sortlabels
        else:
            # in test model, label would be used for evaluation
            if task_obj is not None:
                task_obj.test_prepare_label(batch_info, labels)
            else:
                batch_info["label"] = labels
                if task_type == TaskType.Ranking:
                    batch_info["true_label"] = [
                        sample["true_label"] for sample in batch
                    ]
                if task_type == TaskType.ClozeChoice:
                    batch_info["answer"] = [
                        sample["answer"] for sample in batch
                    ]
                    batch_info["choice"] = [
                        sample["choice"] for sample in batch
                    ]
                    batch_info["pairwise_size"] = chunk_sizes
                if task_type == TaskType.Span or task_type == TaskType.SpanYN:
                    batch_info["offset_mapping"] = [
                        sample["offset_mapping"] for sample in batch
                    ]
                    batch_info["token_is_max_context"] = [
                        sample.get("token_is_max_context", None) for sample in batch
                    ]
                    batch_info["context"] = [sample["context"] for sample in batch]
                    batch_info["answer"] = [sample["answer"] for sample in batch]
                    batch_info["label"] = [
                        sample["label"] if "label" in sample else None
                        for sample in batch
                    ]
                if task_type == TaskType.SeqenceGeneration:
                    batch_info["answer"] = [sample["answer"] for sample in batch]

        batch_info["uids"] = [sample["uid"] for sample in batch]  # used in scoring
        return batch_info, batch_data

    def _get_max_len(self, batch, key="token_id"):
        tok_len = max(len(x[key]) for x in batch)
        tok_len = self.max_seq_len if self.do_padding else tok_len
        return tok_len

    def _get_batch_size(self, batch):
        return len(batch)

    def _prepare_model_input(self, batch, data_type):
        batch_size = self._get_batch_size(batch)
        tok_len = self._get_max_len(batch, key="token_id")
        if self.encoder_type == EncoderModelType.ROBERTA:
            token_ids = torch.LongTensor(batch_size, tok_len).fill_(1)
            type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            masks = torch.LongTensor(batch_size, tok_len).fill_(0)
        else:
            token_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            masks = torch.LongTensor(batch_size, tok_len).fill_(0)
        if self.__if_pair__(data_type):
            hypothesis_masks = torch.BoolTensor(batch_size, tok_len).fill_(1)
            premise_masks = torch.BoolTensor(batch_size, tok_len).fill_(1)
        for i, sample in enumerate(batch):
            select_len = min(len(sample["token_id"]), tok_len)
            tok = sample["token_id"]
            if self.is_train:
                tok = self.__random_select__(tok)
            token_ids[i, :select_len] = torch.LongTensor(tok[:select_len])
            type_ids[i, :select_len] = torch.LongTensor(sample["type_id"][:select_len])
            masks[i, :select_len] = torch.LongTensor([1] * select_len)
            if self.__if_pair__(data_type):
                plen = len(sample["type_id"]) - sum(sample["type_id"])
                premise_masks[i, :plen] = torch.LongTensor([0] * plen)
                for j in range(plen, select_len):
                    hypothesis_masks[i, j] = 0
        if self.__if_pair__(data_type):
            batch_info = {
                "token_id": 0,
                "segment_id": 1,
                "mask": 2,
                "premise_mask": 3,
                "hypothesis_mask": 4,
            }
            batch_data = [token_ids, type_ids, masks, premise_masks, hypothesis_masks]
        else:
            batch_info = {"token_id": 0, "segment_id": 1, "mask": 2}
            batch_data = [token_ids, type_ids, masks]
        return batch_info, batch_data
