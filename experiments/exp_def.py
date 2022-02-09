# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import yaml
from data_utils.vocab import Vocabulary
from data_utils.task_def import TaskType, DataFormat, EncoderModelType
from data_utils.metrics import Metric
from mt_dnn.loss import LossCriterion


class TaskDef(dict):
    def __init__(
        self,
        label_vocab,
        n_class,
        data_type,
        task_type,
        metric_meta,
        split_names,
        enable_san,
        dropout_p,
        loss,
        kd_loss,
        adv_loss,
    ):
        """
        :param label_vocab: map string label to numbers.
            only valid for Classification task or ranking task.
            For ranking task, better label should have large number
        """
        super().__init__(
            **{k: repr(v) for k, v in locals().items()}
        )  # ensure the class is JSON serializable
        self.label_vocab = label_vocab
        self.n_class = n_class
        self.data_type = data_type
        self.task_type = task_type
        self.metric_meta = metric_meta
        self.split_names = split_names
        self.enable_san = enable_san
        self.dropout_p = dropout_p
        self.loss = loss
        self.kd_loss = kd_loss
        self.adv_loss = adv_loss

    @classmethod
    def from_dict(cls, dict_rep):
        return cls(**dict_rep)


class TaskDefs:
    def __init__(self, task_def_path):
        self._task_def_dic = yaml.safe_load(open(task_def_path))
        global_map = {}
        n_class_map = {}
        data_type_map = {}
        task_type_map = {}
        metric_meta_map = {}
        split_names_map = {}
        enable_san_map = {}
        dropout_p_map = {}
        loss_map = {}
        kd_loss_map = {}
        adv_loss_map = {}

        for task, task_def in self._task_def_dic.items():
            assert "_" not in task, (
                "task name should not contain '_', current task name: %s" % task
            )
            n_class_map[task] = task_def["n_class"]
            data_format = DataFormat[task_def["data_format"]]
            data_type_map[task] = data_format
            task_type_map[task] = TaskType[task_def["task_type"]]
            metric_meta_map[task] = tuple(
                Metric[metric_name] for metric_name in task_def["metric_meta"]
            )
            split_names_map[task] = task_def.get(
                "split_names", ["train", "dev", "test"]
            )
            enable_san_map[task] = task_def["enable_san"]
            if "labels" in task_def:
                labels = task_def["labels"]
                label_mapper = Vocabulary(True)
                for label in labels:
                    label_mapper.add(label)
                global_map[task] = label_mapper
            if "dropout_p" in task_def:
                dropout_p_map[task] = task_def["dropout_p"]
            # loss map
            if "loss" in task_def:
                t_loss = task_def["loss"]
                loss_crt = LossCriterion[t_loss]
                loss_map[task] = loss_crt
            else:
                loss_map[task] = None

            if "kd_loss" in task_def:
                t_loss = task_def["kd_loss"]
                loss_crt = LossCriterion[t_loss]
                kd_loss_map[task] = loss_crt
            else:
                kd_loss_map[task] = None

            if "adv_loss" in task_def:
                t_loss = task_def["adv_loss"]
                loss_crt = LossCriterion[t_loss]
                adv_loss_map[task] = loss_crt
            else:
                adv_loss_map[task] = None

        self._global_map = global_map
        self._n_class_map = n_class_map
        self._data_type_map = data_type_map
        self._task_type_map = task_type_map
        self._metric_meta_map = metric_meta_map
        self._split_names_map = split_names_map
        self._enable_san_map = enable_san_map
        self._dropout_p_map = dropout_p_map
        self._loss_map = loss_map
        self._kd_loss_map = kd_loss_map
        self._adv_loss_map = adv_loss_map

        self._task_def_dic = {}

    def get_task_names(self):
        return list(self._task_type_map.keys())

    def get_task_def(self, task_name):
        if task_name not in self._task_def_dic:
            assert task_name in self._task_type_map
            self._task_def_dic[task_name] = TaskDef(
                self._global_map.get(task_name, None),
                self._n_class_map[task_name],
                self._data_type_map[task_name],
                self._task_type_map[task_name],
                self._metric_meta_map[task_name],
                self._split_names_map[task_name],
                self._enable_san_map[task_name],
                self._dropout_p_map.get(task_name, None),
                self._loss_map[task_name],
                self._kd_loss_map[task_name],
                self._adv_loss_map[task_name],
            )
        return self._task_def_dic[task_name]
