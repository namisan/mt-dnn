import yaml
from data_utils.vocab import Vocabulary
from data_utils.task_def import TaskType, DataFormat, EncoderModelType
from data_utils.metrics import Metric

class TaskDefs:
    def __init__(self, task_def_path):
        self._task_def_dic = yaml.safe_load(open(task_def_path))
        global_map = {}
        n_class_map = {}
        data_type_map = {}
        task_type_map = {}
        metric_meta_map = {}
        enable_san_map = {}
        dropout_p_map = {}
        encoderType_map = {}
        uniq_encoderType = set()
        for task, task_def in self._task_def_dic.items():
            assert "_" not in task, "task name should not contain '_', current task name: %s" % task
            n_class_map[task] = task_def["n_class"]
            data_format = DataFormat[task_def["data_format"]]
            data_type_map[task] = data_format
            task_type_map[task] = TaskType[task_def["task_type"]]
            metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def["metric_meta"])
            enable_san_map[task] = task_def["enable_san"]
            uniq_encoderType.add(EncoderModelType[task_def["encoder_type"]])
            if "labels" in task_def:
                labels = task_def["labels"]
                label_mapper = Vocabulary(True)
                for label in labels:
                    label_mapper.add(label)
                global_map[task] = label_mapper
            if "dropout_p" in task_def:
                dropout_p_map[task] = task_def["dropout_p"]

        assert len(uniq_encoderType) == 1, 'The shared encoder has to be the same.'
        self.global_map = global_map
        self.n_class_map = n_class_map
        self.data_type_map = data_type_map
        self.task_type_map = task_type_map
        self.metric_meta_map = metric_meta_map
        self.enable_san_map = enable_san_map
        self.dropout_p_map = dropout_p_map
        self.encoderType = uniq_encoderType.pop()
