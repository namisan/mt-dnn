import yaml
from data_utils.vocab import Vocabulary

class TaskDefs:
    def __init__(self, task_def_path):
        self._task_def_dic = yaml.safe_load(open(task_def_path))
        global_map = {}
        n_class_map = {}
        for task, task_def in self._task_def_dic.items():
            n_class_map[task] = task_def["n_class"]
            if "labels" in task_def:
                labels = task_def["labels"]
                label_mapper = Vocabulary(True)
                for label in labels:
                    label_mapper.add(label)
                global_map[task] = label_mapper

        self.global_map = global_map
        self.n_class_map = n_class_map
