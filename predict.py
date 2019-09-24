import argparse
import json
import os
import torch

from data_utils.task_def import TaskType
from experiments.exp_def import TaskDefs, EncoderModelType
#from experiments.glue.glue_utils import eval_model

from mt_dnn.batcher import BatchGen
from mt_dnn.model import MTDNNModel
from data_utils.metrics import calc_metrics

def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def eval_model(model, data, metric_meta, use_cuda=True, with_label=True):
    data.reset()
    if use_cuda:
        model.cuda()
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for batch_meta, batch_data in data:
        score, pred, gold = model.predict(batch_meta, batch_data)
        predictions.extend(pred)
        golds.extend(gold)
        scores.extend(score)
        ids.extend(batch_meta['uids'])
    if with_label:
        metrics = calc_metrics(metric_meta, golds, predictions, scores, ids)
    return metrics, predictions, scores, golds, ids


parser = argparse.ArgumentParser()
parser.add_argument("--task_def", type=str, default="experiments/glue/glue_task_def.yml")
parser.add_argument("--task", type=str)
parser.add_argument("--task_id", type=int, help="the id of this task when training")

parser.add_argument("--prep_input", type=str)
parser.add_argument("--with_label", action="store_true")
parser.add_argument("--score", type=str, help="score output path")

parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--batch_size_eval', type=int, default=8)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')

parser.add_argument("--checkpoint", default='mt_dnn_models/bert_model_base_uncased.pt', type=str)

args = parser.parse_args()

# load task info
task_defs = TaskDefs(args.task_def)
assert args.task in task_defs.task_type_map
assert args.task in task_defs.data_type_map
assert args.task in task_defs.metric_meta_map
data_type = task_defs.data_type_map[args.task]
task_type = task_defs.task_type_map[args.task]
metric_meta = task_defs.metric_meta_map[args.task]

# load model
checkpoint_path = args.checkpoint
assert os.path.exists(checkpoint_path)
if args.cuda:
    state_dict = torch.load(checkpoint_path)
else:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
config = state_dict['config']
config["cuda"] = args.cuda
model = MTDNNModel(config, state_dict=state_dict)
model.load(checkpoint_path)
encoder_type = config.get('encoder_type', EncoderModelType.BERT)
# load data
test_data = BatchGen(BatchGen.load(args.prep_input, False, task_type=task_type, maxlen=args.max_seq_len),
                     batch_size=args.batch_size_eval,
                     gpu=args.cuda, is_train=False,
                     task_id=args.task_id,
                     maxlen=args.max_seq_len,
                     data_type=data_type,
                     task_type=task_type,
                     encoder_type=encoder_type)


with torch.no_grad():
    test_metrics, test_predictions, scores, golds, test_ids = eval_model(model, test_data,
                                                                         metric_meta=metric_meta,
                                                                         use_cuda=args.cuda, with_label=args.with_label)

    results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': scores}
    dump(args.score, results)
    if args.with_label:
        print(test_metrics)
