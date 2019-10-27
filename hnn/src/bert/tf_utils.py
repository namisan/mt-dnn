import tensorflow as tf
import torch
import os

def truncated_normal(shape, mean=0, stdv=1.0, seed=None):
  verb = tf.logging.get_verbosity()
  tf.logging.set_verbosity('ERROR')
  os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
  os.environ["TF_CPP_MIN_VLOG_LEVEL"]="3"
  config=tf.ConfigProto(device_count={'GPU':0})
  seed=torch.randint(0,1000000, size=[], dtype=torch.int, device=torch.device('cpu')).item() if seed is None else seed
  value=tf.truncated_normal(shape, mean, stdv, seed=seed)
  with tf.Session(config=config) as sess:
    result = sess.run(value)
    tf.logging.set_verbosity(verb)
    return torch.as_tensor(result)

def truncated_normal_init(tensor, mean=0, stdv=1.0, seed=None):
  tensor.copy_(truncated_normal(tensor.size(), mean, stdv, seed))

