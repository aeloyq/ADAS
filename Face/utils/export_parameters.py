HOME_PATH = 'models'
MODEL_NAME = 'face'
CKPT_PATH = 'train_quantize'
CKPT_NAME = 'model.ckpt'

import os
import tensorflow as tf
import numpy as np
from collections import OrderedDict

CKPT_FILE=os.path.join(HOME_PATH,MODEL_NAME,CKPT_PATH,CKPT_NAME)
ckpt_reader = tf.train.NewCheckpointReader(CKPT_FILE)
parameters_shape_dict=ckpt_reader.get_variable_to_shape_map()
parameters_names = sorted([name for name in parameters_shape_dict.keys()])
parameters=OrderedDict()
for name in parameters_names:
  parameters[name]=ckpt_reader.get_tensor(name)
pass
p=OrderedDict()
for n in parameters_names:
  if n.startswith('BoxPredictor'):
    if len(parameters_shape_dict[n])==0:
      p[n]=ckpt_reader.get_tensor(n)
