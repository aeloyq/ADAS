HOME_PATH = 'models'
MODEL_NAME = 'face'
GRAPH_PATH = 'inference_quantize_full_/tfmobile_model.pb'
LOG_DIR = 'inference_quantize_full_'
import os
import tensorflow as tf

with tf.Graph().as_default():
  graph_path = os.path.join(HOME_PATH, MODEL_NAME, GRAPH_PATH)
  with tf.gfile.GFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def)
  graph = tf.get_default_graph()
  graph_summary = tf.summary.FileWriter(os.path.join(HOME_PATH, MODEL_NAME, LOG_DIR), graph=graph)
  graph_summary.close()
