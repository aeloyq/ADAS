# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import tqdm
import shutil
import PIL.Image
import numpy as np
import tensorflow as tf

from lxml import etree

sys.path.append('..')
sys.path.append('../slim')
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('input', '', 'Root directory to raw adas face dataset.')
flags.DEFINE_string('output', 'data/face/', 'Path to output TFRecord')
flags.DEFINE_string('sets', 'train,valid', 'Convert training set, validation set or merged set.')
FLAGS = flags.FLAGS

SETS = ['train', 'valid', 'trainvalid', 'test', 'dev']


def dict_to_tf_example(example_name, annotation, image, label_map_dict):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    annotation: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    image: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """

  width = int(annotation['size']['width'])
  height = int(annotation['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  difficult_obj = []
  if 'object' in annotation:
    for obj in annotation['object']:
      # obj_name = obj['name'].encode('utf8')
      difficult = False  # name in [b'side_face',b'smoke',b'close_eyes']
      difficult_obj.append(int(difficult))
      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(
      annotation['filename'].encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(
      annotation['filename'].encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(image),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    'image/object/difficult': dataset_util.int64_list_feature(difficult_obj)
  }))
  return example


def main(_):
  try:
    sets_strings = FLAGS.sets
    sets_strings_list = sets_strings.split(',')
    sets = []
    for set_string in sets_strings_list:
      sets.append(set_string.strip())
  except:
    raise SyntaxError(
      "Wrong Format Of Atribute --Set {}.\nIt Should Be Somrthing Like train Or train,valid ".format(FLAGS.sets))

  input_dir = FLAGS.input
  output_dir = FLAGS.output
  images_dir = os.path.join(input_dir, 'images')
  annotations_dir = os.path.join(input_dir, 'annotations')
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  shutil.copy(os.path.join(input_dir, 'label_map.pbtxt'), os.path.join(output_dir, 'label_map.pbtxt'))
  shutil.copy(os.path.join(input_dir, 'summary.txt'), os.path.join(output_dir, 'summary.txt'))
  label_map_dict = label_map_util.get_label_map_dict(os.path.join(output_dir, 'label_map.pbtxt'))
  print('Generating ADAS FACE Dataset Within Tfrecord Format {}.'.format(str(sets)))
  for set in sets:
    print(set + ":")
    writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, set + ".record"))
    examples_metadata_path = os.path.join(input_dir, 'metadata', set + '.txt')
    with open(examples_metadata_path, 'r') as examples_list_file:
      examples_list = [line.replace('\n', '') for line in examples_list_file.readlines()]
      np.random.shuffle(examples_list)
    for example_name in tqdm.tqdm(examples_list):
      annotation_path = os.path.join(annotations_dir, example_name + '.xml')
      with tf.gfile.GFile(annotation_path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      annotation = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
      image_path = os.path.join(images_dir, example_name + '.jpg')
      with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
      encoded_jpg_io = io.BytesIO(encoded_jpg)
      image = PIL.Image.open(encoded_jpg_io)
      if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
      tf_example = dict_to_tf_example(example_name, annotation, encoded_jpg, label_map_dict)
      writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
  tf.app.run()
