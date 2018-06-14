from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

DATA_HOME_DEFAULT = ""
DATA_OUT_DIR = "output"
DATA_DIV_OUT_DIR = "category"
IMAGES = "images"
ANNOTATIONS = "annotations"
METADATA = "metadata"
METADATA_DIV = "metadata"
LABEL_MAP = "label_map.pbtxt"
SUMMARY = "summary.txt"
TRAIN = "train"
VALID = "valid"
TRAIN_ADDITION = "train_addition"
TRAINVALID = "trainvalid"
TEST = "test"
DEV = "dev"
DEV_ADDITION = "dev_addition"
CLASS_DEFAULT = ''
ARGUMENT_DEFAULT = ''
RATIO_DEFAULT = 0.1
BATCH_SIZE = 1
IMAGE_SIZE = '480, 640, 3'

import os
import cv2
import sys
import time
import tqdm
import copy
import shutil
import argparse
import functools
import numpy as np
import tensorflow as tf

from lxml import etree
from multiprocessing.dummy import Pool

sys.path.append('..')
sys.path.append('../slim')
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=DATA_HOME_DEFAULT, help="Data Input Home Path")
parser.add_argument("--ratio", type=int, default=RATIO_DEFAULT,
                    help="Ratio(Valid) For Dividing The Dataset To Train And Valid")
parser.add_argument("--classes", type=str, default=CLASS_DEFAULT,
                    help="Which Classes In Dataset Will Be Merged To This Path output_[time_stamp]/dev_images\n For Example: --classes 1,2 Refers To First Snd Second Classes")
parser.add_argument("--arguments", type=str, default=ARGUMENT_DEFAULT,
                    help="Which Argument Methods Will Be Used\n For Example --arguments 1,2 Refers That We Will Use First and Second Argument Method\n1.Brightness Up 2.Brightness Down 3.Saturation Up 4.Saturation Down 5.Contrast Up 6.Contrast Down")
parser.add_argument("--argument_all", action='store_true',
                    help="Argument On All Data Instead Of Images in output_[time_stamp]/dev_images")
parser.add_argument("--class_all", action='store_true',
                    help="Output All Divided Data To output_[time_stamp]/Category")
parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                    help="Batch Size Of Tensorflow Session")
parser.add_argument("--size", type=str, default=IMAGE_SIZE,
                    help="image Size Of Tensorflow Session")
parser.add_argument("--previous", type=str, default="",
                    help="Provide Previous Preprocessed Data Dir So That You Can Use Given Train/Valid/TrainValid Metadata Instead Of Randomly Generate A New One ")
args = parser.parse_args()
DATA_HOME = args.path
IMAGES_IN = os.path.join(DATA_HOME, IMAGES)
ANNOTATIONS_IN = os.path.join(DATA_HOME, ANNOTATIONS)
DATA_OUT_DIR = os.path.join(DATA_HOME, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + '_' + DATA_OUT_DIR)
IMAGES_OUT = os.path.join(DATA_OUT_DIR, IMAGES)
ANNOTATIONS_OUT = os.path.join(DATA_OUT_DIR, ANNOTATIONS)
DATA_DIV_DIR = os.path.join(DATA_OUT_DIR, DATA_DIV_OUT_DIR)
DATA_DIV_TRAIN_DIR = os.path.join(DATA_DIV_DIR, TRAIN)
DATA_DIV_VALID_DIR = os.path.join(DATA_DIV_DIR, VALID)
SUMMARY_FILE = os.path.join(DATA_OUT_DIR, SUMMARY)
METADATA_PATH = os.path.join(DATA_OUT_DIR, METADATA)
TRAIN_FILE = os.path.join(DATA_OUT_DIR, METADATA, TRAIN + '.txt')
TRAIN_ADDITION_FILE = os.path.join(DATA_OUT_DIR, METADATA, TRAIN_ADDITION + '.txt')
VALID_FILE = os.path.join(DATA_OUT_DIR, METADATA, VALID + '.txt')
TRAIN_DIV_FILE_PATH = os.path.join(DATA_DIV_DIR, METADATA_DIV, TRAIN)
VALID_DIV_FILE_PATH = os.path.join(DATA_DIV_DIR, METADATA_DIV, VALID)
TRAINVALID_FILE = os.path.join(DATA_OUT_DIR, METADATA, TRAINVALID + '.txt')
TEST_FILE = os.path.join(DATA_OUT_DIR, METADATA, TEST + '.txt')
DEV_FILE = os.path.join(DATA_OUT_DIR, METADATA, DEV + '.txt')
DEV_ADDITION_FILE = os.path.join(DATA_OUT_DIR, METADATA, DEV_ADDITION + '.txt')
RATIO = args.ratio
ARGUMENT_ALL = args.argument_all
CLASSES_ALL = args.class_all
BATCH_SIZE = args.batch
PREVIOUS = args.previous
try:
  CHOSEN_CLASSES = eval("[" + args.classes + "]")
except:
  raise SyntaxError("Not Allowed Format Of Argument --classes {}".format(
    args.classes) + "\nYou Shoud Use For Example --classes 1,2 instead")
try:
  if args.arguments not in ["None", "none", "-1", "0"]:
    CHOSEN_ARGUMENTS = eval("[" + args.arguments + "]")
  else:
    CHOSEN_ARGUMENTS = []
except:
  raise SyntaxError("Not Allowed Format Of Argument --arguments {}".format(
    args.arguments) + "\nYou Shoud Use For Example --arguments 1,1 instead")
try:
  IMAGE_SIZE = eval("[" + args.size + "]")
except:
  raise SyntaxError("Not Allowed Format Of Argument --size {}".format(
    args.classes) + "\nYou Shoud Use For Example --size 480,640,3 instead")


def rmdir(path):
  for i in os.listdir(path):
    path_file = os.path.join(path, i)
    if os.path.isfile(path_file):
      os.remove(path_file)
    else:
      rmdir(path_file)
  os.rmdir(path)


# tf.image.adjust_hue(img, 0.1)
# tf.image.adjust_saturation(img, 1.5)

with tf.device('/cpu:0'):
  image_placeholder = tf.placeholder('uint8', [None, None, None, 3], 'image_tensor')
  brightness_up_tensor = tf.image.adjust_brightness(image_placeholder, 0.2)
  brightness_down_tensor = tf.image.adjust_brightness(image_placeholder, -0.1)
  adjust_hue_tensor = tf.image.adjust_hue(image_placeholder, 0.1)
  saturation_up_tensor = tf.image.adjust_saturation(image_placeholder, 1.5)
  saturation_down_tensor = tf.image.adjust_saturation(image_placeholder, -0.5)
  contrast_up_tensor = tf.image.adjust_contrast(image_placeholder, 1.5)
  contrast_down_tensor = tf.image.adjust_contrast(image_placeholder, -0.5)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def brightness_up(img):
  return sess.run(brightness_up_tensor, {'image_tensor:0': img})


def brightness_down(img):
  return sess.run(brightness_down_tensor, {'image_tensor:0': img})


def adjust_hue(img):
  return sess.run(adjust_hue_tensor, {'image_tensor:0': img})


def saturation_up(img):
  return sess.run(saturation_up_tensor, {'image_tensor:0': img})


def saturation_down(img):
  return sess.run(saturation_down_tensor, {'image_tensor:0': img})


def contrast_up(img):
  return sess.run(contrast_up_tensor, {'image_tensor:0': img})


def contrast_down(img):
  return sess.run(contrast_down_tensor, {'image_tensor:0': img})


def argument_wrapper(method, train_list, index):
  img_list = train_list[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
  pathin = [os.path.join(IMAGES_IN, img + '.jpg') for img in img_list]
  img_tensor = [cv2.imread(path) for path in pathin]
  if BATCH_SIZE > 1:
    img_tensor = [i for i in img_tensor if i.shape[0] == IMAGE_SIZE[0] and i.shape[1] == IMAGE_SIZE[1]]
    img_list = [j for i, j in zip(img_tensor, img_list) if
                i.shape[0] == IMAGE_SIZE[0] and i.shape[1] == IMAGE_SIZE[1]]
  if len(img_list) == 0:
    return []
  pathout = [os.path.join(IMAGES_OUT, img + '_' + method.__name__ + '.jpg') for img in img_list]
  img_result = method(np.asarray(img_tensor))
  for i, img in enumerate(img_list):
    cv2.imwrite(pathout[i], img_result[i])
    shutil.copy(os.path.join(ANNOTATIONS_IN, img + '.xml'),
                os.path.join(ANNOTATIONS_OUT, img + '_' + method.__name__ + '.xml'))
  return [img + '_' + method.__name__ for img in img_list]


ARGUMENT_METHODS = [brightness_up, brightness_down, saturation_up, saturation_down, contrast_up, contrast_down]

assert os.path.exists(DATA_HOME), "Data Home Path Not Found!"
assert os.path.exists(IMAGES_IN), "Images Dir Path Not Found!"
assert os.path.exists(ANNOTATIONS_IN), "Annotations Path Not Found!"

if not os.path.exists(DATA_OUT_DIR):
  os.mkdir(DATA_OUT_DIR)
  os.mkdir(IMAGES_OUT)
  os.mkdir(ANNOTATIONS_OUT)
  print("Data Output Path [{},{},{}] Created".format(DATA_OUT_DIR, IMAGES_OUT, ANNOTATIONS_OUT))
else:
  rmdir(os.path.join(DATA_HOME, DATA_OUT_DIR))
  os.mkdir(DATA_OUT_DIR)
  os.mkdir(IMAGES_OUT)
  os.mkdir(ANNOTATIONS_OUT)
  print("Data Output Path [{},{},{}] Created".format(DATA_OUT_DIR, IMAGES_OUT, ANNOTATIONS_OUT))

if not os.path.exists(DATA_DIV_DIR):
  os.mkdir(DATA_DIV_DIR)
  print("Divided Output Path [{}] Created".format(DATA_DIV_DIR))
  os.mkdir(DATA_DIV_TRAIN_DIR)
  print("Divided Output Path [{}] Created".format(DATA_DIV_TRAIN_DIR))
  os.mkdir(DATA_DIV_VALID_DIR)
  print("Divided Output Path [{}] Created".format(DATA_DIV_VALID_DIR))
  os.mkdir(os.path.join(DATA_DIV_DIR, METADATA_DIV))
  os.mkdir(TRAIN_DIV_FILE_PATH)
  os.mkdir(VALID_DIV_FILE_PATH)
  print("Divided Output Path [{}] Created".format(os.path.join(DATA_DIV_DIR, METADATA_DIV)))
if not os.path.exists(METADATA_PATH):
  os.mkdir(METADATA_PATH)
  print("Metadata Path [{}] Created".format(METADATA_PATH))

if __name__ == "__main__":
  annotation_files = os.listdir(ANNOTATIONS_IN)
  trainvalid_image_files = os.listdir(IMAGES_IN)

  shutil.copy(os.path.join(DATA_HOME, LABEL_MAP), os.path.join(DATA_OUT_DIR, LABEL_MAP))
  label_map_dict = label_map_util.get_label_map_dict(os.path.join(DATA_OUT_DIR, LABEL_MAP))
  trainvalid_label_count = dict(
    zip(list(label_map_dict.keys()), np.zeros([len(label_map_dict), 2]).astype("int32").tolist()))
  train_label_count = dict(
    zip(list(label_map_dict.keys()), np.zeros([len(label_map_dict), 2]).astype("int32").tolist()))
  valid_label_count = dict(
    zip(list(label_map_dict.keys()), np.zeros([len(label_map_dict), 2]).astype("int32").tolist()))
  test_label_count = dict(
    zip(list(label_map_dict.keys()), np.zeros([len(label_map_dict), 2]).astype("int32").tolist()))
  dev_label_count = dict(
    zip(list(label_map_dict.keys()), np.zeros([len(label_map_dict), 2]).astype("int32").tolist()))
  trainvalid_label_count_sum = [0, 0, 0]
  train_label_count_sum = [0, 0, 0]
  valid_label_count_sum = [0, 0, 0]
  test_label_count_sum = [0, 0, 0]
  dev_label_count_sum = [0, 0, 0]

  label_list = dict(zip(list(label_map_dict.keys()), np.zeros([len(label_map_dict), 2]).astype("int32").tolist()))
  trainvalid_list = []
  train_list = []
  train_list_addition = []
  valid_list = []
  train_cat_list = {}
  valid_cat_list = {}
  test_list = []
  dev_list = []
  dev_list_addition = []

  CLASSES = list(label_map_dict.keys())
  CHOSEN_CLASSES = [CLASSES[i - 1] for i in CHOSEN_CLASSES]
  CHOSEN_ARGUMENTS_NAME = str([ARGUMENT_METHODS[i - 1].__name__ for i in CHOSEN_ARGUMENTS])
  CHOSEN_ARGUMENTS = [functools.partial(argument_wrapper, ARGUMENT_METHODS[i - 1]) for i in CHOSEN_ARGUMENTS]
  print("Argument with: " + CHOSEN_ARGUMENTS_NAME)

  for cls in CLASSES:
    if cls in CHOSEN_CLASSES or CLASSES_ALL:
      os.mkdir(os.path.join(DATA_DIV_TRAIN_DIR, cls))
      os.mkdir(os.path.join(DATA_DIV_VALID_DIR, cls))
    train_cat_list[cls] = []
    valid_cat_list[cls] = []

  print('Summarizing ADAS TrainValid Dataset...')
  for img in tqdm.tqdm(trainvalid_image_files):
    if not (img.replace(".jpg", ".xml") in annotation_files):
      print("[{}] Not found!".format(img.replace(".jpg", ".xml")))
      continue
    with tf.gfile.GFile(os.path.join(ANNOTATIONS_IN, img.replace(".jpg", ".xml")), 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    obj_name_set = set([obj["name"] for obj in data["object"]])
    for obj in data["object"]:
      trainvalid_label_count[obj["name"]][1] += 1
      trainvalid_label_count_sum[1] += 1
    for label in label_map_dict.keys():
      if label in obj_name_set:
        trainvalid_label_count[label][0] += 1
        trainvalid_label_count_sum[0] += 1
    trainvalid_label_count_sum[2] += 1
    trainvalid_list.append(img.replace(".jpg", ""))

  print('Dividing Dataset To TrainSet And ValidSet...')
  if PREVIOUS != "":
    print('Reading Summarized ADAS TrainValid Data From Previous Output Directory [{}]...'.format(PREVIOUS))
    with open(os.path.join(PREVIOUS, METADATA, 'valid.txt'), 'r') as f:
      valid_list = [i.replace('\n', '') for i in f.readlines()]
  for img in valid_list:
    with tf.gfile.GFile(os.path.join(ANNOTATIONS_IN, img + ".xml"), 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    obj_name_set = set([obj["name"] for obj in data["object"]])
    for obj in data["object"]:
      valid_label_count[obj["name"]][1] += 1
      valid_label_count_sum[1] += 1
    valid_label_count_sum[2] += 1
  trainvalid_remain_list = copy.copy(trainvalid_list)
  for img in valid_list:
    if img in trainvalid_remain_list:
      trainvalid_remain_list.remove(img)
  for label_index, label in enumerate(label_list):
    count_require = int(RATIO * trainvalid_label_count[label][1])
    count_require = count_require - valid_label_count[label][1]
    np.random.shuffle(trainvalid_remain_list)
    index = 0
    need_list = []
    while count_require > 0:
      index += 1
      with tf.gfile.GFile(os.path.join(ANNOTATIONS_IN, trainvalid_remain_list[index] + '.xml'), 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
      obj_name_set = set([obj["name"] for obj in data["object"]])
      found = False
      for obj in data["object"]:
        if obj["name"] == label:
          found = True
          count_require -= 1
      if found:
        need_list.append(trainvalid_remain_list[index])
        for obj in data["object"]:
          valid_label_count[obj["name"]][1] += 1
          valid_label_count_sum[1] += 1
        valid_label_count_sum[2] += 1
    valid_list.extend(need_list)
    for need_name in need_list:
      trainvalid_remain_list.remove(need_name)
  for name in trainvalid_list:
    if name not in valid_list:
      train_list.append(name)
  if ARGUMENT_ALL:
    print('Argumenting ADAS Train Dataset...')
    train_list_addition = []
    for method in CHOSEN_ARGUMENTS:
      n_batch = (len(train_list) + 1) // BATCH_SIZE - 1
      for i in tqdm.tqdm(range(n_batch)):
        train_list_addition.extend(method(train_list, i))
    train_list.extend(train_list_addition)
  print('Generating Dev Set')
  for img in train_list:
    if os.path.exists(os.path.join(ANNOTATIONS_OUT, img + ".xml")):
      with tf.gfile.GFile(os.path.join(ANNOTATIONS_OUT, img + ".xml"), 'r') as fid:
        xml_str = fid.read()
    else:
      with tf.gfile.GFile(os.path.join(ANNOTATIONS_IN, img + ".xml"), 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    obj_name_set = set([obj["name"] for obj in data["object"]])
    for obj in data["object"]:
      if obj["name"] in CHOSEN_CLASSES:
        dev_list.append(img)
  print('Generating Test Set')
  for img in valid_list:
    if os.path.exists(os.path.join(ANNOTATIONS_OUT, img + ".xml")):
      with tf.gfile.GFile(os.path.join(ANNOTATIONS_OUT, img + ".xml"), 'r') as fid:
        xml_str = fid.read()
    else:
      with tf.gfile.GFile(os.path.join(ANNOTATIONS_IN, img + ".xml"), 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    obj_name_set = set([obj["name"] for obj in data["object"]])
    for obj in data["object"]:
      if obj["name"] in CHOSEN_CLASSES:
        test_list.append(img)
  if not ARGUMENT_ALL:
    print('Argumenting ADAS Dev Dataset...')
    dev_list_addition = []
    for method in CHOSEN_ARGUMENTS:
      n_batch = (len(dev_list) + 1) // BATCH_SIZE - 1
      for i in tqdm.tqdm(range(n_batch)):
        dev_list_addition.extend(method(dev_list, i))
    dev_list.extend(dev_list_addition)
  dev_list = list(set(dev_list))
  test_list = list(set(test_list))

  trainvalid_label_count_sum = [0, 0, 0]
  train_label_count_sum = [0, 0, 0]
  valid_label_count_sum = [0, 0, 0]
  trainvalid_label_count = dict(
    zip(list(label_map_dict.keys()), np.zeros([len(label_map_dict), 2]).astype("int32").tolist()))
  train_label_count = dict(
    zip(list(label_map_dict.keys()), np.zeros([len(label_map_dict), 2]).astype("int32").tolist()))
  valid_label_count = dict(
    zip(list(label_map_dict.keys()), np.zeros([len(label_map_dict), 2]).astype("int32").tolist()))
  print('Reading From ADAS Valid Dataset...')
  for img in tqdm.tqdm(valid_list):
    with tf.gfile.GFile(os.path.join(ANNOTATIONS_IN, img + ".xml"), 'r') as fid:
      xml_str = fid.read()
    shutil.copy(os.path.join(IMAGES_IN, img + '.jpg'),
                os.path.join(IMAGES_OUT, img + '.jpg'))
    shutil.copy(os.path.join(ANNOTATIONS_IN, img + '.xml'),
                os.path.join(ANNOTATIONS_OUT, img + '.xml'))
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    obj_name_set = set([obj["name"] for obj in data["object"]])
    for i, obj in enumerate(data["object"]):
      valid_label_count[obj["name"]][1] += 1
      valid_label_count_sum[1] += 1
      trainvalid_label_count[obj["name"]][1] += 1
      trainvalid_label_count_sum[1] += 1
      valid_cat_list[obj["name"]].append(img)
      if (obj["name"] in CHOSEN_CLASSES) or CLASSES_ALL:
        shutil.copy(os.path.join(IMAGES_IN, img + '.jpg'),
                    os.path.join(DATA_DIV_VALID_DIR, obj["name"], img + '.jpg'))
    for label in label_map_dict.keys():
      if label in obj_name_set:
        valid_label_count[label][0] += 1
        valid_label_count_sum[0] += 1
        trainvalid_label_count[label][0] += 1
        trainvalid_label_count_sum[0] += 1
    valid_label_count_sum[2] += 1
    trainvalid_label_count_sum[2] += 1

  print('Reading From ADAS Train Dataset...')
  for img in tqdm.tqdm(train_list):
    if img in train_list_addition:
      with tf.gfile.GFile(os.path.join(ANNOTATIONS_OUT, img + ".xml"), 'r') as fid:
        xml_str = fid.read()
    else:
      with tf.gfile.GFile(os.path.join(ANNOTATIONS_IN, img + ".xml"), 'r') as fid:
        xml_str = fid.read()
      shutil.copy(os.path.join(IMAGES_IN, img + '.jpg'),
                  os.path.join(IMAGES_OUT, img + '.jpg'))
      shutil.copy(os.path.join(ANNOTATIONS_IN, img + '.xml'),
                  os.path.join(ANNOTATIONS_OUT, img + '.xml'))
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    obj_name_set = set([obj["name"] for obj in data["object"]])
    for i, obj in enumerate(data["object"]):
      train_label_count[obj["name"]][1] += 1
      train_label_count_sum[1] += 1
      trainvalid_label_count[obj["name"]][1] += 1
      trainvalid_label_count_sum[1] += 1
      train_cat_list[obj["name"]].append(img)
      if (obj["name"] in CHOSEN_CLASSES) or CLASSES_ALL:
        shutil.copy(os.path.join(IMAGES_OUT, img + '.jpg'),
                    os.path.join(DATA_DIV_TRAIN_DIR, obj["name"], img + '.jpg'))
    for label in label_map_dict.keys():
      if label in obj_name_set:
        train_label_count[label][0] += 1
        train_label_count_sum[0] += 1
        trainvalid_label_count[label][0] += 1
        trainvalid_label_count_sum[0] += 1
    train_label_count_sum[2] += 1
    trainvalid_label_count_sum[2] += 1
  print('Reading From ADAS Test Dataset...')
  for img in tqdm.tqdm(test_list):
    with tf.gfile.GFile(os.path.join(ANNOTATIONS_OUT, img + ".xml"), 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    obj_name_set = set([obj["name"] for obj in data["object"]])
    for obj in data["object"]:
      test_label_count[obj["name"]][1] += 1
      test_label_count_sum[1] += 1
    for label in label_map_dict.keys():
      if label in obj_name_set:
        test_label_count[label][0] += 1
        test_label_count_sum[1] += 1
    test_label_count_sum[2] += 1
  print('Reading From ADAS Dev Dataset...')
  for img in tqdm.tqdm(dev_list):
    with tf.gfile.GFile(os.path.join(ANNOTATIONS_OUT, img + ".xml"), 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    obj_name_set = set([obj["name"] for obj in data["object"]])
    for obj in data["object"]:
      dev_label_count[obj["name"]][1] += 1
      dev_label_count_sum[1] += 1
    for label in label_map_dict.keys():
      if label in obj_name_set:
        dev_label_count[label][0] += 1
        dev_label_count_sum[1] += 1
    dev_label_count_sum[2] += 1

  print("Testing ...")
  assert len(trainvalid_list) == len(set(trainvalid_list)), "Dummy label in trainvalid list! {}!={}".format(
    len(trainvalid_list), len(set(trainvalid_list)))
  assert len(train_list) == len(set(train_list)), "Dummy label in train list! {}!={}".format(len(train_list),
                                                                                             len(set(train_list)))
  assert len(valid_list) == len(set(valid_list)), "Dummy label in valid list! {}!={}".format(len(valid_list),
                                                                                             len(set(valid_list)))
  assert len(dev_list) == len(set(dev_list)), "Dummy label in dev list! {}!={}".format(len(dev_list),
                                                                                       len(set(dev_list)))
  assert len(test_list) == len(set(test_list)), "Dummy label in test list! {}!={}".format(len(test_list),
                                                                                          len(set(test_list)))
  for img in train_list:
    assert img not in valid_list, "Image in train list shouldn't be in valid list! {}".format(img)
    assert img not in test_list, "Image in train list shouldn't be in test list! {}".format(img)
  for img in dev_list:
    assert img not in valid_list, "Image in dev list shouldn't be in valid list! {}".format(img)
    assert img not in test_list, "Image in dev list shouldn't be in test list! {}".format(img)
  if PREVIOUS != "":
    with open(os.path.join(PREVIOUS, METADATA, 'valid.txt'), 'r') as f:
      valid_list_previous = [i.replace('\n', '') for i in f.readlines()]
      for img in train_list:
        assert img not in valid_list_previous, "Image in train list shouldn't be in previous valid list! {}".format(
          img)
      for img in dev_list:
        assert img not in valid_list_previous, "Image in dev list shouldn't be in previous valid list! {}".format(
          img)

  with open(SUMMARY_FILE, "w") as sumfile:
    sumfile.write("Merged {} in dev sets\n".format(str(CHOSEN_CLASSES)))
    if ARGUMENT_ALL:
      sumfile.write("Argument on all training set with: " + CHOSEN_ARGUMENTS_NAME)
    else:
      sumfile.write("Argument on all dev set with: " + CHOSEN_ARGUMENTS_NAME)

    sumfile.write("\n\n\n\nTrainValid\n")
    sumfile.write("-" * 39 + "\n")
    sumfile.write("Cat          ImgCount          BoxCount\n")
    for label in label_map_dict.keys():
      sumstr = label
      sumstr += " " * (21 - len(label) - len(str(trainvalid_label_count[label][0]))) + str(
        trainvalid_label_count[label][0])
      sumstr += " " * (18 - len(str(trainvalid_label_count[label][1]))) + str(trainvalid_label_count[label][1])
      sumfile.write(sumstr + '\n')
    sumstr = "Total"
    sumstr += " " * (21 - len("Total") - len(str(trainvalid_label_count_sum[0]))) + str(
      trainvalid_label_count_sum[0])
    tmpstr = str(trainvalid_label_count_sum[1]) + "/" + str(trainvalid_label_count_sum[2])
    sumstr += " " * (18 - len(tmpstr)) + tmpstr
    sumfile.write(sumstr + '\n')

    sumfile.write("\n\n\nTrain\n")
    sumfile.write("-" * 39 + "\n")
    sumfile.write("Cat          ImgCount          BoxCount\n")
    for label in label_map_dict.keys():
      sumstr = label
      sumstr += " " * (21 - len(label) - len(str(train_label_count[label][0]))) + str(
        train_label_count[label][0])
      sumstr += " " * (18 - len(str(train_label_count[label][1]))) + str(train_label_count[label][1])
      sumfile.write(sumstr + '\n')
    sumstr = "Total"
    sumstr += " " * (21 - len("Total") - len(str(train_label_count_sum[0]))) + str(
      train_label_count_sum[0])
    tmpstr = str(train_label_count_sum[1]) + "/" + str(train_label_count_sum[2])
    sumstr += " " * (18 - len(tmpstr)) + tmpstr
    sumfile.write(sumstr + '\n')

    sumfile.write("\n\n\nValid\n")
    sumfile.write("-" * 39 + "\n")
    sumfile.write("Cat          ImgCount          BoxCount\n")
    for label in label_map_dict.keys():
      sumstr = label
      sumstr += " " * (21 - len(label) - len(str(valid_label_count[label][0]))) + str(
        valid_label_count[label][0])
      sumstr += " " * (18 - len(str(valid_label_count[label][1]))) + str(valid_label_count[label][1])
      sumfile.write(sumstr + '\n')
    sumstr = "Total"
    sumstr += " " * (21 - len("Total") - len(str(valid_label_count_sum[0]))) + str(
      valid_label_count_sum[0])
    tmpstr = str(valid_label_count_sum[1]) + "/" + str(valid_label_count_sum[2])
    sumstr += " " * (18 - len(tmpstr)) + tmpstr
    sumfile.write(sumstr + '\n')

    sumfile.write("\n\n\nTest\n")
    sumfile.write("-" * 39 + "\n")
    sumfile.write("Cat          ImgCount          BoxCount\n")
    for label in label_map_dict.keys():
      sumstr = label
      sumstr += " " * (21 - len(label) - len(str(test_label_count[label][0]))) + str(
        test_label_count[label][0])
      sumstr += " " * (18 - len(str(test_label_count[label][1]))) + str(test_label_count[label][1])
      sumfile.write(sumstr + '\n')
    sumstr = "Total"
    sumstr += " " * (21 - len("Total") - len(str(test_label_count_sum[0]))) + str(
      test_label_count_sum[0])
    tmpstr = str(test_label_count_sum[1]) + "/" + str(test_label_count_sum[2])
    sumstr += " " * (18 - len(tmpstr)) + tmpstr
    sumfile.write(sumstr + '\n')

    sumfile.write("\n\n\nDev\n")
    sumfile.write("-" * 39 + "\n")
    sumfile.write("Cat          ImgCount          BoxCount\n")
    for label in label_map_dict.keys():
      sumstr = label
      sumstr += " " * (21 - len(label) - len(str(dev_label_count[label][0]))) + str(
        dev_label_count[label][0])
      sumstr += " " * (18 - len(str(dev_label_count[label][1]))) + str(dev_label_count[label][1])
      sumfile.write(sumstr + '\n')
    sumstr = "Total"
    sumstr += " " * (21 - len("Total") - len(str(dev_label_count_sum[0]))) + str(
      dev_label_count_sum[0])
    tmpstr = str(dev_label_count_sum[1]) + "/" + str(dev_label_count_sum[2])
    sumstr += " " * (18 - len(tmpstr)) + tmpstr
    sumfile.write(sumstr + '\n')

  with open(TRAINVALID_FILE, "w") as f:
    for trainvalid_name in trainvalid_list:
      f.write(trainvalid_name + "\n")
  with open(TRAIN_FILE, "w") as f:
    if PREVIOUS != "":
      np.random.shuffle(train_list)
    for train_name in train_list:
      f.write(train_name + "\n")
  with open(TRAIN_FILE, "w") as f:
    if PREVIOUS != "":
      np.random.shuffle(train_list)
    for train_name in train_list:
      f.write(train_name + "\n")
  with open(VALID_FILE, "w") as f:
    for valid_name in valid_list:
      f.write(valid_name + "\n")
  with open(TEST_FILE, "w") as f:
    for test_name in test_list:
      f.write(test_name + "\n")
  with open(DEV_FILE, "w") as f:
    np.random.shuffle(dev_list)
    for dev_name in dev_list:
      f.write(dev_name + "\n")
  with open(DEV_ADDITION_FILE, "w") as f:
    for dev_name in dev_list_addition:
      f.write(dev_name + "\n")
  for i, label in enumerate(label_list):
    with open(os.path.join(TRAIN_DIV_FILE_PATH, label + '.txt'), "w") as f:
      for train_cat_name in train_cat_list[label]:
        f.write(train_cat_name + "\n")
  for i, label in enumerate(label_list):
    with open(os.path.join(VALID_DIV_FILE_PATH, label + '.txt'), "w") as f:
      for valid_cat_name in valid_cat_list[label]:
        f.write(valid_cat_name + "\n")
  print("\nFinnished")
