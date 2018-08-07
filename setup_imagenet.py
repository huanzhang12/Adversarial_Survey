## Modified by Huan Zhang for the updated Inception-v3 model (inception_v3_2016_08_28.tar.gz)
## Modified by Nicholas Carlini to match model structure for attack code.
## Original copyright license follows.


# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Simple image classification with an ImageNet Classifier.

Run image classification with an ImageNet Classifier (Inception, ResNet, AlexNet, etc) trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
from functools import partial
import random
import tarfile
import scipy.misc

import numpy as np
from six.moves import urllib
import tensorflow as tf

import PIL
from PIL import Image


model_params = {}


"""Add a new new entry to ImageNet models

Parameters:
name: name of the new model, like "resnet"
url: URL to download the model
image_size: image size, usually 224 or 299
model_filename: model protobuf file name (.pb)
label_filename: a text file contains the mapping from class ID to human readable string
input_tensor: input tensor of the network defined by protobuf, like "input:0"
logit: logit output tensor of the network, like "resnet_v2_50/predictions/Reshape:0"
prob: probability output tensor of the network, like "resnet_v2_50/predictions/Reshape_1:0"
shape: tensor for reshaping the final output, like "resnet_v2_50/predictions/Shape:0".
       Set to None if no reshape needed.

All the tensor names can be viewed and found in TensorBoard.
"""
def AddModel(name, url, model_filename, image_size, label_filename, input_tensor, logit, prob, shape):
  global model_params
  param = {}
  param['url'] = url
  param['model_filename'] = model_filename
  param['size'] = image_size
  param['input'] = input_tensor
  param['logit'] = logit
  param['prob'] = prob
  param['shape'] = shape
  param['label_filename'] = label_filename
  param['name'] = name
  model_params[name] = param

# pylint: disable=line-too-long
AddModel('resnet_v2_50', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_resnet_v2_50.pb', 299, 'labels.txt', 'input:0',
         'resnet_v2_50/predictions/Reshape:0', 'resnet_v2_50/predictions/Reshape_1:0', 'resnet_v2_50/predictions/Shape:0')
AddModel('resnet_v2_101', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_resnet_v2_101.pb', 299, 'labels.txt', 'input:0',
         'resnet_v2_101/predictions/Reshape:0', 'resnet_v2_101/predictions/Reshape_1:0', 'resnet_v2_101/predictions/Shape:0')
AddModel('resnet_v2_152', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_resnet_v2_152.pb', 299, 'labels.txt', 'input:0',
         'resnet_v2_152/predictions/Reshape:0', 'resnet_v2_152/predictions/Reshape_1:0', 'resnet_v2_152/predictions/Shape:0')
AddModel('inception_v1', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_inception_v1.pb', 224, 'labels.txt', 'input:0',
         'InceptionV1/Logits/Predictions/Reshape:0', 'InceptionV1/Logits/Predictions/Reshape_1:0', 'InceptionV1/Logits/Predictions/Shape:0')
AddModel('inception_v2', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_inception_v2.pb', 224, 'labels.txt', 'input:0',
         'InceptionV2/Predictions/Reshape:0', 'InceptionV2/Predictions/Reshape_1:0', 'InceptionV2/Predictions/Shape:0')
AddModel('inception_v3', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_inception_v3.pb', 299, 'labels.txt', 'input:0',
         'InceptionV3/Predictions/Reshape:0', 'InceptionV3/Predictions/Softmax:0', 'InceptionV3/Predictions/Shape:0')
AddModel('inception_v4', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_inception_v4.pb', 299, 'labels.txt', 'input:0',
         'InceptionV4/Logits/Logits/BiasAdd:0', 'InceptionV4/Logits/Predictions:0', '')
AddModel('inception_resnet_v2', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_inception_resnet_v2.pb', 299, 'labels.txt', 'input:0',
         'InceptionResnetV2/Logits/Logits/BiasAdd:0', 'InceptionResnetV2/Logits/Predictions:0', '')
AddModel('vgg_16', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_vgg_16.pb', 224, 'labels.txt', 'input:0',
         'vgg_16/fc8/squeezed:0', 'vgg_16/fc8/squeezed:0', '')
AddModel('vgg_19', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_vgg_19.pb', 224, 'labels.txt', 'input:0',
         'vgg_19/fc8/squeezed:0', 'vgg_19/fc8/squeezed:0', '')
AddModel('mobilenet_v1_025', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_mobilenet_v1_025.pb', 224, 'labels.txt', 'input:0',
         'MobilenetV1/Predictions/Reshape:0', 'MobilenetV1/Predictions/Reshape_1:0', 'MobilenetV1/Predictions/Shape:0')
AddModel('mobilenet_v1_050', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_mobilenet_v1_050.pb', 224, 'labels.txt', 'input:0',
         'MobilenetV1/Predictions/Reshape:0', 'MobilenetV1/Predictions/Reshape_1:0', 'MobilenetV1/Predictions/Shape:0')
AddModel('mobilenet_v1_100', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_mobilenet_v1_100.pb', 224, 'labels.txt', 'input:0',
         'MobilenetV1/Predictions/Reshape:0', 'MobilenetV1/Predictions/Reshape_1:0', 'MobilenetV1/Predictions/Shape:0')
AddModel('nasnet_large', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'frozen_nasnet_large.pb', 331, 'labels.txt', 'input:0',
         'final_layer/FC/BiasAdd:0', 'final_layer/predictions:0', '')
AddModel('densenet121_k32', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'densenet121_k32_frozen.pb', 224, 'labels.txt', 'input:0',
         'densenet121/predictions/Reshape:0', 'densenet121/predictions/Reshape_1:0', 'densenet121/predictions/Shape:0')
AddModel('densenet169_k32', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'densenet169_k32_frozen.pb', 224, 'labels.txt', 'input:0',
         'densenet169/predictions/Reshape:0', 'densenet169/predictions/Reshape_1:0', 'densenet169/predictions/Shape:0')
AddModel('densenet161_k48', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'densenet161_k48_frozen.pb', 224, 'labels.txt', 'input:0',
         'densenet161/predictions/Reshape:0', 'densenet161/predictions/Reshape_1:0', 'densenet161/predictions/Shape:0')
AddModel('alexnet', 'http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/frozen_imagenet_models_v1.1.tar.gz',
         'alexnet_frozen.pb', 227, 'labels.txt', 'Placeholder:0',
         'fc8/fc8:0', 'Softmax:0', '')

# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'labels.txt')
    self.node_lookup = self.load(label_lookup_path)

  def load(self, label_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to integer node ID.
    node_id_to_name = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line:
        words = line.split(':')
        target_class = int(words[0])
        name = words[1]
        node_id_to_name[target_class] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

LOADED_GRAPH = None

def create_graph(model_param):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  global LOADED_GRAPH
  with tf.gfile.FastGFile(os.path.join(
    #  FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
      FLAGS.model_dir, model_param['model_filename']), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    #for line in repr(graph_def).split("\n"):
    #  if "tensor_content" not in line:
    #    print(line)
    LOADED_GRAPH = graph_def


class ImageNetModelPrediction:
  def __init__(self, sess, use_softmax = False, model_name = "resnet_v2_50", softmax_tensor = None):
    self.sess = sess
    self.use_softmax = use_softmax
    model_param = model_params[model_name]
    self.output_name = model_param['prob'] if self.use_softmax else model_param['logit']
    self.input_name = model_param['input']
    self.shape_name = model_param['shape']
    self.model_name = model_param['name']
    self.image_size = model_param['size']
    self.img = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))
    if not softmax_tensor:
      # no existing graph
      self.softmax_tensor = tf.import_graph_def(
              LOADED_GRAPH,
              # sess.graph.as_graph_def(),
              input_map={self.input_name: self.img},
              return_elements=[self.output_name])
      if 'vgg' in self.model_name and use_softmax == True:
        # the pretrained VGG network output is logits, need an extra softmax
        self.softmax_tensor = tf.nn.softmax(self.softmax_tensor)
    else:
      # use an existing graph
      self.softmax_tensor = softmax_tensor
    print("GraphDef Size:", self.sess.graph_def.ByteSize())

  def predict(self, dat):
    dat = np.squeeze(dat)
    if 'vgg' in self.model_name:
      # VGG uses 0 - 255 image as input
      dat = (0.5 + dat) * 255.0
      imagenet_mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
      dat -= imagenet_mean
    elif 'alexnet' in self.model_name:
      if dat.ndim == 3:
        dat = dat[:,:,::-1]
      else:
        dat = dat[:,:,:,::-1] # change RGB to BGR
      dat = (0.5 + dat) * 255.0
      imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
      dat -= imagenet_mean
    elif 'densenet' in self.model_name:
      dat = (0.5 + dat) * 255.0
      imagenet_mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
      dat -= imagenet_mean
      dat = dat * 0.017
    else:
      dat = dat * 2.0


    if dat.ndim == 3:
      scaled = dat.reshape((1,) + dat.shape)
    else:
      scaled = dat
    # print(scaled.shape)
    predictions = self.sess.run(self.softmax_tensor,
                         {self.img: scaled})
    predictions = np.squeeze(predictions)
    return predictions
    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()
    top_k = predictions.argsort()#[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      print('id',node_id)
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
    return top_k[-1]


CREATED_GRAPH = False
class ImageNetModel:
  def __init__(self, sess, use_softmax = False, model_name = "resnet_v2_50", create_prediction = True):
    global CREATED_GRAPH
    self.sess = sess
    self.use_softmax = use_softmax
    model_param = model_params[model_name]
    maybe_download_and_extract(model_param)

    if not CREATED_GRAPH:
      create_graph(model_param)
      CREATED_GRAPH = True
    self.num_channels = 3
    self.output_name = model_param['prob'] if self.use_softmax else model_param['logit']
    self.input_name = model_param['input']
    self.shape_name = model_param['shape']
    self.model_name = model_param['name']
    self.num_labels = 1000 if 'vgg' in self.model_name or 'densenet' in self.model_name or 'alexnet' in self.model_name else 1001
    self.image_size = model_param['size']
    self.use_softmax = use_softmax
    if create_prediction:
      self.model = ImageNetModelPrediction(sess, use_softmax, model_name)

  def predict(self, img):
    if 'vgg' in self.model_name:
      # VGG uses 0 - 255 image as input
      img = (0.5 + img) * 255.0
      imagenet_mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
      img -= imagenet_mean
    elif 'alexnet' in self.model_name:
      img = tf.reverse(img,axis=[-1])# change RGB to BGR
      img = (0.5 + img) * 255.0
      imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
      img -= imagenet_mean
    elif 'densenet' in self.model_name:
      # convert to 0 - 255 image as input
      img = (0.5 + img) * 255.0
      imagenet_mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
      img -= imagenet_mean
      img = img * 0.017
    else:
      img = img * 2.0

    if img.shape.is_fully_defined() and img.shape.as_list()[0] and self.shape_name:
      # check if a shape has been specified explicitly
      shape = (int(img.shape[0]), self.num_labels)
      self.softmax_tensor = tf.import_graph_def(
        LOADED_GRAPH,
        # self.sess.graph.as_graph_def(),
        input_map={self.input_name: img, self.shape_name: shape},
        return_elements=[self.output_name])
      if 'vgg' in self.model_name and self.use_softmax == True:
        # the pretrained VGG network output is logitimport_graph_defs, need an extra softmax
        self.softmax_tensor = tf.nn.softmax(self.softmax_tensor)
    else:
      # placeholder shape
      self.softmax_tensor = tf.import_graph_def(
        LOADED_GRAPH,
        # self.sess.graph.as_graph_def(),
        input_map={self.input_name: img},
        return_elements=[self.output_name])
      if 'vgg' in self.model_name and self.use_softmax == True:
        # the pretrained VGG network output is logits, need an extra softmax
        self.softmax_tensor = tf.nn.softmax(self.softmax_tensor)
    print("GraphDef Size:", self.sess.graph_def.ByteSize())
    return self.softmax_tensor[0]


def maybe_download_and_extract(model_param):
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = model_param['url'].split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  modelname = model_param['model_filename'].split('/')[-1]
  modelpath = os.path.join(dest_directory, modelname)
  if not os.path.exists(modelpath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(model_param['url'], filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    if os.path.splitext(filename)[1] != '.pb':
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  param = model_params[FLAGS.model_name]
  maybe_download_and_extract(param)
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  # run_inference_on_image(image)
  create_graph(param)
  image_size = param['size']
  with tf.Session() as sess:
    dat = np.array(scipy.misc.imresize(scipy.misc.imread(image),(image_size, image_size)), dtype = np.float32)
    dat /= 255.0
    dat -= 0.5
    # print(dat)
    model = ImageNetModelPrediction(sess, True, FLAGS.model_name)
    predictions = model.predict(dat)
    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()
    top_k = predictions.argsort()#[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      score = predictions[node_id]
      if 'vgg' in FLAGS.model_name or 'densenet' in FLAGS.model_name or 'alexnet' in FLAGS.model_name:
        node_id += 1
      print('id',node_id)
      human_string = node_lookup.id_to_string(node_id)
      print('%s (score = %.5f)' % (human_string, score))


def keep_aspect_ratio_transform(img, img_size):

    s_0, s_1 = img.size
    if s_0 < s_1:
        ratio = (img_size / float(s_0))
        size_1 = int((float(img.size[1]) * float(ratio)))
        img = img.resize((img_size, size_1), PIL.Image.ANTIALIAS)
    else:
        ratio = (img_size / float(s_1))
        size_0 = int((float(img.size[0]) * float(ratio)))
        img = img.resize((size_0, img_size), PIL.Image.ANTIALIAS)

    c_0 = img.size[0] // 2
    c_1 = img.size[1] // 2

    if img_size % 2 == 0:
        w_left = h_top = img_size // 2
        w_right = h_bottom = img_size // 2
    else:
        w_left = h_top = img_size // 2
        w_right = h_bottom = img_size // 2 + 1

    transformed_img = img.crop(
        (
            c_0 - w_left,
            c_1 - h_top,
            c_0 + w_right,
            c_1 + h_bottom
        )
    )

    return transformed_img

def readimg(ff, img_size):
  f = "./imagenetdata/imgs/"+ff
  # img = scipy.misc.imread(f)
  # skip small images (image should be at least img_size X img_size)

  # if img.shape[0] < img_size or img.shape[1] < img_size:
  #   return None

  # img = np.array(scipy.misc.imresize(img,(img_size, img_size)),dtype=np.float32)/255.0-.5
  img = Image.open(f)
  transformed_img = keep_aspect_ratio_transform(img, img_size)

  img = np.array(transformed_img)/255.0-.5
  if img.shape != (img_size, img_size, 3):
    return None
  return [img, int(ff.split(".")[0])]

class ImageNet:
  def __init__(self, img_size, load_total_imgs = 1000):
    from multiprocessing import Pool, cpu_count
    pool = Pool(cpu_count())
    file_list = sorted(os.listdir("./imagenetdata/imgs/"))
    random.shuffle(file_list)
    # for efficiency, we only load first 1000 images
    # You can pass load_total_imgs to load all images
    short_file_list = file_list[:load_total_imgs]
    r = pool.map(partial(readimg, img_size=img_size), short_file_list)
    print(short_file_list)
    print("Loaded imagenet", len(short_file_list), "of", len(file_list), "images")

    r = [x for x in r if x != None]
    test_data, test_labels = zip(*r)
    self.test_data = np.array(test_data)
    self.test_labels = np.zeros((len(test_labels), 1001))
    self.test_labels[np.arange(len(test_labels)), test_labels] = 1

    pool.close()
    pool.join()

if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  tf.app.flags.DEFINE_string(
      'model_dir', 'tmp/imagenet',
      """Path to classify_image_graph_def.pb, """
      """imagenet_synset_to_human_label_map.txt, and """
      """imagenet_2012_challenge_label_map_proto.pbtxt.""")
  tf.app.flags.DEFINE_string('image_file', '',
                             """Absolute path to image file.""")
  tf.app.flags.DEFINE_string('model_name', 'resnet_v2_101',
                             """Absolute path to image file.""")
  tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                              """Display this many predictions.""")
  tf.app.run()
else:
  # starting from TF 1.5, an parameter unkown by tf.app.flags will raise an error
  # so we cannot use tf.app.flags when loading this file as a module, because the
  # main program may define other options.
  from argparse import Namespace
  FLAGS = Namespace(model_dir="tmp/imagenet")

