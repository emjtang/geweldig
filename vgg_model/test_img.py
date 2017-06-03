import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', default='../data/images_top10/train')
parser.add_argument('--model_path', default='vgg_16.ckpt', type=str)

def importLabels():
  return [i for i in np.arange(11)]

def importImg(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
  image = tf.cast(image_decoded, tf.float32)

  smallest_side = 225.0
  height, width = tf.shape(image)[0], tf.shape(image)[1]
  height = tf.to_float(height)
  width = tf.to_float(width)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)

  resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
  return resized_image, label

def main(args):
  graph = tf.Graph()
  image, label = importImg("test.png")

  vgg = tf.contrib.slim.nets.vgg
  with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
      logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
                             dropout_keep_prob=args.dropout_keep_prob)

  # Specify where the model checkpoint is (pretrained weights).
  model_path = args.model_path

  variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=[])
  init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
  with graph.as_default():
    with tf.Session(graph=graph) as sess:
      init_fn(sess)
      # saver = tf.train.Saver()
      # ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
      predictions = sess.run(label, feed_dict={x: image})
      print(predictions)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)