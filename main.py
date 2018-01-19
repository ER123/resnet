#coding:utf-8
import tensorflow as tf 
import numpy as np  
import os
from time import time
from PIL import Image

from resnet import *
from train import *

image_size = 96

def load_data(tfrecord_file, batch_size):
	#every epoch shuffle
	filename_queue = tf.train.string_input_producer([tfrecord_file], shuffle=True)
	reader = tf.TFRecordReader()
	_, serialized_exmple = reader.read(filename_queue)

	image_features = tf.parse_single_example(
		serialized_exmple,
		features={
			"image":tf.FixedLenFeature([], tf.string),
			"label":tf.FixedLenFeature([256], tf.float32),
		}
	)
	#print("image_features:",image_features)
	image = tf.decode_raw(image_features["image"], tf.uint8)
	image = tf.reshape(image, [image_size, image_size, 3])
	image = (tf.cast(image, tf.float32)-127.5)/128

	label = tf.cast(image_features["label"], tf.float32)
	#print("label:",label)
	#print("image:",image)
	#for i in range(10):
	#	img = Image.fromarray(image, 'RGB')
	#	img.write("G:\\ready_for_tfrecord\\" + str(i) + ".jpg")

	image, label = tf.train.batch(
		[image, label],
		batch_size=batch_size,
		num_threads=2,
		capacity=1*batch_size
	)
	#label = tf.reshape(label, [batch_size])
	#print("label:",label)
	#print("image:",image)


	return image, label

def main(_):
	tfrecord_file = "G:\\ready_for_tfrecord\\train.tfrecord_shuffle"
	batch_size = 64
	images, labels = load_data(tfrecord_file, batch_size)
	is_training = tf.placeholder('bool',[], name='is_training')

	logits = test_net(images, labels, is_training=is_training)

	#logits = inference_small(images, is_training=is_training, num_classes=256)

	#logits = conv_layers(images, is_training)
	#sess =  tf.Session()
	#print("logits:",sess.run(logits))

	train(is_training, logits, images, labels)
	
if __name__ == '__main__':
	tf.app.run()