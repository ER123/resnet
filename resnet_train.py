#coding:utf-8
import tensorflow as tf 
import numpy as np 
import os
import sys
from datetime import datetime
from PIL import Image

import numpy.random as npr
import random
import cv2

from easydict import EasyDict as edict

from resnet import test_net

config = edict()
config.BATCH_SIZE = 64
config.LR_EPOCH = [20, 20]

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
	#image = (tf.cast(image, tf.float32)-127.5)/128

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


def train_model(base_lr, loss, data_num):
	lr_factor = 0.1
	global_step = tf.Variable(0, trainable=False)

	boundaries = [int(epoch*data_num/config.BATCH_SIZE) for epoch in range(10)]

	lr_values = [base_lr*(lr_factor ** x) for x in range(0, 4)]

	lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
	optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
	train_op = optimizer.minimize(loss, global_step)

	return train_op, lr_op

def cos_loss(logits, labels):
	#print("logits:",logits)
	#print("labels:",labels)
	#sess = tf.Session()
	#print("labels:",sess.run(labels[0]))
	logits = tf.nn.l2_normalize(logits, 1)
	labels = tf.nn.l2_normalize(labels, 1)
	#loss_dis = tf.sqrt(tf.reduce_sum(tf.square(logits-labels), 1))
	#print("loss_dis:::::",loss_dis)
	print("logits:",logits)
	print("labels:",labels)
	loss = tf.losses.cosine_distance(logits, labels, dim=0)
	print("loss:", loss)
	return loss

def dis_loss(logits, labels):
	loss = tf.sqrt(tf.reduce_sum(tf.square(logits-labels), 1))
	return loss

def norm_loss(output, labels):
	output_norm = tf.sqrt(tf.reduce_sum(tf.square(output), axis=1))
    	label_norm = tf.sqrt(tf.reduce_sum(tf.square(label), axis=1))
    
   	label_output = tf.reduce_sum(tf.multiply(output, label), axis=1)
    	loss = 1 - tf.reduce_mean(  label_output / tf.multiply( label_norm, output_norm) )
    
    	return loss

def train(net, tfrecord_file, image_size, base_lr, num, end_epoch):

	image_batch, label_batch = load_data(tfrecord_file, config.BATCH_SIZE)

	input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 96, 96, 3], name="input_image")
	label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 256], name="label")
	is_training = tf.placeholder('bool',[], name='is_training')

	#output_op = test_net(input_image, label, is_training=is_training)
	output_op = LCNN4(input_image)
	#loss_op = cos_loss(output_op, label)
 	
	train_op, lr_op = train_model(base_lr, loss_op, num)

	init = tf.global_variables_initializer()

	config1 = tf.ConfigProto()
	config1.gpu_options.allow_growth = True

	sess = tf.Session(config=config1)

	saver = tf.train.Saver(max_to_keep=0)

	sess.run(init)

	tf.summary.scalar("loss", loss_op)
	summary_op = tf.summary.merge_all()

	logs_dir = "C:\\Users\\salaslyrin\\Desktop\\ResNet\\_MYRESNET/logs"
	if os.path.exists(logs_dir) == False:
		os.mkdir(logs_dir)

	writer = tf.summary.FileWriter(logs_dir, sess.graph)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	#for i in range(1):
	#	example, l = sess.run([image_batch,label_batch])#在会话中取出image和label
	#	for j in range(10):
	#		img=Image.fromarray(example[j], 'RGB')#这里Image是之前提到的
	#		img.save("C:\\Users\\salaslyrin\\Desktop\\ResNet\\_MYRESNET/"+str(j)+".jpg")#存下图片
	#		#img.show()
	#		print("l:",l[j])

	i=0
	MAX_STEP = 	int(num/config.BATCH_SIZE + 1)*end_epoch

	epoch = 0
	sess.graph.finalize()
	try:
		for step in range(MAX_STEP):
			i += 1
			if coord.should_stop():
				break
			image_batch_array, label_batch_array = sess.run([image_batch, label_batch])

			#print("label_batch_array:",label_batch_array[0])
			_, _, summary = sess.run([train_op, lr_op, summary_op], feed_dict={input_image: image_batch_array, label:label_batch_array})
			#sess.run([train_op, lr_op], feed_dict={input_image: image_batch_array, label:label_batch_array})
			if (step+1) % 10 == 0:
				loss, lr, output = sess.run([loss_op, lr_op, output_op], feed_dict={input_image: image_batch_array, label:label_batch_array})
				print("output:",output[0])
				print("label_batch_array:",label_batch_array[0])
				print("lllllllllllllllllllloss:",loss)
				#print("loss[0]:",loss[0])
				#print("loss:",loss.get_shape())

				print("%s : step: %d, loss: %2f, lr: %6f"%(datetime.now(), step+1, loss, lr))
			if i*config.BATCH_SIZE >= num*2:
				epoch = epoch+1
				i=0
				saver.save(sess, "C:\\Users\\salaslyrin\\Desktop\\ResNet\\_MYRESNET\\resnet\\resnet", global_step=epoch*2)
				writer.add_summary(summary,global_step=step)
	except tf.errors.OutOfRangeError:
		print("完成！！！")
	finally:
		coord.request_stop()
		writer.close()
	coord.join(
