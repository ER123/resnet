#coding:utf-8
import tensorflow as tf 
import numpy as np 
import os
import sys
from datetime import datetime
from PIL import Image
from resnet_structure import LCNN4, test_net, O_net
import numpy.random as npr
import random
import cv2

from easydict import EasyDict as edict
#from resnet import test_net, LCNN4

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = edict()
config.BATCH_SIZE = 16
config.LR_EPOCH = 100

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

f1 = open("E:/ExtractionNet/output_onet_norm0.txt", 'w')
f2 = open("E:/ExtractionNet/labels_onet_norm0.txt", 'w')

def load_data(tfrecord_file, batch_size):
	#every epoch shuffle
	filename_queue = tf.train.string_input_producer([tfrecord_file], shuffle=True)
	reader = tf.TFRecordReader()
	_, serialized_exmple = reader.read(filename_queue)

	image_features = tf.parse_single_example(
		serialized_exmple,
		features={
			"image":tf.FixedLenFeature([], tf.string),
			#"label":tf.FixedLenFeature([256], tf.float32),
			"label":tf.FixedLenFeature([1024], tf.float32),
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


def train_model(base_lr, loss, data_num):
	lr_factor = 0.1
	global_step = tf.Variable(0, trainable=False)

	boundaries = [int(10*epoch*data_num/config.BATCH_SIZE) for epoch in range(1,6)]
	print("boundaries:",boundaries)
	lr_values = [base_lr*(lr_factor ** x) for x in range(6)]
	print("lr_values:",lr_values)
	lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
	optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
	#optimizer = tf.train.GradientDescentOptimizer(lr_op)
	train_op = optimizer.minimize(loss, global_step)

	return train_op, lr_op


def cos_loss(output, labels):
	output = tf.nn.l2_normalize(output, 1)
	print("output:",output)
	labels = tf.nn.l2_normalize(labels, 1)
	print("labels:",labels)
	loss = tf.losses.cosine_distance(output, labels, dim=0)
	return loss

def dis_loss(output, labels):
	loss = tf.sqrt(tf.reduce_sum(tf.square(output-labels), 1))
	print("dis_loss loss:",loss)
	loss = (tf.reduce_sum(loss))/config.BATCH_SIZE
	print("sum(loss):",loss)
	return loss

def norm_loss(output, label):
    output_norm = tf.sqrt(tf.reduce_sum(tf.square(output), axis=1))
    label_norm = tf.sqrt(tf.reduce_sum(tf.square(label), axis=1))
    
    label_output = tf.reduce_sum(tf.multiply(output, label), axis=1)
    loss = 1 - tf.reduce_mean(  label_output / tf.multiply( label_norm, output_norm) )
    
    return loss

def train(net, tfrecord_file, image_size, base_lr, num, end_epoch):

	image_batch, label_batch = load_data(tfrecord_file, config.BATCH_SIZE)

	input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 96, 96, 3], name="input_image")
	label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 1024], name="label")
	#label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name="label")
	is_training = tf.placeholder('bool',[], name='is_training')

	#output_op = test_net(input_image, is_training=is_training)
	#output_op = LCNN4(input_image)
	output_op = O_net(input_image)

	#loss_op = cos_loss(output_op, label)
	loss_op = norm_loss(output_op, label)
	#loss_op = dis_loss(output_op, label)

	#train_op, lr_op = train_model(base_lr, loss_op, num)

	global_ = tf.Variable(tf.constant(0))
	lr_op = tf.train.exponential_decay(0.06, global_, 1, 0.8, staircase=True)

	regularization_losses = tf.add_n(tf.losses.get_regularization_losses())
	train_op = tf.train.AdamOptimizer(lr_op).minimize( (loss_op + 0.001*regularization_losses) )

	init = tf.global_variables_initializer()

	config1 = tf.ConfigProto()
	#config1.gpu_options.per_process_gpu_memory_fraction = 0.4
	config1.gpu_options.allow_growth = True
	sess = tf.Session(config=config1)
	
	#sess = tf.Session()

	saver = tf.train.Saver(max_to_keep=0)

	sess.run(init)

	tf.summary.scalar("loss", loss_op)
	summary_op = tf.summary.merge_all()

	logs_dir = "E:/ExtractionNet/logs/onet1024_1"
	if os.path.exists(logs_dir) == False:
		os.mkdir(logs_dir)

	writer = tf.summary.FileWriter(logs_dir, sess.graph)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	MAX_STEP = 	int(num/config.BATCH_SIZE + 1)*end_epoch

	i = 0
	epoch = 0
	sess.graph.finalize()
	try:
		for step in range(MAX_STEP):
			i += 1
			if coord.should_stop():
				break
			image_batch_array, label_batch_array = sess.run([image_batch, label_batch])

			#print("label_batch_array:",label_batch_array[0])
			_, _, summary = sess.run([train_op, lr_op, summary_op], feed_dict={input_image: image_batch_array, label:label_batch_array, global_:epoch})
			#sess.run([train_op, lr_op], feed_dict={input_image: image_batch_array, label:label_batch_array})
			if (step+1) % 10 == 0:
				output, loss, lr = sess.run([output_op, loss_op, lr_op], feed_dict={input_image: image_batch_array, label:label_batch_array, global_:epoch})
				
				if (step+1) %100 == 0:
					output, loss = sess.run([output_op, loss_op], feed_dict={input_image: image_batch_array, label:label_batch_array, global_:epoch})
					print("output:",output[0])					
					for out, label_ in zip(output, label_batch_array):
						for res1, res2 in zip(out, label_):
							f1.write(str(res1)+" ")
							f2.write(str(res2) + " ")
						f1.write("\n")
						f2.write("\n")
				print("%s : step: %d, loss: %2f, lr: %6f"%(datetime.now(), step+1, loss, lr))
			if i*config.BATCH_SIZE >= num:
				epoch = epoch + 1
				i = 0
				print ("---------------------------------->")
				saver.save(sess, "E:/ExtractionNet/ckpt/onet/onet1024_1", global_step=epoch)
				writer.add_summary(summary,global_step=step)
	except tf.errors.OutOfRangeError:
		print("完成！！！")
	finally:
		coord.request_stop()
		writer.close()
	coord.join(threads)
	sess.close()

if __name__ == '__main__':
	tfrecord_file = "E:/gen_TFRecord/train_1024_test.tfrecord"
	image_size = 96
	base_lr = 0.01
	num = 126
	end_epoch = config.LR_EPOCH
	net = test_net
	train( net, tfrecord_file, image_size, base_lr, num, end_epoch)
