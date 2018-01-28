#coding:utf-8
import sys
import os
import cv2
import numpy as np
import struct
import tensorflow as tf 

def _get_output_filename(output_dir):
    #st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #return '%s/%s_%s_%s.tfrecord' % (output_dir, name, net, st)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return '%s/train_1024_test.tfrecord' % (output_dir)

def _process_image_withoutcoder(filename):

	image = cv2.imdecode(np.fromfile(filename,dtype=np.uint8),3)
	#print("image:",image.shape)
	image = image[27:123, 27:123, :]
	image_data = image.tostring()

	assert len(image.shape) == 3
	height = image.shape[0]
	width = image.shape[1]
	assert image.shape[2] == 3

	return image_data, height, width

def _float_feature(value):
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    """Wrapper for insert bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _convert_to_example_simple(LabelName, image_buffer):

	f = open(LabelName, 'rb')
	all_data = f.read()
	label_list = []

	for i in range(2, 1026):
		label_elem, = struct.unpack('f', all_data[i*4:i*4+4])
		label_list.append(label_elem)
	f.close()

	example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image_buffer),
        'label': _float_feature(label_list),
    }))
	return example


def _add_to_tfrecord(PicName, LabelName, tfrecord_writer):

	image_data, height, width = _process_image_withoutcoder(PicName)
	example = _convert_to_example_simple(LabelName, image_data)
	tfrecord_writer.write(example.SerializeToString())



def readName(pic_dir, out_dir):
	#g = os.walk("D:\\Face_150_150_3.0\\90_CDWRWS_Prisoner\\90600001")
	#print(pic_dir, out_dir)

	tf_filename = _get_output_filename(out_dir)

	g = os.walk(pic_dir)

	idx = 0

	print ('<<<<<<<<<<<<<<<  START CONVERT  >>>>>>>>>>>>>>>>>>')
	
	with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
		for path, d, filelist in g:
			for filename in filelist:
				if filename.endswith(".jpg"):

					idx += 1

					PicName = os.path.join(path, filename)
					LabelName0 = os.path.splitext(PicName)
					LabelName = LabelName0[0] + ".dat"
					#print(PicName)
					#print(LabelName)				
					#_gen_TFRecord(PicName, LabelName, tf_filename)

					_add_to_tfrecord(PicName, LabelName, tfrecord_writer)
				
					if idx%1 == 0:
						print("%d pics done!"%idx)


if __name__ == '__main__':
	pic_dir = "/raid/LSJ1/gen_tfrecord"
	out_dir = "/raid/LSJ1/gen_tfrecord"
	readName(pic_dir, out_dir)
