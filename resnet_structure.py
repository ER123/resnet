#coding:utf-8
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.contrib import slim
from config import Config
from resnet import *
import datetime
import numpy as np
import os
import time

activation = tf.nn.relu

def LCNN4(x):

    c = Config()
    c['ksize'] = 3
    c['stride'] = 1
    c['fc_units_out'] = 256
    c['conv_filters_out'] = 64

    print("input.shape:", x.get_shape())

    with tf.variable_scope("conv1"):
        c['conv_filters_out'] = 48
        c['ksize'] = 5
        x = conv(x,c)
        x = activation(x)
    print("conv1.shape:",x.get_shape())

    with tf.variable_scope("max_pool1"):
        x = _max_pool(x, ksize=2, stride=2)
    print("max_pool1.shape:",x.get_shape())

    with tf.variable_scope("conv2"):
        c['ksize'] = 1
        c['conv_filters_out'] = 48
        x = conv(x, c)
        x = activation(x)
    print("conv2.shape:",x.get_shape())

    with tf.variable_scope("conv3"):
        c['ksize'] = 3
        c['conv_filters_out'] = 96
        x = conv(x, c)
        x = activation(x)
    print("conv3.shape:",x.get_shape())

    with tf.variable_scope("max_pool2"):
        x = _max_pool(x, ksize=2, stride=2)
    print("max_pool2.shape:",x.get_shape())

    with tf.variable_scope("conv4"):
        c['ksize'] = 1
        c['conv_filters_out'] = 96
        x = conv(x, c)
        x = activation(x)
    print("conv4.shape:",x.get_shape())

    with tf.variable_scope("conv5"):
        c['ksize'] = 3
        c['conv_filters_out'] = 128
        x = conv(x, c)
        x = activation(x)
    print("conv5.shape:",x.get_shape())

    with tf.variable_scope("conv6"):
        c['ksize'] = 1
        c['conv_filters_out'] = 128
        x = conv(x, c)
        x = activation(x)
    print("conv6.shape:",x.get_shape())

    with tf.variable_scope("conv7"):
        c['ksize'] = 3
        c['conv_filters_out'] = 128
        x = conv(x, c)
        x = activation(x)
    print("conv7.shape:",x.get_shape())

    with tf.variable_scope("max_pool3"):
        x = _max_pool(x, ksize=2, stride=2)
    print("max_pool3.shape:",x.get_shape())

    x = slim.flatten(x)

    with tf.variable_scope("fc1"):
        c['fc_units_out'] = 512
        x = fc(x, c)
    print("fc1.shape:",x.get_shape())

    x = tf.nn.dropout(x, keep_prob=0.5)

    with tf.variable_scope("fc2"):
    	c['fc_units_out'] = 256
    	x = fc(x, c)
    print("fc2.shape:",x.get_shape())

    return x

def test_net(x, is_training, use_bias=True):
	c = Config()
	c['is_training'] = tf.convert_to_tensor(is_training, dtype='bool',name='is_training')
	c['ksize'] = 3
	c['stride'] = 1
	c['use_bias'] = use_bias
	c['fc_units_out'] = 256
	c['stack_stride'] = 1
	c['bottleneck'] = False
	c['num_blocks'] = 1

	print(" input.shape:",x.get_shape())

	with tf.variable_scope('scale1'):
		c['conv_filters_out'] = 64
		c['ksize'] = 6
		c['stride'] = 2
		x = conv(x, c)
		x = bn(x, c)
		x = activation(x)
	print("sacle1.shape:",x.get_shape())

	with tf.variable_scope('scale2'):
		x = _max_pool(x, ksize=3, stride=2)
		c['num_blocks'] = 1
		c['stack_stride'] = 1
		c['block_filters_internal'] = 64
		x = stack(x, c)
	print("scale2.shape:",x.get_shape())

	with tf.variable_scope('scale3'):
		c['num_blocks'] = 1
		c['block_filters_internal'] = 128
		assert c['stack_stride'] == 1
		x = stack(x, c)
	print("scale3.shape:",x.get_shape())

	with tf.variable_scope('scale4'):
		c['num_blocks'] = 1
		c['block_filters_internal'] = 256
		x = stack(x, c)
	print("scale4.shape:",x.get_shape())

	with tf.variable_scope('scale5'):
		c['num_blocks'] = 1
		c['block_filters_internal'] = 256
		x = stack(x, c)
	print("scale5.shape:",x.get_shape())

	#x = tf.reduce_mean(x, reduction_indices=[1,2], name="avg_pool")
	x = _max_pool(x, ksize=3, stride=2)
	print("max_pool.shape:",x.get_shape())
	x = slim.flatten(x)
	print("flatten.shape:",x.get_shape())

	with tf.variable_scope('fc'):
		x = fc(x, c)
	print("fc.shape:",x.get_shape())

	return x


def inference_small(x,
                    is_training,
                    num_blocks=3, # 6n+2 total weight layers will be used.
                    use_bias=False, # defaults to using batch norm
                    num_classes=256):
    c = Config()
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['num_classes'] = num_classes
    inference_small_config(x, c)

def inference_small_config(x, c):
    c['bottleneck'] = False
    c['ksize'] = 3
    c['stride'] = 1
    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 16
        c['block_filters_internal'] = 16
        c['stack_stride'] = 1
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)
        x = stack(x, c)

    with tf.variable_scope('scale2'):
        c['block_filters_internal'] = 32
        c['stack_stride'] = 2
        x = stack(x, c)

    with tf.variable_scope('scale3'):
        c['block_filters_internal'] = 64
        c['stack_stride'] = 2
        x = stack(x, c)

    # post-net
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

    if c['num_classes'] != None:
        with tf.variable_scope('fc'):
            x = fc(x, c)

    return x