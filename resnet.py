#coding:utf-8
#import skimage.io  # bug. need to import this before tensorflow
#import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from config import Config

import datetime
import numpy as np
import os
import time

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

tf.app.flags.DEFINE_integer('input_size', 96, "input image size")


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


def conv_layers(x, labels, is_training, use_bias=False):
	c = Config()
	c['is_training'] = tf.convert_to_tensor(is_training, dtype='bool', name='is_training')
	#c['is_training'] = True
	c['ksize'] = 3
	c['stride'] = 1
	c['use_bias'] = use_bias
	c['fc_units_out'] = 256
	c['conv_filters_out'] = 64
	c['fc_units_out'] = 256

	print("input.shape:",x.get_shape())

	with tf.variable_scope("conv1"):
		c['conv_filters_out'] = 64
		c['strides'] = 2
		x = conv(x, c)
		x = bn(x, c)
		x = activation(x)
	print("conv1.shape:",x.get_shape())

	with tf.variable_scope("conv2"):
		c['conv_filters_out'] = 64
		x = conv(x, c)
		x = bn(x, c)
		x = activation(x)
	print("conv2.shape:",x.get_shape())

	with tf.variable_scope("conv3"):
		c['conv_filters_out'] = 128
		c['strides'] = 2
		x = conv(x, c)
		x = bn(x, c)
		x = activation(x)
	print("conv3.shape:",x.get_shape())

	with tf.variable_scope("conv4"):
		c['conv_filters_out'] = 128
		x = conv(x, c)
		x = bn(x, c)
		x = activation(x)
	print("conv4.shape:",x.get_shape())

	with tf.variable_scope("conv5"):
		c['conv_filters_out'] = 256
		c['strides'] = 2
		x = conv(x, c)
		x = bn(x, c)
		x = activation(x)
	print("conv5.shape:",x.get_shape())

	with tf.variable_scope("conv6"):
		c['conv_filters_out'] = 256
		x = conv(x, c)
		x = bn(x, c)
		x = activation(x)
	print("conv6.shape:",x.get_shape())
	with tf.variable_scope("conv7"):
		c['conv_filters_out'] = 512
		x = conv(x, c)
		x = bn(x, c)
		x = activation(x)
	print("conv7.shape:",x.get_shape())
	with tf.variable_scope("conv8"):
		c['conv_filters_out'] = 512
		x = conv(x, c)
		x = bn(x, c)
		x = activation(x)
	print("conv8.shape:",x.get_shape())

	with tf.variable_scope("conv9"):
		c['conv_filters_out'] = 512
		x = conv(x, c)
		x = bn(x, c)
		x = activation(x)
	print("conv9.shape:",x.get_shape())

	x = tf.reduce_mean(x, reduction_indices=[1,2], name="avg_pool")

	with tf.variable_scope("fc"):
		x = fc(x, c)
	print("fc.shape:",x.get_shape())

	x = cos_loss(x, labels)

	return x

def test_net(x, labels, is_training, use_bias=True):
	c = Config()
	c['is_training'] = tf.convert_to_tensor(is_training, dtype='bool',name='is_training')
	c['ksize'] = 3
	c['stride'] = 1
	c['use_bias'] = use_bias
	c['fc_units_out'] = 256
	c['stack_stride'] = 1
	c['bottleneck'] = False
	c['num_blocks'] = 1

	print("input.shape:",x.get_shape())

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
		c['block_filters_internal'] = 512
		x = stack(x, c)
	print("scale5.shape:",x.get_shape())

	x = tf.reduce_mean(x, reduction_indices=[1,2], name="avg_pool")

	with tf.variable_scope('fc'):
		x = fc(x, c)
	print("fc.shape:",x.get_shape())

	#x = cos_loss(x, labels)

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


def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        #print (s)
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    #m = 4 if c['bottleneck'] else 1
    filters_out = c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    with tf.variable_scope('A'):
        c['stride'] = c['block_stride']
        assert c['ksize'] == 3
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)

    with tf.variable_scope('B'):
        c['conv_filters_out'] = filters_out
        assert c['ksize'] == 3
        assert c['stride'] == 1
        x = conv(x, c)
        x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)

    return activation(x + shortcut)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape, initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer, trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x


def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer, weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=shape, dtype='float', initializer=initializer, weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')

def cos_loss(logits, labels):
	#print("logits:",logits)
	#print("labels:",labels)
	#sess = tf.Session()
	#print("labels:",sess.run(labels[0]))
	#logits = tf.nn.l2_normalize(logits, 1)
	#labels = tf.nn.l2_normalize(labels, 1)
	#loss_dis = tf.sqrt(tf.reduce_sum(tf.square(logits-labels), 1))
	#print("loss_dis:::::",loss_dis)
	#print("logits:",logits)
	#print("labels:",labels)
	loss = tf.losses.cosine_distance(logits, labels, dim=0)
	print("loss:", loss)
	return loss

def Euclidean_loss(predict, labels):
	loss = tf.sqrt(tf.reduce_sum(tf.square(logits-labels), 1))
	return loss
