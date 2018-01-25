#coding:utf-8
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.contrib import slim
from config import Config
from resnet import stack, block, bn, fc, _get_variable, conv, _max_pool, prelu
import datetime
import numpy as np
import os
import time

activation = tf.nn.relu

def O_net(x):
	with slim.arg_scope([slim.conv2d],
						activation_fn=prelu,
						weights_initializer=slim.xavier_initializer(),
						biases_initializer=tf.zeros_initializer(),
						weights_regularizer=slim.l2_regularizer(0.001),
						padding="SAME"):
		print("input.shape:",x.get_shape())
		net = slim.conv2d(x, num_outputs=32, kernel_size=[3, 3], stride=1, scope="conv1")
		print("conv1.shape:",net.get_shape())
		net = tf.nn.lrn( net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
		print("norm1.shape:",net.get_shape())
		net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
		print("pool1.shape:",net.get_shape())
		net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
		#net = BatchNorm_1( net, is_training = True )
		net = tf.nn.lrn( net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
		print("norm2.shape:",net.get_shape())
		net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
		print("pool2.shape:",net.get_shape())
		net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
		print("conv3.shape:",net.get_shape())
		#net = BatchNorm_1( net, is_training = True )
		net = tf.nn.lrn( net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
		print("norm3.shape:",net.get_shape())
		net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool3", padding='SAME')
		print("pool3.shape:",net.get_shape())
		net = slim.conv2d(net,num_outputs=96,kernel_size=[3,3],stride=1,scope="conv4")
		print("conv4.shape:",net.get_shape())
		#net = BatchNorm_1( net, is_training = True )
		net = slim.conv2d(net,num_outputs=96,kernel_size=[3,3],stride=1,scope="conv5")
		print("conv5.shape:",net.get_shape())
		net = slim.conv2d(net,num_outputs=96,kernel_size=[3,3],stride=1,scope="conv6")
		print("conv6.shape:",net.get_shape())
		#net = BatchNorm_1( net, is_training = True )
		net = tf.nn.lrn( net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm4')
		print("norm4.shape:",net.get_shape())
		net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool4", padding='SAME')
		print("pool4.shape:",net.get_shape())
		net = slim.conv2d(net,num_outputs=128,kernel_size=[3,3],stride=1,scope="conv7")
		print("conv7.shape:",net.get_shape())
		#net = BatchNorm_1( net, is_training = True )
		net = slim.conv2d(net,num_outputs=128,kernel_size=[3,3],stride=1,scope="conv8")
		print("conv7.shape:",net.get_shape())
		#net = BatchNorm_1( net, is_training = True )
		net = tf.nn.lrn( net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm5')
		print("norm5.shape:",net.get_shape())
		#net = _batch_norm( 'bn1', net )              
		fc_flatten = slim.flatten(net)
		print("flatten.shape:",fc_flatten.get_shape())
		fc1 = slim.fully_connected(fc_flatten, num_outputs=2048,scope="fc1", activation_fn=prelu)
		print("fc1.shape:",fc1.get_shape())
		#fc1 = slim.dropout(fc1, 0.85, scope='dropout7')
		#fc1 = tf.tanh(fc1, name='tah1')
		#fc1 = tf.nn.lrn( fc1, 2, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
		fc2 = slim.fully_connected(fc1, num_outputs=1024,scope="fc2", activation_fn=prelu)
		print("fc2.shape:",fc2.get_shape())
		#fc2 = slim.dropout(fc2, 0.9, scope='dropout7')
		#fc2 = tf.nn.lrn( fc2, 2, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
		fc3 = slim.fully_connected(fc2, num_outputs=256,scope="fc3", activation_fn=None)
		print("fc3.shape:",fc3.get_shape())

		return fc3



def LCNN4(x):

    c = Config()
    c['ksize'] = 3
    c['stride'] = 1
    c['fc_units_out'] = 256
    c['conv_filters_out'] = 48

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

    with tf.variable_scope("flatten"):
        x = slim.flatten(x)
    print("flatten.shape:",x.get_shape())

    with tf.variable_scope("fc1"):
        c['fc_units_out'] = 512
        x = fc(x, c)
    print("fc1.shape:",x.get_shape())

    x = tf.nn.dropout(x, keep_prob=0.7)

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