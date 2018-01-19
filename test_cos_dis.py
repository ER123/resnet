#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np 

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.platform import tf_logging as logging

#a = tf.constant([[1,1], [2,13],[2, 4]],tf.float32)
#b = tf.constant([[-1,1], [-3,4], [-10, 1000]], tf.float32)
a = tf.constant([[1,1,1],[1,1,1]], tf.float32)
b = tf.constant([[-1,1,0],[-1,-1,-1]], tf.float32)

c = [1,2,3]
d = [4,5,6]

class Reduction(object):
  """Types of loss reduction."""

  # Un-reduced weighted losses with the same shape as input.
  NONE = "none"

  # Scalar sum of `NONE`.
  SUM = "weighted_sum"

  # Scalar `SUM` divided by sum of weights.
  MEAN = "weighted_mean"

  # Scalar `SUM` divided by number of non-zero weights.
  SUM_BY_NONZERO_WEIGHTS = "weighted_sum_by_nonzero_weights"

  @classmethod
  def all(cls):
    return (
        cls.NONE,
        cls.SUM,
        cls.MEAN,
        cls.SUM_BY_NONZERO_WEIGHTS)

  @classmethod
  def validate(cls, key):
    if key not in cls.all():
      raise ValueError("Invalid ReductionKey %s." % key)

def _safe_div(numerator, denominator, name="value"):
  """Computes a safe divide which returns 0 if the denominator is zero.
  Note that the function contains an additional conditional check that is
  necessary for avoiding situations where the loss is zero causing NaNs to
  creep into the gradient computation.
  Args:
    numerator: An arbitrary `Tensor`.
    denominator: `Tensor` whose shape matches `numerator` and whose values are
      assumed to be non-negative.
    name: An optional name for the returned op.
  Returns:
    The element-wise value of the numerator divided by the denominator.
  """
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.div(numerator, array_ops.where(
          math_ops.equal(denominator, 0),
          array_ops.ones_like(denominator), denominator)),
      array_ops.zeros_like(numerator),name=name)

def _safe_mean(losses, num_present):
  """Computes a safe mean of the losses.
  Args:
    losses: `Tensor` whose elements contain individual loss measurements.
    num_present: The number of measurable elements in `losses`.
  Returns:
    A scalar representing the mean of `losses`. If `num_present` is zero,
      then zero is returned.
  """
  total_loss = math_ops.reduce_sum(losses)
  return _safe_div(total_loss, num_present)

def _num_present(losses, weights, per_batch=False):
  """Computes the number of elements in the loss function induced by `weights`.
  A given weights tensor induces different numbers of usable elements in the
  `losses` tensor. The `weights` tensor is broadcast across `losses` for all
  possible dimensions. For example, if `losses` is a tensor of dimension
  `[4, 5, 6, 3]` and `weights` is a tensor of shape `[4, 5]`, then `weights` is,
  in effect, tiled to match the shape of `losses`. Following this effective
  tile, the total number of present elements is the number of non-zero weights.
  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    weights: `Tensor` of shape `[]`, `[batch_size]` or
      `[batch_size, d1, ... dK]`, where K < N.
    per_batch: Whether to return the number of elements per batch or as a sum
      total.
  Returns:
    The number of present (non-zero) elements in the losses tensor. If
      `per_batch` is `True`, the value is returned as a tensor of size
      `[batch_size]`. Otherwise, a single scalar tensor is returned.
  """
  with ops.name_scope(None, "num_present", (losses, weights)) as scope:
    weights = math_ops.to_float(weights)
    present = array_ops.where(
        math_ops.equal(weights, 0.0),
        array_ops.zeros_like(weights),
        array_ops.ones_like(weights))
    present = weights_broadcast_ops.broadcast_weights(present, losses)
    if per_batch:
      return math_ops.reduce_sum(
          present, axis=math_ops.range(1, array_ops.rank(present)),
          keep_dims=True, name=scope)
    return math_ops.reduce_sum(present, name=scope)

def compute_weighted_loss(
    losses, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Computes the weighted loss.
  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `losses`, and must be broadcastable to `losses` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: the scope for the operations performed in computing the loss.
    loss_collection: the loss will be added to these collections.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss `Tensor` of the same type as `losses`. If `reduction` is
    `NONE`, this has the same shape as `losses`; otherwise, it is scalar.
  Raises:
    ValueError: If `weights` is `None` or the shape is not compatible with
      `losses`, or if the number of dimensions (rank) of either `losses` or
      `weights` is missing.
  """
  Reduction.validate(reduction)
  with ops.name_scope(scope, "weighted_loss", (losses, weights)):
    with ops.control_dependencies((
        weights_broadcast_ops.assert_broadcastable(weights, losses),)):
      losses = ops.convert_to_tensor(losses)
      input_dtype = losses.dtype
      losses = math_ops.to_float(losses)
      weights = math_ops.to_float(weights)
      weighted_losses = math_ops.multiply(losses, weights)
      if reduction == Reduction.NONE:
        loss = weighted_losses
      else:
        loss = math_ops.reduce_sum(weighted_losses)
        if reduction == Reduction.MEAN:
          loss = _safe_mean(
              loss,
              math_ops.reduce_sum(array_ops.ones_like(losses) * weights))
        elif reduction == Reduction.SUM_BY_NONZERO_WEIGHTS:
          loss = _safe_mean(loss, _num_present(losses, weights))

      # Convert the result back to the input type.
      loss = math_ops.cast(loss, input_dtype)
      util.add_loss(loss, loss_collection)  
  return loss

with tf.Session() as sess:  
  """	
	a_norm = tf.sqrt(tf.reduce_sum(tf.square(a), axis=0))

	b_norm = tf.sqrt(tf.reduce_sum(tf.square(b), axis=0))

	a_b = tf.reduce_sum(tf.multiply(a, b), axis=0)

	cosine = a_b/(a_norm * b_norm)

	#cosine1 = tf.pide(a_b, tf.multiply(a_norm, b_norm))

	a,b,c,d = sess.run([a_norm, b_norm, a_b, cosine])

	print(a,b,c,d)  """
  #a_norm_l2 = tf.nn.l2_normalize(a, [1])
  #b_norm_l2 = tf.nn.l2_normalize(b, [1])
  #radial_diffs = math_ops.multiply(a_norm_l2, b_norm_l2)
  #losses = 1 - math_ops.reduce_sum(radial_diffs, axis=(1,), keep_dims=True)
  
  #loss = compute_weighted_loss(losses)
  #loss1 = tf.losses.cosine_distance(a_norm_l2, b_norm_l2, dim=1)
  #loss_ = tf.losses.cosine_distance(a,b,dim=0)
  loss_op = tf.sqrt(tf.reduce_sum(tf.square(c,d),2))
  loss = sess.run(loss_op)
  #norm1, norm2, loss1, loss_, loss,radial_diffs, losses = sess.run([a_norm_l2, b_norm_l2, loss1, loss_, loss, radial_diffs, losses])  
  #print("norm1:",norm1)
  #print("norm2:",norm2)
  #print("loss1:",loss1)
  #print("loss_:", loss_)
  print("loss:", loss)
  #print("losses:",losses)
  #print("radial_diffs:",radial_diffs)

with tf.Session() as sess:
 
  dis = sess.run(tf.square(x3-x4))
 
  dis1 = sess.run(tf.reduce_sum(tf.square(x3-x4), 1))
 
  euclidean = sess.run(tf.sqrt(tf.reduce_sum(tf.square(x3-x4), 1)))
  euclidean = np.sum(euclidean)
  print("dis:",dis)
  print("dis1:",dis1)
  print (euclidean)
