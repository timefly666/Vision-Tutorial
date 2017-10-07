import tensorflow as tf


def linear(x, size, wd=0):

  weights = tf.get_variable(
      name='weights',
      shape=[x.get_shape()[1], size],
      initializer=tf.contrib.layers.xavier_initializer())
  biases = tf.get_variable(
      'biases', shape=[size], initializer=tf.constant_initializer(0.0))
  out = tf.matmul(x, weights) + biases

  if wd != 0:
    # tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer, wd)
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

  return out


def linear_relu(x, size, wd=0):
  return tf.nn.relu(linear(x, size, wd), name=tf.get_default_graph().get_name_scope())


def conv_relu(x, kernel_size, width, wd=0):
  weights = tf.get_variable(
      'weights',
      shape=[kernel_size, kernel_size,
             x.get_shape()[3], width],
      initializer=tf.contrib.layers.xavier_initializer())
  biases = tf.get_variable(
      'biases', shape=[width], initializer=tf.constant_initializer(0.0))
  conv = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

  if wd != 0:
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

  out = tf.nn.relu(conv + biases, name=tf.get_default_graph().get_name_scope())
  return out


def pool(x, size):
  return tf.nn.max_pool(
      x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')
