import tensorflow as tf


def linear(x, output_size, wd=0):

  input_size = x.get_shape()[1].value
  weight = tf.get_variable(
      name='weight',
      shape=[input_size, output_size],
      initializer=tf.contrib.layers.xavier_initializer())
  bias = tf.get_variable(
      'bias', shape=[output_size], initializer=tf.constant_initializer(0.0))
  out = tf.matmul(x, weight) + bias

  if wd != 0:
    weight_decay = tf.multiply(tf.nn.l2_loss(weight), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

  return out


def linear_relu(x, output_size, wd=0):
  return tf.nn.relu(
      linear(x, output_size, wd), name=tf.get_default_graph().get_name_scope())


def conv_relu(x, kernel_size, width, wd=0):

  input_size = x.get_shape()[3]
  weight = tf.get_variable(
      name='weight',
      shape=[kernel_size, kernel_size, input_size, width],
      initializer=tf.contrib.layers.xavier_initializer())
  bias = tf.get_variable(
      'bias', shape=[width], initializer=tf.constant_initializer(0.0))
  conv = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')

  if wd != 0:
    weight_decay = tf.multiply(tf.nn.l2_loss(weight), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

  out = tf.nn.relu(conv + bias, name=tf.get_default_graph().get_name_scope())
  return out


def pool(x, size):
  return tf.nn.max_pool(
      x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')
