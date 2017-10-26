import tensorflow as tf
from common.net.net import Net
from common.net.layer.basic_layer import *


class BasicCNN(Net):

  def __init__(self, **kwargs):
    self.output_size = kwargs.get('output_size', 1)
    return

  def inference(self, data):

    with tf.variable_scope('conv1'):
      conv1 = conv_relu(data, kernel_size=3, width=32)
      pool1 = pool(conv1, size=2)

    with tf.variable_scope('conv2'):
      conv2 = conv_relu(pool1, kernel_size=2, width=64)
      pool2 = pool(conv2, size=2)

    with tf.variable_scope('conv3'):
      conv3 = conv_relu(pool2, kernel_size=2, width=128)
      pool3 = pool(conv3, size=2)

    # Flatten convolutional layers output
    shape = pool3.get_shape().as_list()
    flattened = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])

    # Fully connected layers
    with tf.variable_scope('fc4'):
      fc4 = linear_relu(flattened, output_size=100)

    with tf.variable_scope('fc5'):
      fc5 = linear_relu(fc4, output_size=100)

    with tf.variable_scope('out'):
      prediction = linear(fc5, output_size=self.output_size)

    return {"predictions": prediction, 'data': data}

  def loss(self, layers, labels):
    predictions = layers['predictions']
    with tf.variable_scope('losses'):
      loss = tf.reduce_mean(tf.square(predictions - labels), name='mse')
    return loss

  def metric(self, layers, labels):
    predictions = layers['predictions']
    with tf.variable_scope('metrics'):
      metrics = {
        "mse": tf.metrics.mean_squared_error(
          labels=labels, predictions=predictions)}
    return metrics
