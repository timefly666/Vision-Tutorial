import tensorflow as tf
from common.net.net import Net


class Softmax(Net):

  def __init__(self, **kwargs):
    self.output_size = kwargs.get('output_size', 1)
    return

  def inference(self, data):
    feature_size = data.get_shape()[1].value

    with tf.name_scope('weights'):
      W = tf.Variable(tf.zeros([feature_size, self.output_size]))
    with tf.name_scope('biases'):
      b = tf.Variable(tf.zeros([self.output_size]), name='bias')
    with tf.name_scope('y'):
      y = tf.matmul(data, W) + b
    with tf.name_scope('predictions'):
      predictions = tf.nn.softmax(y)

    return {'logits': y, 'predictions': predictions}

  def loss(self, layers, labels):
    logits = layers['logits']

    with tf.variable_scope('losses'):
      loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return loss

  def metric(self, layers, labels):
    predictions = layers['predictions']
    with tf.variable_scope('metrics'):
      metrics = {
          "accuracy":
              tf.metrics.accuracy(
                  tf.argmax(labels, 1), predictions=tf.argmax(predictions, 1))
      }
    return metrics
