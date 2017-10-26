import tensorflow as tf
from common.net.net import Net
from common.net.layer.basic_layer import linear_relu, linear


class BasicMLP(Net):

  def __init__(self, **kwargs):
    self.output_size = kwargs.get('output_size', 1)
    self.mode = kwargs.get('output_size', 'train')
    return

  def inference(self, data):
    with tf.variable_scope('hidden1'):
      hidden1 = linear_relu(data, 128)

    with tf.variable_scope('hidden2'):
      hidden2 = linear_relu(hidden1, 32)

    with tf.variable_scope('softmax_linear'):
      y = linear(hidden2, self.output_size)

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
