from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
from tensorflow.contrib.learn import learn_runner

from .standard_solver import StandardSolver


class EstimatorSolver(StandardSolver):

  def __init__(self, dataset, net, **kwargs):
    super(EstimatorSolver, self).__init__(dataset, net, **kwargs)
    tf.logging.set_verbosity(tf.logging.INFO)

  def model_fn(self, features, labels, mode, params):
    """Model function used in the estimator.
    Args:
      features (Tensor): Input features to the model.
      labels (Tensor): Labels tensor for training and evaluation.
      mode (ModeKeys): Specifies if training, evaluation or prediction.
      params (HParams): hyperparameters.
    Returns:
      (EstimatorSpec): Model to be run by Estimator.
    """
    if mode != tf.estimator.ModeKeys.TRAIN:
      self.net.phase = 'inference'

    layers = self.net.inference(features)

    # predict
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=layers)

    loss = self.net.loss(layers, labels)
    total_loss = tf.add_n(tf.losses.get_regularization_losses() + [loss])

    # eval
    if mode == tf.estimator.ModeKeys.EVAL:
      metrics = self.net.metric(layers, labels)
      for key, value in metrics.iteritems():
        metric, _ = value
        tf.summary.scalar(key, metric)

      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=layers,
          loss=total_loss,
          eval_metric_ops=metrics)

    tf.summary.scalar('loss', total_loss)
    tf.summary.scalar('loss_without_regularization', loss)

    # train
    train_op = self.build_optimizer(total_loss)
    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=layers, loss=total_loss, train_op=train_op)

  def experiment_fn(self, run_config, params):
    train_input_fn = partial(self.dataset.batch, 'train')
    eval_input_fn = partial(self.dataset.batch, 'eval')

    estimator = tf.estimator.Estimator(
        model_fn=self.model_fn, params=params, config=run_config)

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=params.train_steps,
        min_eval_frequency=params.min_eval_frequency,
        eval_steps=None)

    return experiment

  def train(self):
    params = tf.contrib.training.HParams(
        train_steps=self.max_steps, min_eval_frequency=self.snapshot_iter)

    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(
        model_dir=self.snapshot_dir, save_checkpoints_steps=self.snapshot_iter)

    learn_runner.run(
        experiment_fn=self.experiment_fn,
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=params)
