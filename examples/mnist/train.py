# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST example.
This script trains a simple Convolutional Neural Net on the MNIST dataset.
The data is loaded using tensorflow_datasets.
"""

from absl import app
from absl import flags
from absl import logging

from flax import nn
from flax import optim

import jax
from jax import random

import jax.numpy as jnp

import numpy as onp

import tensorflow_datasets as tfds


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.1,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_integer(
    'batch_size', default=128,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=10,
    help=('Number of training epochs.'))


def load_split(split):
  ds = tfds.load('mnist', split=split, batch_size=-1)
  data = tfds.as_numpy(ds)
  data['image'] = onp.float32(data['image']) / 255.
  return data


class CNN(nn.Module):
  """A simple CNN model."""

  def apply(self, x):
    x = nn.Conv(x, features=32, kernel_size=(3, 3))
    x = jax.nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(x, features=64, kernel_size=(3, 3))
    x = jax.nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=256, name='representation')
    x = jax.nn.relu(x)
    penultimate_layer = x
    x = nn.Dense(x, features=10)
    x = jax.nn.log_softmax(x)
    return x, penultimate_layer


class RefineCNN(nn.Module):
  """Refine CNN Model."""

  def apply(self, x):
    x = nn.Dense(x, features=64)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=10, name='final')
    x = jax.nn.log_softmax(x)
    return x


def create_model(key):
  model_def = CNN()
  _, model = model_def.create_by_shape(key, [((1, 28, 28, 1), jnp.float32)])
  refine_def = RefineCNN()
  _, refine_model = refine_def.create_by_shape(key, [((1, 256), jnp.float32)])
  return model, refine_model


def create_optimizer(model, learning_rate, beta):
  optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
  optimizer = optimizer_def.create(model)
  return optimizer


def onehot(labels, num_classes=10):
  x = (labels[..., None] == jnp.arange(num_classes)[None])
  return x.astype(jnp.float32)


def cross_entropy_loss(logits, labels):
  return -jnp.mean(jnp.sum(onehot(labels) * logits, axis=-1))


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


@jax.jit
def train_step(optimizer, base_model, batch):
  """Train for a single step."""
  def loss_fn(model):
    _, base_representation = base_model(batch['image'])
    logits = model(base_representation)
    loss = cross_entropy_loss(logits, batch['label'])
    return loss, logits
  optimizer, _, logits = optimizer.optimize(loss_fn)
  metrics = compute_metrics(logits, batch['label'])
  return optimizer, metrics


@jax.jit
def eval_step(model, base_model, batch):
  _, base_representation = base_model(batch['image'])
  logits = model(base_representation)
  return compute_metrics(logits, batch['label'])



def train_epoch(optimizer, base_model, train_ds, batch_size, epoch):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = onp.random.permutation(len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm] for k, v in train_ds.items()}
    optimizer, metrics = train_step(optimizer, base_model, batch)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: onp.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
               epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100)

  return optimizer

def eval_model(model, base_model, test_ds):
  metrics = eval_step(model, base_model, test_ds)
  metrics = jax.device_get(metrics)
  summary = jax.tree_map(lambda x: x.item(), metrics)
  return summary['loss'], summary['accuracy']


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  train_ds = load_split(tfds.Split.TRAIN)
  test_ds = load_split(tfds.Split.TEST)
  return train_ds, test_ds


def train(train_ds, test_ds):
  """Train MNIST to completion."""
  rng = random.PRNGKey(0)

  batch_size = FLAGS.batch_size
  num_epochs = FLAGS.num_epochs

  base_model, refine_model = create_model(rng)
  optimizer = create_optimizer(refine_model,
                               FLAGS.learning_rate, FLAGS.momentum)

  print('Base model weights:')
  print(base_model.params['representation']['kernel'][0:5, 0:5])
  print('Refine model weights:')
  print(refine_model.params['final']['kernel'][0:5, 0:5])

  for epoch in range(1, num_epochs + 1):
    optimizer = train_epoch(optimizer, base_model, train_ds, batch_size, epoch)
    loss, accuracy = eval_model(optimizer.target, base_model, test_ds)
    logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                 epoch, loss, accuracy * 100)

  refine_model = optimizer.target
  print('Base model weights should be the same:')
  print(base_model.params['representation']['kernel'][0:5, 0:5])
  print('Refine model weights should be changed:')
  print(refine_model.params['final']['kernel'][0:5, 0:5])
  print('-----')


def main(_):
  train_ds, test_ds = get_datasets()
  train(train_ds, test_ds)


if __name__ == '__main__':
  app.run(main)