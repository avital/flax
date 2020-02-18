# Flax: A neural network library for JAX designed for flexibility

**NOTE**: This is pre-release software and not yet ready for general use. If you want to use it, please get in touch with us at flax-dev@google.com.

## Background: JAX

[JAX](https://github.com/google/jax) is NumPy + autodiff + GPU/TPU

It allows for fast scientific computing and machine learning
with the normal NumPy API
(+ additional APIs for special accelerator ops when needed)

JAX has some super powerful primitives, which you can compose arbitrarily:

* Autodiff (`jax.grad`): Efficient any-order gradients w.r.t any variables
* JIT compilation (`jax.jit`): Trace any function ⟶ fused accelerator ops
* Vectorization (`jax.vmap`): Automatically batch code written for individual samples
* Parallelization (`jax.pmap`): Automatically parallelize code across multiple accelerators (including across hosts, e.g. for TPU pods)

## What is Flax?

Flax is a neural network library for
JAX that is **designed for flexibility**:
Try new forms of training by forking an example and by modifying the training
loop, not by adding features to the framework.

Flax comes with:

* Common layers (`flax.nn`): Dense, Conv, BatchNorm, Attention, ...

* Optimizers (`flax.optim`): SGD, Momentum, Adam, LARS

* ...with replication (`optimizer.replicate()`): Multi-device training with any
  optimizer

* A ResNet ImageNet example, ready to be forked for your research.

* ...more examples in the works

### Flax Modules

In its core, Flax is built around parameterised functions called Modules.
These Modules override `apply` and can be used just like normal functions.

TODO: Clarify the nuances in the statement above.

For example you can define a learned linear transformation as follows:

```py
from flax import nn
import jax.numpy as jnp

class Linear(nn.Module):
  def apply(self, x, num_features, kernel_init_fn):
    input_features = x.shape[-1]
    W = self.param('W', (input_features, num_features), kernel_init_fn)
    return jnp.dot(x, W)
```

You can also use `nn.module` as a function decorator to create a new module, as
long as you don't need access to `self` for creating parameters directly:

```py
@nn.module
def DenseLayer(x, features):
  x = flax.nn.Dense(x, features)
  x = flax.nn.relu(x)
  return x
```

## CPU-only Installation

You will need Python 3.5 or later.

Now install `flax` from Github:

```
> pip install git+https://github.com/google-research/flax.git@prerelease
```

## GPU accelerated installation

First install `jaxlib`; please follow the instructions in the
[Jax readme](https://github.com/google/jax/blob/master/README.md).
If they are not already installed, you will need to install
[CUDA](https://developer.nvidia.com/cuda-downloads) and
[CuDNN](https://developer.nvidia.com/cudnn) runtimes.

Now install `flax` from Github:

```
> pip install git+https://github.com/google-research/flax.git@prerelease
```


## Full end-to-end MNIST example

**NOTE**: See [docs/annotated_mnist.md](docs/annotated_mnist.md) for a version
with detailed annotations for each code block.

```py
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
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(x, features=64, kernel_size=(3, 3))
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=256)
    x = nn.relu(x)
    x = nn.Dense(x, features=10)
    x = nn.log_softmax(x)
    return x


def create_model(key):
  _, model = CNN.create_by_shape(key, [((1, 28, 28, 1), jnp.float32)])
  return model


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
def train_step(optimizer, batch):
  """Train for a single step."""
  def loss_fn(model):
    logits = model(batch['image'])
    loss = cross_entropy_loss(logits, batch['label'])
    return loss, logits
  optimizer, _, logits = optimizer.optimize(loss_fn)
  metrics = compute_metrics(logits, batch['label'])
  return optimizer, metrics


@jax.jit
def eval_step(model, batch):
  logits = model(batch['image'])
  return compute_metrics(logits, batch['label'])


def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = rng.permutation(len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm] for k, v in train_ds.items()}
    optimizer, metrics = train_step(optimizer, batch)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: onp.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
               epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100)

  return optimizer, epoch_metrics_np


def eval_model(model, test_ds):
  metrics = eval_step(model, test_ds)
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

  model = create_model(rng)
  optimizer = create_optimizer(model, FLAGS.learning_rate, FLAGS.momentum)

  input_rng = onp.random.RandomState(0)

  for epoch in range(1, num_epochs + 1):
    optimizer, _ = train_epoch(
        optimizer, train_ds, batch_size, epoch, input_rng)
    loss, accuracy = eval_model(optimizer.target, test_ds)
    logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                 epoch, loss, accuracy * 100)
  return optimizer


def main(_):
  train_ds, test_ds = get_datasets()
  train(train_ds, test_ds)


if __name__ == '__main__':
  app.run(main)
```

## More end-to-end examples

**NOTE**: We are still testing these examples across all supported hardware configurations.

* [ResNet on ImageNet](examples/imagenet)

* [Language Modeling on LM1b](examples/lm1b) with a Transformer architecture

## HOWTOs

HOWTOs are sample diffs showing how to change various things in your training
code.

Here are a few examples.

### Polyak averaging

This diff shows how to modify the MNIST example above to evaluate with
an exponential moving average of parameters over the course of training.

Note that no special framework support was needed.

([Full diff view](https://github.com/marcvanzee/flax/compare/prerelease..howto-polyak-averaging?diff=split))
```diff
diff --git a/examples/mnist/train.py b/examples/mnist/train.py
index 020838b..46eaab2 100644
--- a/examples/mnist/train.py
+++ b/examples/mnist/train.py
@@ -109,15 +109,19 @@ def compute_metrics(logits, labels):
 
 
 @jax.jit
-def train_step(optimizer, batch):
+def train_step(optimizer, params_ema, batch):
   """Train for a single step."""
   def loss_fn(model):
     logits = model(batch['image'])
     loss = cross_entropy_loss(logits, batch['label'])
     return loss, logits
   optimizer, _, logits = optimizer.optimize(loss_fn)
+  params_ema = jax.tree_multimap(
+      lambda p_ema, p: p_ema * 0.99 + p * 0.01,
+      params_ema, optimizer.target.params)
+
   metrics = compute_metrics(logits, batch['label'])
-  return optimizer, metrics
+  return optimizer, params_ema, metrics
 
 
 @jax.jit
@@ -126,18 +130,18 @@ def eval_step(model, batch):
   return compute_metrics(logits, batch['label'])
 
 
-def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
+def train_epoch(optimizer, params_ema, train_ds, batch_size, epoch):
   """Train for a single epoch."""
   train_ds_size = len(train_ds['image'])
   steps_per_epoch = train_ds_size // batch_size
 
-  perms = rng.permutation(len(train_ds['image']))
+  perms = onp.random.permutation(len(train_ds['image']))
   perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
   perms = perms.reshape((steps_per_epoch, batch_size))
   batch_metrics = []
   for perm in perms:
     batch = {k: v[perm] for k, v in train_ds.items()}
-    optimizer, metrics = train_step(optimizer, batch)
+    optimizer, params_ema, metrics = train_step(optimizer, params_ema, batch)
     batch_metrics.append(metrics)
 
   # compute mean of metrics across each batch in epoch.
@@ -175,15 +179,22 @@ def train(train_ds, test_ds):
 
   model = create_model(rng)
   optimizer = create_optimizer(model, FLAGS.learning_rate, FLAGS.momentum)
+  params_ema = model.params
 
   input_rng = onp.random.RandomState(0)
 
   for epoch in range(1, num_epochs + 1):
     optimizer, _ = train_epoch(
-        optimizer, train_ds, batch_size, epoch, input_rng)
+        optimizer, params_ema, train_ds, batch_size, epoch, input_rng)
     loss, accuracy = eval_model(optimizer.target, test_ds)
     logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                  epoch, loss, accuracy * 100)
+
+    model_ema = optimizer.target.replace(params=params_ema)
+    polyak_loss, polyak_accuracy = eval_model(model_ema, test_ds)
+    logging.info('polyak eval epoch: %d, loss: %.4f, accuracy: %.2f',
+                 epoch, polyak_loss, polyak_accuracy * 100)
+
   return optimizer
 
 
```

## Getting involved

**Have questions? Want to learn more? Reach out to us at flax-dev@google.com**

### Want to help?

We're happy to work together, either remotely or in Amsterdam.

In addition to general improvements
to the framework, here are some specific things that would be great to have:

#### Help build more HOWTOs

(TODO: clarify list)

#### Help build new end-to-end examples

- Semantic Segmentation
- GAN
- VAE
- ...and your proposal!

# Note

This is not an official Google product.