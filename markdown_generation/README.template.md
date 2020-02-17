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

@insert_code examples/mnist/train.py

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

@insert_branch_diff howto-polyak-averaging

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
