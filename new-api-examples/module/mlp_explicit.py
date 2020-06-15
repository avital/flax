import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.module import Module, autonames
from dataclasses import dataclass


@dataclass
class DenseExplicit(Module):
  in_features: int
  out_features: int
  with_bias: bool = True
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros
  name: str = None

  def ready(self):
    self.kernel = self.param('kernel', self.kernel_init, (self.in_features, self.out_features))

    if self.with_bias:
      self.bias = self.param('bias', self.bias_init, (self.out_features,))
  
  def __call__(self, x):
    x = jnp.dot(x, self.kernel)
    if self.with_bias:
      x = x + self.bias
    return x

@dataclass
class MLPExplicit(Module):
  features: List[int]

  # NOTE: Could use @autonames here and then you don't need to give
  # submodules names
  def ready(self):
    self.layers = [
      DenseExplicit(self, self.features[i], self.features[i+1], name='dense' + str(i))
      for i in range(len(self.features)-1)
    ]

  def __call__(self, x):
    for l in self.layers[:-1]:
      x = nn.relu(l(x))
    return self.layers[-1](x)


mlp_expl = MLPExplicit.toplevel(
  features=[3, 4, 5, 6],
  rngs={'param': jax.random.PRNGKey(1)},
  mutable=['param'],  # QUESTION: Should this be required here? We don't want param mutable afterwards.
)

# Note that submodules are here immediately, no need for lazy init like
# you do with "define-by-call" MLPs
print("MLP layers[1].kernel", mlp_expl.layers[1].kernel)
print("output", mlp_expl(jnp.ones((3, 3))))
