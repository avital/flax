import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.module import Module, dataclass
from flax import module

@module.dataclass
class DenseExplicit(Module):
  in_features: int
  out_features: int
  with_bias: bool = True
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros

  def setup(self):
    self.kernel = self.param('kernel', self.kernel_init, (self.in_features, self.out_features))
    if self.with_bias:
      self.bias = self.param('bias', self.bias_init, (self.out_features,))
  
  def __call__(self, x):
    x = jnp.dot(x, self.kernel)
    if self.with_bias:
      x = x + self.bias
    return x


@module.dataclass
class TwoLayerMLP(Module):
  def setup(self, x):
    self.dense1 = DenseExplicit(self, in_features=3, out_features=3)
    self.dense2 = DenseExplicit(self, in_features=3, out_features=3)

  def __call__(self, x):
    return self.dense2(nn.relu(self.dense1(x)))


@module.dataclass
class MLPExplicit(Module):
  features: List[int]

  def setup(self):
    self.layers = [
      DenseExplicit(self, self.features[i], self.features[i+1])
      for i in range(len(self.features)-1)
    ]

  def __call__(self, x):
    for l in self.layers[:-1]:
      x = nn.relu(l(x))
    return self.layers[-1](x)


mlp_expl = MLPExplicit(
  parent=None,
  features=[3, 4, 5, 6],
  rngs={'params': jax.random.PRNGKey(1)},
)

# Note that submodules are here immediately, no need for lazy init like
# you do with "define-by-call" MLPs
print("MLP layers[1].kernel", mlp_expl.layers[1].kernel)
print("output", mlp_expl(jnp.ones((3, 3))))
