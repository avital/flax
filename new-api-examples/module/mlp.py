import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.module import Module, autonames
from flax import module
from dataclasses import dataclass

@module.dataclass
class Dense(Module):
  features: int
  bias: bool = True
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros

  def __call__(self, x):
    kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features))
    y = jnp.dot(x, kernel)
    if self.bias:
      y = y + self.param('bias', self.bias_init, (self.features,))
    return y


# MLP where layers are defined in __call__
@module.dataclass
class MLP(Module):
  widths: Tuple

  def __call__(self, x):
    for width in self.widths[:-1]:
      x = nn.relu(Dense(self, width)(x))
    x = Dense(self, self.widths[-1])(x)
    return x


@dataclass
class Sequential(Module):
  layers: Tuple[Module]
  name: str = None

  def __call__(self, x):
    for layer in layers:
      x = layer(x)
    return x

# MLP where layers are defined in ready() (what we use instead of __init__)
# and used in __call__
@module.dataclass
class MLP2(Module):
  widths: Tuple
  name: str = None

  def setup(self):
    # TODO: Can we make it throw an error if we do `self.layers = [Dense(self, width)] * 3`?
    self.layers = [Dense(self, width) for width in self.widths]
    
  def __call__(self, x):
    for layer in self.layers[:-1]:
      x = nn.relu(layer(x))
    x = self.layers[-1](x)
    return x

if __name__ == '__main__':
  x = jnp.ones((1, 2))
  dense = Dense(None, features=3, rngs={'params': random.PRNGKey(0)}).initialized(jnp.ones((3, 3)))
  print("Dense instance", dense)
  print("parameters after call", dense.param)

  print()
  print("-------")
  print()

  mlp = MLP(None, [2,3,4], rngs={'params': random.PRNGKey(42)}).initialized(jnp.ones((3, 3)))
  print("mlp output", mlp(jnp.ones((1, 3))))
  print("mlp params", mlp.params)

  # QUESTION: Is this annoying?
  mlp2_params = {k.replace('Dense/', 'layers/'): mlp.params[k] for k in mlp.params}

  # note that nothing is mutable here and we don't need prngs.
  mlp2 = MLP2(None, [2,3,4], variables={"params": mlp2_params})
  print(mlp2.variables)
  print("mlp2 output", mlp2(jnp.ones((1, 3))))
    
