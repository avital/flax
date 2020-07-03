import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.module import Module, autonames
from dataclasses import dataclass

@dataclass
class Dense(Module):
  features: int
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros
  name: str = None

  def __call__(self, x):
    kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features))
    return jnp.dot(x, kernel) + self.param('bias', self.bias_init, (self.features,))

@dataclass
class AutoEncoder(Module):
  encoder_widths: Iterable
  decoder_widths: Iterable
  in_shape: Tuple = None
  name: str = None

  def reconstruct(self, x):
    return self.decode(self.encode(x))
  
  @autonames.prefix('encode:')
  def encode(self, x):
    self.in_shape = x.shape[1:]
    for width in self.encoder_widths[:-1]:
      x = nn.relu(Dense(self, width)(x))
    z = Dense(self, self.encoder_widths[-1])(x)
    return z
  
  @autonames.prefix('decode:')
  def decode(self, z):
    for width in self.decoder_widths[:-1]:
      z = nn.relu(Dense(self, width)(z))
    x = Dense(self, self.decoder_widths[-1])(z)
    x = x.reshape(x.shape[:-1] + self.in_shape)
    return x

ae = AutoEncoder.toplevel(
  encoder_widths=[3,4,5], decoder_widths=[4,3,2],
  rngs={'param': random.PRNGKey(42)}, mutable=['param'])
print("reconstruct", ae.reconstruct(jnp.ones((2, 2))))
print("encoder", ae.encode(jnp.ones((2, 2))))
print("vars", ae.variables())