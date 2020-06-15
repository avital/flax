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

  def ready(self):
    # NOTE that had you originally tried making encoder
    # and decoder methods on this class, then:
    #
    # 1. You'd get an error that you need to wrap in autonames
    #    to get autonamed submodules
    # 2. You'd put @autonamed around both methods and then get
    #    an error saying that only one method can use @autonames
    #    and that if you want the two methods to not share submodules
    #    create a concrete Module instead of each of the two methods
    # 3. Then, hopefully you'd end up doing this:
    self.encoder = AutoEncoder.Encoder(self)
    self.decoder = AutoEncoder.Decoder(self)
  
  def reconstruct(self, x):
    return self.decoder(self.encoder(x))
  
  @dataclass
  class Encoder(Module):
    name: str = 'encoder'
    
    @autonames
    def __call__(self, x):
      self.in_shape = x.shape[1:]
      # QUESTION: Is this a legitimate use of `self.parent`?
      # Alternatively, you could pass in `widths` as an attribute
      # of Encoder
      for width in self.parent.encoder_widths[:-1]:
        x = nn.relu(Dense(self, width)(x))
      z = Dense(self, self.parent.encoder_widths[-1])(x)
      return z
  
  @dataclass
  class Decoder(Module):
    name: str = 'decoder'
    
    @autonames
    def __call__(self, z):
      for width in self.parent.decoder_widths[:-1]:
        z = nn.relu(Dense(self, width)(z))
      x = Dense(self, self.parent.decoder_widths[-1])(z)
      # QUESTION: Is this weird? Navigating up then into encoder?
      # We could pass `in_shape` here, but we'd only know that after
      # encoding once.
      x = x.reshape(x.shape[:-1] + self.parent.encoder.in_shape)
      return x

ae = AutoEncoder.toplevel(
  encoder_widths=[3,4,5], decoder_widths=[4,3,2],
  rngs={'param': random.PRNGKey(42)}, mutable=['param'])
print("reconstruct", ae.reconstruct(jnp.ones((2, 2))))
print("encoder", ae.encoder(jnp.ones((2, 2))))
print("vars", ae.variables())