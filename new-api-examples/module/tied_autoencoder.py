"""An autoencoder where decoder weights are transposed encoder weights."""
# (Does this thing have a name?)

# TODO: This is a weird import idiom.
from flax import module
from flax.module import Module
from flax import nn
from jax import numpy as jnp
from jax import random
import jax

from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union

@module.dataclass
class TiedAutoencoder(Module):
  in_features: int = 4*4
  name: Optional[str] = None

  def setup(self):
    self.encoder = nn.Dense(parent=None, features=3)

    class Decoder(nn.Dense):
      def param(decoder, name, init_fun, shape):
        if name == 'kernel':
          return self.encoder.variables()['param']['kernel'].T
        else:
          return super().param(name, init_fun, shape)
    self.decoder = Decoder(parent=None, features=self.in_features)

  def __call__(self, x):
    z = nn.sigmoid(self.encoder(x))
    return nn.sigmoid(self.decoder(z))

def init_vars():
  tae = TiedAutoencoder.toplevel(rngs={'param': random.PRNGKey(0)})
  tae = tae.initialized(jnp.ones((1, tae.in_features)))
  return tae.variables()

def loss(variables):
  X = jnp.ones((16, 4*4)) * 0.5
  Y = jnp.ones((16, 4*4)) * 0.5
  tae = TiedAutoencoder.toplevel(variables=variables)
  return jnp.mean(jnp.abs(tae(X) - Y))

variables = init_vars()
# train loop
print("initial loss", loss(variables))
for _ in range(100):
  variables = jax.tree_multimap(lambda v, g: v - 0.01 * g, variables, jax.grad(loss)(variables))
print("100 training steps loss", loss(variables))
print("variables")
print(jax.grad(loss)(variables))



