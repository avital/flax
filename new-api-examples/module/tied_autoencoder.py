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
    # NOTE: I tried overriding __setattr__
    self.encoder = nn.Dense(self, features=3)
    self.decoder = nn.Dense(self, features=self.in_features)

    # QUESTION: Should we do something better than jnp.ones here?
    # (like current Flax's init_by_shape?)
    self(jnp.ones((1, self.in_features)))  # initialize dense parameters defined lazily

    # TODO: Use EasyDict so that we get this syntax: 
    # self.decoder.variables.param.kernel = self.encoder.variables.param.kernel.T
    self.decoder.variables()['param']['kernel'] = self.encoder.variables()['param']['kernel'].T

  def __call__(self, x):
    z = nn.sigmoid(self.encoder(x))
    return nn.sigmoid(self.decoder(z))

@jax.jit
def init_vars():
  tae = TiedAutoencoder.toplevel(rngs={'param': random.PRNGKey(0)})
  return tae.variables()

@jax.jit
def loss(variables):
  X = jnp.ones((16, 4*4))
  Y = jnp.ones((16, 4*4))
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



