import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax import module
from flax.module import Module, autonames
from jax import jit

@module.dataclass
class Dense(Module):
  features: int
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros
  name: str = None

  def __call__(self, x):
    kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features))
    return jnp.dot(x, kernel) + self.param('bias', self.bias_init, (self.features,))

@module.dataclass
class MLP(Module):
  widths: Tuple
  name: str = None

  @autonames
  def __call__(self, x):
    for width in self.widths[:-1]:
      x = nn.relu(Dense(self, width)(x))
    x = Dense(self, self.widths[-1])(x)
    return x

X = jnp.ones((1, 10))
Y = jnp.ones((5, ))

@jit
def predict(params):
  return MLP.toplevel([3, 4, 5], variables={'param': params})(X)
  
@jit
def loss_fn(params):
  return jnp.mean(jnp.abs(Y - predict(params)))

@jit
def init_params(rng):
  mlp = MLP.toplevel([3, 4, 5], rngs={'param': rng})
  # calling `mlp(X)` here leads to an error -- variables are frozen.
  # instead, use `mlp.initialized` that returns a clone that has
  # initialized variables
  #
  # TODO: Make `rngs` an argument to `initialized` instead of placing it on
  # the top level MLP above.
  #
  # TODO2: Consider whether .toplevel() becomes necessary at that point.
  mlp = mlp.initialized(X)
  return mlp.variables()['param']

# You can evaluate the loss function with a given PRNG
loss_fn(init_params(jax.random.PRNGKey(42)))

# You can take gradients of the loss function w.r.t. parameters
# (in this case we're evaluating at the initial parameters)
jax.grad(loss_fn)(init_params(jax.random.PRNGKey(42)))

# Run SGD.
params = init_params(jax.random.PRNGKey(42))
for i in range(50):
  loss, grad = jax.value_and_grad(loss_fn)(params)
  print(i, "loss = ", loss, "Yhat = ", predict(params))
  lr = 0.03
  params = jax.tree_multimap(lambda x, d: x - lr * d, params, grad)
  