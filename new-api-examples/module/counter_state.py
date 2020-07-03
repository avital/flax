import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax import module
from flax.module import Module, autonames
from jax import jit, vmap
from flax.core import lift

@module.dataclass
class Dense(Module):
  features: int
  name: str = None

  def __call__(self, x):
    kernel = self.param('kernel', initializers.lecun_normal(), (x.shape[-1], self.features))
    return jnp.dot(x, kernel) + self.param('bias', initializers.zeros, (self.features,))

@module.dataclass
class Counter(Module):
  name: str = None

  def __call__(self, x):
    # TODO: Why do we need shape here?
    counter, set_counter = self.variable('counters', 'count', lambda *args: 0, shape=None)
    set_counter(counter+1)
    return x

@module.dataclass
class DenseAndCounter(Module):
  name: str = None

  def __call__(self, x):
    counter = Counter(self)
    return Dense(self, features=2)(counter(x))

X = jnp.ones((2, 2))
Y = jnp.ones((2, 2))

@jit
def predict(variables):
  mlp = DenseAndCounter(None, variables=variables)
  with mlp.mutate(mutable=['counters']) as mutated:
    y = mutated(X)
    return y, mutated.counters
  
@jit
def loss_fn(params, counter):
  pred, new_counters = predict({'params': params, 'counters': counter})
  return jnp.mean(jnp.abs(Y - pred)), new_counters

@jit
def init_variables(rng):
  # TODO: Why do I need a PRNG for the "counter" kind?
  # TODO: Why do I need to pass an empty variables dict for 'counter'?
  mlp = DenseAndCounter(None, 
    rngs={'params': rng, 'counters': rng},
    variables={'params': {}, 'counters': {}})
  mlp = mlp.initialized(X)
  return mlp.variables

# Run SGD.
variables = init_variables(jax.random.PRNGKey(42))
for i in range(50):
  print(variables)
  loss, grad = jax.value_and_grad(loss_fn, has_aux=True)(variables['params'], variables['counters'])
  y, new_counter = predict(variables)
  lr = 0.03
  variables = {
    'params': jax.tree_multimap(lambda x, d: x - lr * d, variables['params'], grad),
    'counters': new_counter
  }



