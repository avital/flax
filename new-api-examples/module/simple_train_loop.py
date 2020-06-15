import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.module import Module, autonames
from dataclasses import dataclass
from jax import jit

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

## To use JAX transformations like @jit, you need to follow certain patterns
# to make pure functions. Example below include predicting from a parameter dict,
# and returning the initialized parameters for a module. You could do arbitrary things
# between construction of a module instance until you return values from the function,
# which gives you a lot of flexibility -- what if you want to apply the module, then reset
# batch norm stats and run it again, computing the difference between the two runs?
# Now you can!

@jit
def predict(params):
  # TODO: Think about the fact that you have to put the hyperparameters here  
  mlp = MLP.toplevel([3, 4, 5], variables={'param': params})
  return mlp(X)
  
@jit
def loss_fn(params):
  Yhat = predict(params)
  # TODO: Print in jit
  return jnp.mean(jnp.abs(Y - Yhat))

@jit
def init_params(rng):
  # TODO: Think about the fact that you have to put the hyperparameters here  
  mlp = MLP.toplevel([3, 4, 5], rngs={'param': rng}, mutable=['param', 'state'])
  # Pass an input in to initialize parameters
  mlp(X)
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
  