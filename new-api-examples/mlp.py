import functools
import jax
from jax import numpy as jnp, random, lax
from flax import nn, struct
from flax.core.scope import Scope, init, apply, Array, group_kinds

def dense(scope: Scope, inputs: Array, features: int, bias: bool = True,
          kernel_init=nn.linear.default_kernel_init,
          bias_init=nn.initializers.zeros):
  kernel = scope.param('kernel', kernel_init, (inputs.shape[-1], features))
  y = jnp.dot(inputs, kernel)
  if bias:
    y += scope.param('bias', bias_init, (features,))
  return y

model_fn = functools.partial(dense, features=3)

x = jnp.ones((1, 2))
y, params = init(model_fn)(random.PRNGKey(0), x)
print(params)

def mlp(scope: Scope, inputs: Array, features: int):
  hidden = dense(scope.push('hidden'), inputs, features)
  hidden = nn.relu(hidden)
  return dense(scope.push('out'), hidden, 1)

init(mlp)(random.PRNGKey(0), x, features=3)
