from jax import jit

X = np.ones((1, 10))
Y = np.ones((5, ))

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
  