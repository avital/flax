# NOTE: It would be nice to make this throw an error,
# but how? I'd like to avoid requiring people to wrap /all/
# methods in a decorator (or the similar metaclass approach with
# hk.transparent).
#
# ...but maybe, we could use a metaclass that /only/ does error checking
# like this?
#
# QUESTION: Can we resolve this by inspecting stack traces 
# when constucting modules, or when using them? Only during
# "DEBUG" runs
class TryReusingByNameCausesError(Module):
  def __call__(self, x):
    return Dense(self, 3, name="foo")(x) + Dense(self, 3, name="foo")(x)

  def call2(self, x):
    return Dense(self, 3, name="foo")(x)
  
try_reuse = TryReusingByNameCausesError.toplevel(rngs={'param': jax.random.PRNGKey(0)}, mutable=['param'])
try_reuse(np.ones((3, 3)))
try_reuse(np.ones((3, 3)))
try_reuse.call2(np.ones((3, 3)))