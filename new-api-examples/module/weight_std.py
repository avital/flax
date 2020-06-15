# ***THIS PROBABLY DOESN'T WORK*** I never ran it!

def standardize(x, axis, eps=1e-8):
  x = x - jnp.mean(x, axis=axis, keepdims=True)
  x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
  return x
  
@dataclass
class StdWeight:
  module: Module
  
  def __call__(self, x):
    if not self.module.params():
      # initialize parameters
      self.module(x)

    param = self.module.variables.param
    # TODO: Test that I would get an error if I directly modified `param`
    std_param = param.copy(kernel=standardize(param['kernel'], axis=[0, 1]))
    return module.copy({'param': std_param})(x)

class MyModule:
  def foo(self, x):
    module = Dense(self, 3)
    std_module = StdWeight(module)
    std_module(x)  # parameters