# Lint as: python3
# pylint: disable=invalid-name

"""Foo."""

import contextlib
import enum

from . import initializers

import jax
import jax.numpy as jnp
from collections.abc import Callable
from typing import Any

import numpy as np

import dataclasses
from dataclasses import dataclass
from dataclasses import field

class MiniModule:
  """MiniModule."""

  params = None
  ExecutionMode = enum.Enum('ExecutionMode', 'INIT APPLY')
  execution_mode = None

  def __init__(self, name=None):
    self.name = name

  def __call__(self, *args, **kwargs):
    assert MiniModule.execution_mode is not None
    if self.name is not None:  # xcxc code smell?
      if self.name not in MiniModule.params:
        MiniModule.params[self.name] = {}
      subparams = MiniModule.params[self.name]
    else:
      subparams = MiniModule.params

    with MiniModule.params_scope(subparams):
      return self.apply(*args, **kwargs)

  @staticmethod
  @contextlib.contextmanager
  def params_scope(params):
    """Apply scope."""
    orig_params = MiniModule.params
    orig_execution_mode = MiniModule.execution_mode

    MiniModule.params = params
    try:
      yield
    finally:
      MiniModule.params = orig_params
      MiniModule.execution_mode = orig_execution_mode

  def apply_with_params(self, params, *args, **kwargs):
    """Apply with parameters."""
    assert MiniModule.execution_mode is None
    assert MiniModule.params is None, MiniModule.params
    MiniModule.execution_mode = MiniModule.ExecutionMode.APPLY
    try:
      with MiniModule.params_scope(params):
        return self.apply(*args, **kwargs)
    finally:
      MiniModule.execution_mode = None

  def init_params(self, *args, **kwargs):
    """Init parameters."""
    assert MiniModule.execution_mode is None
    assert MiniModule.params is None, MiniModule.params
    MiniModule.execution_mode = MiniModule.ExecutionMode.INIT
    params = {}
    try:
      def _apply():
        self.apply(*args, **kwargs)
      with MiniModule.params_scope(params):
        jax.eval_shape(_apply)
      return params
    finally:
      MiniModule.execution_mode = None

  def param(self, name, shape, init_fn):
    """Define or use a parameter."""
    if MiniModule.execution_mode is None:
      raise ValueError('Should be in either init or apply execution mode.')

    elif MiniModule.execution_mode == MiniModule.ExecutionMode.INIT:
      # xcxc prng key
      # xcxc check if parameter already declared.
      MiniModule.params[name] = init_fn(jax.random.PRNGKey(0), shape)
      return jnp.ones(shape)  # xcxc

    elif MiniModule.execution_mode == MiniModule.ExecutionMode.APPLY:
      return MiniModule.params[name]


class MiniDense(MiniModule):
  """A linear transformation applied over the last dimension of the input."""

  def apply(self, inputs, features,
            kernel_init=initializers.lecun_normal(),
            bias_init=initializers.zeros):
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.
      features: the number of output features.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    Returns:
      The transformed input.
    """
    kernel = self.param('kernel', (inputs.shape[-1], features), kernel_init)
    bias = self.param('bias', (features,), bias_init)
    y = jnp.dot(inputs, kernel) + bias
    return y











# ===


@dataclass
class MiniModule2:
  """MiniModule."""

#  params: ClassVar = None
#  ExecutionMode: ClassVar = enum.Enum('ExecutionMode', 'INIT APPLY')
#  execution_mode: ClassVar = None

  name: str

  def __call__(self, *args, **kwargs):
    assert MiniModule.execution_mode is not None
    if self.name is not None:  # xcxc code smell?
      if self.name not in MiniModule.params:
        MiniModule.params[self.name] = {}
      subparams = MiniModule.params[self.name]
    else:
      subparams = MiniModule.params

    with MiniModule.params_scope(subparams):
      return self.apply(*args, **kwargs)

  @staticmethod
  @contextlib.contextmanager
  def params_scope(params):
    """Apply scope."""
    orig_params = MiniModule.params
    orig_execution_mode = MiniModule.execution_mode

    MiniModule.params = params
    try:
      yield
    finally:
      MiniModule.params = orig_params
      MiniModule.execution_mode = orig_execution_mode

  def apply_with_params(self, params, *args, **kwargs):
    """Apply with parameters."""
    assert MiniModule.execution_mode is None
    assert MiniModule.params is None, MiniModule.params
    MiniModule.execution_mode = MiniModule.ExecutionMode.APPLY
    try:
      with MiniModule.params_scope(params):
        return self.apply(*args, **kwargs)
    finally:
      MiniModule.execution_mode = None

  def init_params(self, *args, **kwargs):
    """Init parameters."""
    assert MiniModule.execution_mode is None
    assert MiniModule.params is None, MiniModule.params
    MiniModule.execution_mode = MiniModule.ExecutionMode.INIT
    params = {}
    try:
      def _apply():
        self.apply(*args, **kwargs)
      with MiniModule.params_scope(params):
        jax.eval_shape(_apply)
      return params
    finally:
      MiniModule.execution_mode = None

  def param(self, name, shape, init_fn):
    """Define or use a parameter."""
    if MiniModule.execution_mode is None:
      raise ValueError('Should be in either init or apply execution mode.')

    elif MiniModule.execution_mode == MiniModule.ExecutionMode.INIT:
      # xcxc prng key
      # xcxc check if parameter already declared.
      MiniModule.params[name] = init_fn(jax.random.PRNGKey(0), shape)
      return jnp.ones(shape)  # xcxc

    elif MiniModule.execution_mode == MiniModule.ExecutionMode.APPLY:
      return MiniModule.params[name]



@dataclass
class MiniDense2(MiniModule2):
  features: int
  kernel_init: Callable
  bias_init: Callable

  def __call__(self, x):
    """Applies a linear transformation to the inputs along the last dimension."""
    kernel = self.param(
        'kernel', (x.shape[-1], self.features), self.kernel_init)
    bias = self.param('bias', (self.features,), self.bias_init)
    return jnp.dot(x, kernel) + bias
  


# ====

class MiniModule3:
  def with_params(self, params):
    # xcxc destructive, and just plain weird.
    self.params = params
    return self



class DynamicVar:
  def __init__(self):
    # xcxc make this thread-safe
    self._val = None

  def peek(self):
    return self._val

  def get(self):
    if self._val is None:
      raise ValueError("Must run within a `with dynamic_var.let:` block")
    else:
      return val

  @contextlib.contextmanager
  def let(self, val):
    assert val is not None
    old_val = self._val
    self._val = val
    yield
    self._val = old_val


class WithParams(MiniModule3):
  _parent = DynamicVar()

  # unfortunate a Union of two bits of functionality:
  # 1/ a module that holds params
  # 2/ a module that has a parent
  # 
  # but it's better than mucking more into multiple subclasses
  # / trying to implement mixins.
  def __init__(self):
    self.params = None

  def let_parent(parent):
    WithParams._parent.let(parent)

  def parent(self):
    parent = self._parent.get()

    if not hasattr(parent, 'children'):
      # xcxc check duplicate names
      parent.children = {}
    parent.children[self.name] = self

  def params(self):
    if self.params:
      assert self._parent.peek() is not None
      return self.params
    else:
      parent_params = self.parent().params()
      if not self.name in parent_params:
        parent_params[self.name] = {}
      return parent_params[self.name]

  def init_param(self, name, shape, init_fn):
    params = self.params()
    if name in params:
      raise ValueError(f"Parameter {name} defined twice")
    # xcxc prngkey
    params[name] = init_fn(jax.random.PRNGKey(0), shape)
    return params[name]

  def get_param(self, name):
    return self.params()[name]


class JointInitApply(WithParams):
  ExecutionMode = enum.Enum('ExecutionMode', 'INIT APPLY')
  # xcxc thread-safe
  execution_mode = DynamicVar()

  def param(self, name, shape, init_fn):
    if self.execution_mode.get() == self.ExecutionMode.INIT:
      return self.init_param(name, shape, init_fn)
    elif self.execution_mode.get() == self.ExecutionMode.APPLY:
      return self.get_param(name)
    else:
      assert False, self.execution_model.get()

  @staticmethod
  @contextlib.contextmanager
  def mode(execution_mode):
    execution_mode.let(execution_mode)


@dataclass
class MiniDense3(JointInitApply):
  features: int
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros
  name: String = None

  def __call__(self, x):
    """Applies a linear transformation to the inputs along the last dimension."""
    kernel = self.param('kernel', (x.shape[-1], self.features), self.kernel_init)
    bias = self.param('bias', (self.features,), self.bias_init)
    return jnp.dot(x, kernel) + bias

def dense_initial_params():
  with JointInitApply.mode(JointInitApply.ExecutionMode.INIT):
    params_container = {}
    dense = MiniDense3(features=32).with_params(params_container)
    dense(np.random.uniform((256, 16)))  # (BATCH, IN_FEATURES)
    assert dense.params() is params_container
    return dense.params()

def dense_apply(params):
  with JointInitApply.mode(JointInitApply.ExecutionMode.APPLY):
    dense = MiniDense3(features=32).with_params(params)
    result = dense(np.random.uniform((256, 16)))  # (BATCH, IN_FEATURES)
    assert params == dense.params()
    return result

class MLP(JointInitApply):
  def __call__(self, x):
    x = MiniDense3(features=2).with_parent(self, name="dense1")(x)
    x = jax.nn.relu(x)
    x = MiniDense3(features=2).with_parent(self, name="dense2")(x)
    return x

  def l1p(self):
    return self.params()['dense1']

def mlp_initial_params():
  with JointInitApply.mode(JointInitApply.ExecutionMode.INIT):
    mlp = MLP().with_params({})
    mlp(np.random.uniform((4, 1)))  # (BATCH, IN_FEATURES)
    return mlp.params(), mlp.l1p()

class MLPv2(JointInitApply):
  def __call__(self, x):
    layer1 = MiniDense3(features=2).with_parent(self, name="dense1")
    x = layer1(x)
    x = jax.nn.relu(x)
    x = MiniDense3(features=2).with_parent(self, name="dense2")(x)
    return layer1.params(), x

def mlp_v2_initial_params():
  with JointInitApply.mode(JointInitApply.ExecutionMode.INIT):
    mlp = MLPv2().with_params({})
    l1p, _ = mlp(np.random.uniform((4, 1)))  # (BATCH, IN_FEATURES)
    return mlp.params(), l1p

class MLPv3(JointInitApply):
  def __call__(self, x):
    with WithParent.set(self):
      layer1 = MiniDense3(features=2, name="dense1")
      x = layer1(x)
      x = jax.nn.relu(x)
      x = MiniDense3(features=2, name="dense2")(x)
      return layer1.params(), x

def mlp_v3_initial_params():
  with JointInitApply.init() as init_params:
    mlp = MLPv3().with_params({})
    l1p, _ = mlp(np.random.uniform((4, 1)))  # (BATCH, IN_FEATURES)
    return mlp.params(), l1p

# the below doesn't work yet but it could, and it's not too far

class MLPv4(JointInitApply):
  @modulemethod
  def __call__(self, x):
    layer1 = MiniDense3(features=2, name="dense1")
    x = layer1(x)
    x = jax.nn.relu(x)
    x = MiniDense3(features=2, name="dense2")(x)
    return layer1.params(), x

def mlp_v4_initial_params():
  with JointInitApply.mode(JointInitApply.ExecutionMode.INIT):
    mlp = MLPv3().with_params({})
    l1p, _ = mlp(np.random.uniform((4, 1)))  # (BATCH, IN_FEATURES)
    return mlp.params(), l1p

def mlp_v4_simple_initial_params():
  # special cases to wrapping __init__ in a context manager
  MLPv3().init(jax.random.PRNGKey(0))

def mlp_v4_simple_initial_params(x):
  # special cases to wrapping __init__ in a context manager
  with JointInitApply.init(prng_key=jax.random.PRNGKey(0)) as init_params:
    return MLPv3()(x)

def mlp_v4_simple_use_params(x):
  # special cases to wrapping __init__ in a context manager
  MLPv3().with_params(params)(x)
