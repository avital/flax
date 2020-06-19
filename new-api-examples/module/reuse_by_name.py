import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.module import Module, autonames
from dataclasses import dataclass
import functools
import sys
import traceback


def no_reuse_by_name(fun):
  @functools.wraps(fun)
  def wrapped(self, *args, **kwargs):
    prev_method = self._current_method
    self._current_method = fun

    # Remove tracks of any submodules created by this method, as we
    # are running it again now.
    ks = [k for k, v in self._method_by_name.items() if v == fun]
    for k in ks:
      del self._method_by_name[k]

    try:
      return fun(self, *args, **kwargs)
    finally:
      self._current_method = prev_method

  return wrapped


@dataclass
class Dense(Module):
  features: int
  name: str = None

  def __call__(self, x):
    kernel = self.param('kernel', initializers.lecun_normal(), (x.shape[-1], self.features))
    return jnp.dot(x, kernel) + self.param('bias', initializers.zeros, (self.features,))


@dataclass
class TestReuseByName(Module):
  @no_reuse_by_name
  def should_work_1(self, x):
    return Dense(self, 3, name="foo")(x) + Dense(self, 3, name="bar")(x)
    
  @no_reuse_by_name
  @autonames
  def should_work_2(self, x):
    return Dense(self, 3)(x) + Dense(self, 3)(x)

  @no_reuse_by_name
  @autonames
  def should_work_3(self, x):
    return _should_work_3_factory(x) + _should_work_3_factory(x)

  @no_reuse_by_name
  def _should_work_3_factory(self, x):
    return Dense(self, 3)(x)

  @no_reuse_by_name
  def should_fail_1(self, x):
    # Reuse by name in a single method
    return Dense(self, 3, name="foo")(x) + Dense(self, 3, name="foo")(x)

  @no_reuse_by_name
  def should_fail_2(self, x):
    def _dense(x):
      return Dense(self, 3, name="foo")(x)
    # Submodule name reused in separate closure, called twice
    return _dense(x) + _dense(x)

  @no_reuse_by_name
  # *********** --->> this is the discussion we are having this morning <<---
  # In Haiku this workd, and I think we may have no alternative but 
  # to make this also work.
  def should_fail_3(self, x):
    # Submodule name reused in same method, but called twice
    return self._should_fail_3_dense(x) + self._should_fail_3_dense(x)

  @no_reuse_by_name
  def _should_fail_3_dense(self, x):
    return Dense(self, 3, name="foo")(x)

  @no_reuse_by_name
  def should_fail_4(self, x):
    # Call two separate methods that reuse submodule by name
    return self._should_fail_3_dense(x) + self._should_fail_4_dense(x)

  @no_reuse_by_name
  def _should_fail_4_dense(self, x):
    return Dense(self, 3, name="foo")(x)



def call_method(method):
  try:
    module = TestReuseByName.toplevel(rngs={"param": random.PRNGKey(32)}, mutable=['param'])
    method(module, jnp.ones((2, 2)))
    print(method.__name__, "Success")
  except Exception as e:
    print(method.__name__, "Error:", e)
#    traceback.print_exc()

print("Should work")
call_method(TestReuseByName.should_work_1)  
call_method(TestReuseByName.should_work_2)  

print("--- Should fail ---")
call_method(TestReuseByName.should_fail_1)  
call_method(TestReuseByName.should_fail_2)  
call_method(TestReuseByName.should_fail_3)  
call_method(TestReuseByName.should_fail_4)  
