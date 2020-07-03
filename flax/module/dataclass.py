import dataclasses
import functools
import inspect
from .module import Module
from .autonames import public 
from flax.core.scope import Variables, RNGs

def forbid_reuse_by_name_method(fun):
  @functools.wraps(fun)
  def wrapped(self, *args, **kwargs):
    # Remove tracks of any submodules created by this method, as we
    # are running it again now.
    ks = [k for k, v in self._method_by_name.items() if v == fun]
    for k in ks:
      del self._method_by_name[k]

    outer_method = self._current_method
    self._current_method = fun
    try:
      return fun(self, *args, **kwargs)
    finally:
      self._current_method = outer_method

  return wrapped

def forbid_reuse_by_name(cls):
  dataclass_fieldnames = set([f.name for f in dataclasses.fields(dataclasses.dataclass(cls))])
  for key, val in cls.__dict__.items():
    if (key not in dataclass_fieldnames and
        (not key.startswith('__') or key == '__call__') and
        inspect.isfunction(val) and
        not inspect.ismethod(val)):
      setattr(cls, key, forbid_reuse_by_name_method(val))
  return cls


def make_call_public(cls):
  if hasattr(cls, '__call__'):
    cls.__call__ = public(cls.__call__)
  return cls

def dataclass(cls):
  if Module not in cls.__bases__:
    raise ValueError("Must extend from Module to use @module.dataclass")
  
  cls.__annotations__["name"] = str
  cls.name = None
  
  cls.__annotations__["variables"] = Variables
  cls.variables = dataclasses.field(default=None, repr=False)

  cls.__annotations__["rngs"] = RNGs
  cls.rngs = dataclasses.field(default=None, repr=False)

  cls = forbid_reuse_by_name(cls)
  cls = make_call_public(cls)

  # Important to put `dataclass` last so that `module.clone()` works --
  # we need instantiating a copy of this data class to also include
  # the wrappers above.
  cls = dataclasses.dataclass(cls)
  return cls
