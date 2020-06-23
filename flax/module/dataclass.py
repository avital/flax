import dataclasses
import functools
import inspect
from .module import Module

def forbid_reuse_by_name_method(fun):
  @functools.wraps(fun)
  def wrapped(self, *args, **kwargs):
    # Remove tracks of any submodules created by this method, as we
    # are running it again now.
    ks = [k for k, v in self._method_by_name.items() if v == fun]
    for k in ks:
      del self._method_by_name[k]

    prev_method = self._current_method
    self._current_method = fun
    try:
      return fun(self, *args, **kwargs)
    finally:
      self._current_method = prev_method

  return wrapped

def forbid_reuse_by_name(cls):
  for key in set(dir(cls)).difference(dir(Module)):
    val = getattr(cls, key)
    if inspect.isfunction(val) and not inspect.ismethod(val):
      setattr(cls, key, forbid_reuse_by_name_method(val))
  return cls

def dataclass(cls):
  if Module not in cls.__bases__:
    raise ValueError("Must extend from Module to use @module.dataclass")
  # TODO: Try adding "name: Optional[str] = None" to the
  # dataclass definition.
  cls = dataclasses.dataclass(cls)
  cls = forbid_reuse_by_name(cls)
  return cls
