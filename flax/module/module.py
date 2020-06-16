from dataclasses import dataclass
import dataclasses
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.core.scope import Scope

# CONSIDER: Rename scope to scoping? I really like being able to name variables `scope`
from flax.core import scope as scoping
import functools

from flax.core.frozen_dict import freeze

@dataclass
# TODO: Document that any class that extends from Module must add
#   name = Optional[None]
class Module:
  """Base class for parametrized functions.

  A module instance is an object that holds variables and optional PRNGs.
  Methods on module instances describe parameterized functions, with arbitrary
  inputs and parameterization on `self.variables`. 

  More TBD
  """
  parent: Union[Type["Module"], Type["Scope"]]
  
  @classmethod
  def toplevel(cls, *args, rngs=None, variables=None, mutable=False, **kwargs):
    # TODO: Think about the fact that `rngs` and `params` live on kwargs. What if
    # someone wants to pass a kwarg named "rngs", to do RNG management manually?
    if rngs is None:
      rngs = {}
    if variables is None:
      variables = {'param': {}}
    variables = scoping._unfreeze_variables(variables, mutable)
    parent = Scope(variables, rngs=rngs)
    module = cls(parent, *args, **kwargs)
    return module

  def _ensure_has_name(self):
    if self.name is None:
      if self.parent._autoname_cursor is None:
        raise ValueError("In order to get autonames, must decorate method with @autonames")
      
      self.name = f"{self.parent._autoname_prefix}{self.__class__.__name__}/{self.parent._autoname_cursor}"
      self.parent._autoname_cursor += 1

  def __post_init__(self):
    if isinstance(self.parent, Module):
      self._ensure_has_name()
      self.parent.submodules[self.name] = self
      self.scope = self.parent.scope.push(self.name)

    elif isinstance(self.parent, Scope):
      self.scope = self.parent

    else:
      raise ValueError("parent must be a Module or Scope")
      
    self.submodules = {}
    self._autoname_prefix = None
    self._autoname_cursor = None
    self._autoname_funs = {}

    self.ready()
    
  def ready(self):
    """Called when module instance receives variables and PRNGs.
    
    If you want to use PRNGs and/or read or write variables during module
    instance construction, override this method. It will be automatically called indirectly
    from `__post_init__`.
    """
    pass
        
  def copy(self, rngs=None, variables=None, mutable=False, **kwargs):
    """Construct a new module instance based on this one, with overrides."""
    return self.__class__.toplevel(
      **dataclasses.asdict(module),
      rngs=rngs, variables=variables, mutable=mutable, **kwargs)

  # QUESTION: Should this be a property? Or should it be assigned
  # to `self.variables` during __post_init__?
  def variables(self):
    return self.scope.variables

  def param(self, name, init_fun, shape):
    return self.scope.param(name, init_fun, shape)

  # TODO: Methods to access non-parameter variables from scope.

def autonames(fun, prefix=''):
  @functools.wraps(fun)
  def wrapped(self, *args, **kwargs):
    if prefix in self._autoname_funs and self._autoname_funs[prefix] != fun:
      raise ValueError(
        "Can't only use @autonames on one method. To reuse submodules across methods, "
        "store submodules on `self` during `ready()`. If you want two methods to each "
        "have distinct sets of autonamed submodules, use `@autonames.prefix`.")
    else:
      self._autoname_funs[prefix] = fun

    if self._autoname_cursor is not None:
      raise ValueError("Can't nest calls to autonamed methods")

    # "Rewind" the autonaming process, and set the prefix
    # NOTE: that these are dyanmically scoped, but only on a per-instance
    # level. Moreover, nesting calls to autonamed methods throws an error so
    # we guard against the most obvious mistakes one could make
    self._autoname_prefix = prefix
    self._autoname_cursor = 0
    try:
      return fun(self, *args, **kwargs)
    finally:
      self._autoname_cursor = None
      self._autoname_prefix = None

  return wrapped

autonames.prefix = lambda prefix: functools.partial(autonames, prefix=prefix)