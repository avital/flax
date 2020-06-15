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
    # TODO: Think about the fact that `rngs` and `params` live on args
    # and kwargs
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
      
      self.name = f"{self.__class__.__name__}/{self.parent._autoname_cursor}"
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
    self._autoname_cursor = None
    self._autoname_fun = None

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


def autonames(fun):
  @functools.wraps(fun)
  def wrapped(self, *args, **kwargs):
    if self._autoname_fun and self._autoname_fun != fun:
      raise Error(
        "Can't only use @autonames on one method. To reuse submodules across methods, "
        "store submodules on `self` during `ready()`. If you want two methods to each "
        "have distinct sets of autonamed submodules, instead make wrapper submodules -- "
        "one for each method. Read more at http://TBD.")
    self._autoname_fun = fun

    # "Rewind" the autonaming process
    self._autoname_cursor = 0
    try:
      return fun(self, *args, **kwargs)
    finally:
      self._autoname_cursor = None

  return wrapped
