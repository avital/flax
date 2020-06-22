from dataclasses import dataclass
import dataclasses
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
# TODO: Make _unfreeze_variables public?
from flax.core.scope import Scope, _unfreeze_variables
import functools

from flax.core.frozen_dict import freeze

from contextlib import contextmanager

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
  def toplevel(cls, *args, rngs=None, variables=None, **kwargs):
    # TODO: Think about the fact that `rngs` and `variables` live on kwargs. What if
    # someone wants to pass a kwarg named "rngs", to do RNG management manually?
    if rngs is None:
      rngs = {}
    if variables is None:
      variables = {'param': {}}
    parent = Scope(freeze(variables), rngs=rngs)
    module = cls(parent, *args, **kwargs)
    return module

  def _ensure_has_name(self):
    if self.name is None:
      if self.parent._autoname_cursor is None:
        raise ValueError("In order to get autonames, must decorate method with @nn.autonames")
      
      self.name = f"{self.parent._autoname_prefix}{self.__class__.__name__}/{self.parent._autoname_cursor}"
      self.parent._autoname_cursor += 1

  def __post_init__(self):
    if isinstance(self.parent, Module):
      self._ensure_has_name()
      self.parent.submodules[self.name] = self

      if self.name in self.parent._method_by_name:
        raise ValueError(f"Trying to share submodule by name in methods {self.parent._current_method} "
                         f"and {self.parent._method_by_name[self.name]}. To share submodules, store "
                         f"module instances as a Python object and reuse. You can store module"
                         f"instance on `self` to share across methods.")
      self.parent._method_by_name[self.name] = self.parent._current_method

      self.scope = self.parent.scope.push(self.name)

    elif isinstance(self.parent, Scope):
      self.scope = self.parent

    else:
      raise ValueError("parent must be a Module or Scope")
      
    self.submodules = {}

    # Optional mechanism for methods to allow for autonamed submodules.
    # See autonames.py
    self._autoname_prefix = None
    self._autoname_cursor = None
    self._autoname_funs = {}

    # Recommended mechanism for forbidding accidental submodule reuse-by-name.
    # See dataclass.py
    self._method_by_name = {}
    self._current_method = None

    # subclasses should implement `ready()` instead of `__init__` or `__post_init__`
    self.ready()
    
  def ready(self):
    """Called when module instance receives variables and PRNGs.
    
    If you want to use PRNGs and/or read or write variables during module
    instance construction, override this method. It will be automatically called indirectly
    from `__post_init__`.
    """
    pass
        
  def update(self, rngs=None, variables=None, **kwargs):
    """Construct a new module instance based on this one, with overrides."""
    attrs = dataclasses.asdict(self)
    del attrs['parent']
    return self.__class__.toplevel(**attrs, rngs=rngs, variables=variables, **kwargs)

  @contextmanager
  def mutate(self, mutable):

    cloned = self.update()
    try:
      cloned.variables = _unfreeze_variables(cloned.variables, mutable)
      yield cloned
    finally:
      cloned.variables = freeze(cloned.variables)

  # QUESTION: Should this be a property? Or should it be assigned
  # to `self.variables` during __post_init__?
  def variables(self):
    return self.scope.variables

  def param(self, name, init_fun, shape):
    return self.scope.param(name, init_fun, shape)
    

  # TODO: Methods to access non-parameter variables from scope.

