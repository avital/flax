from dataclasses import dataclass
import dataclasses
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
# TODO: Make _unfreeze_variables public?
from flax.core.scope import Scope, _unfreeze_variables
import functools
import collections.abc

from flax.core.frozen_dict import freeze, unfreeze

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
  parent: Optional[Type["Module"]]
  
  def __post_init__(self):      
    self.submodules = {}

    # See public_module_methods.py
    self._autoname_state = None
    # See forbid_reuse_by_name.py.
    self._method_by_name = {}
    self._current_method = None

    if self.parent is None:
      if self.rngs is None:
        self.rngs = {}
      if self.variables is None:
        self.variables = {'params': {}}

      # QUESTION: Is it odd that we unfreeze here?
      # This was needed for the "TiedAutoencoder" example
      self.variables = unfreeze(self.variables)
      self.scope = Scope(self.variables, self.rngs)
      self._setup()
    else:
      assert self.variables is None
      assert self.rngs is None

      if self.parent._in_setup:
        # defer naming and registration on parent until __setattr__.
        return

      if self.name is not None:
        self._register_as_submodule(self, self.name)
        self._setup()
      elif self.parent._autoname_state is not None:
        self._register_as_submodule(f"{self.parent._autoname_state['prefix']}{self.__class__.__name__}/{self.parent._autoname_state['cursor']}")
        self._setup()
        self.parent._autoname_state['cursor'] += 1
      else:
        raise ValueError(
            "Can't call private module method that constructs submodules without explicit names.")

  def _register_as_submodule(self, name):
    """Register self as a child of self.parent."""
    if name in self.parent._method_by_name:
      raise ValueError(f"Trying to share submodule by name in methods {self.parent._current_method} "
                        f"and {self.parent._method_by_name[name]}. To share submodules, store "
                        f"module instances as a Python object and reuse. You can store module"
                        f"instance on `self` to share across methods.")

    self.name = name
    self.parent.submodules[name] = self
    self.parent._method_by_name[self.name] = self.parent._current_method
    self.scope = self.parent.scope.push(name)
    self.variables = self.scope._variables
    self.rngs = self.scope.rngs

  def _setup(self):
    self._in_setup = True
    self.setup()
    self.variables = freeze(self.variables)
    self._in_setup = False

  def setup(self):
    """Called when module instance receives variables and PRNGs.
    
    If you want to use PRNGs and/or read or write variables during module
    instance construction, override this method. It will be automatically called indirectly
    from `__post_init__`
    
    Generally, submodules can be created lazily within module methods. But if you want to share submodules
    across methods, you need to register them in `setup`. Submodules assigned to `self` within `setup()` will
    be registered by their attribute name. You can assign register lists and dicts of submodules by
    assigning them to `self`. Submodules must be registered before they can be used.
    """
    pass
        
  def clone(self):
    attrs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
    return self.__class__(**attrs)

  @contextmanager
  def mutate(self, mutable=True):
    cloned = self.clone()
    try:
      cloned.scope._variables = _unfreeze_variables(cloned.scope._variables, mutable)
      # TODO: It is pretty annoying that we have to keep changing variables every time we 
      # change scope.
      cloned.variables = cloned.scope._variables
      yield cloned
    finally:
      cloned.scope._variables = freeze(cloned.scope._variables)
      cloned.variables = cloned.scope._variables

  # TODO: Should initialized() take in rngs?
  def initialized(self, *args, method=lambda self: self.__call__, **kwargs):
    with self.mutate() as initialized:
      method(initialized)(*args, **kwargs)
    return initialized

  def _assert_has_scope(self):
    if not hasattr(self, "scope"):
      assert self.parent._in_setup
      raise ValueError("Must assign module to self before use.")

  # TODO: Not sure I like this, stack traces are ugly. Is there a better solution?
  def __getattr__(self, name):
    if name == 'variables' or name == 'submodules':
      return self.__getattribute__(name)

    if self.variables and name in self.variables:
      return self.variables[name]
    elif name in self.submodules:
      return self.submodules[name]
    else:
      return self.__getattribute__(name)

  def __setattr__(self, name, val):
    # Three possible states:

    if not hasattr(self, '_in_setup'):
      # 1. During dataclass __init__
      super().__setattr__(name, val)

    elif self._in_setup:
      # 2. Within setup()

      def _setup_submodule(submodule, name):
        if submodule.parent != self:
          raise ValueError("Submodules created in setup must have parent=self, e.g. `Dense(self, features=3)`.")
        if submodule.name is not None:
          raise ValueError("During `setup`, submodule names are given by assigning to attributes on `self`")
        submodule._register_as_submodule(name)
        submodule._setup()

      if isinstance(val, Module):
        _setup_submodule(val, name)
      elif isinstance(val, collections.abc.Sequence):
        for i, subval in enumerate(val):
          if isinstance(subval, Module):
            _setup_submodule(subval, f"{name}/{i}")

      # TODO: Support nested sequences? dicts?

      super().__setattr__(name, val)

    else:
      # TODO: This can't work yet because of the dynamically scoped variables we use
      # e.g. `self._autoname_state`. We can fix this by storing all of these on
      # `self._internal` 
      #
      # ANOTHER QUESTION: If we do this, should we add a `self.scratchpad` for small tinkering
      # with instances in a way that you /know/ won't work outside a JIT boundary?
      #
      # 3. After setup() -- in other methods or outer modules
      # raise ValueError(
      #     "Module instances are immutable outside of setup(). This is necessary to ensure "
      #     "correctness with JAX transformations like jax.jit. Consider using parameters: XXX")
      super().__setattr__(name, val)


  def param(self, name, init_fun, shape):
    self._assert_has_scope()
    return self.scope.param(name, init_fun, shape)

  def variable(self, kind, name, default_fun, shape):
    self._assert_has_scope()
    return self.scope.variable(kind, name, default_fun, shape)

