import functools

def public(fun):
  @functools.wraps(fun)
  def wrapped(self, *args, **kwargs):
    if fun.__name__ == "__call__":
      prefix = ""
    else:
      prefix = f"fun.__name__/" 

    # "Rewind" the autonaming process.
    # NOTE: self._autoname_state is dyamically scoped (but per-instance, not global)
    # CONSIDER: Is Anselm right that we should keep the whole stack here for error
    # handling/debugability? (One cool thing is that if we have the Module parent
    # stack then we have all the hyperparameters due to dataclasses which means it
    # would print nicely -- though need to think about how the full parameter dicts
    # would be shown.)
    outer_autoname_state = self._autoname_state
    self._autoname_state = {"prefix": prefix, "cursor": 0}
    try:
      return fun(self, *args, **kwargs)
    finally:
      self._autoname_state = outer_autoname_state

  return wrapped
