import functools

def autonames(fun, prefix=''):
  @functools.wraps(fun)
  def wrapped(self, *args, **kwargs):
    if prefix in self._autoname_funs and self._autoname_funs[prefix] != fun:
      raise ValueError(
        "Can only decorate one method with @autonames. To reuse submodules across methods, "
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
      # TODO: This is wrong if we want to allow nested calls to autonames methods
      # with different prefixes
      self._autoname_cursor = None
      self._autoname_prefix = None

  return wrapped

autonames.prefix = lambda prefix: functools.partial(autonames, prefix=prefix)