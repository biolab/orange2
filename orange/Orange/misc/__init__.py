"""

.. index:: misc

Module Orange.misc contains common functions and classes which are used in other modules.

==================
Counters
==================

.. index:: misc
.. index::
   single: misc; counters

.. automodule:: Orange.misc.counters
  :members:

==================
Render
==================

.. index:: misc
.. index::
   single: misc; render

.. automodule:: Orange.misc.render
  :members:

==================
Selection
==================

.. index:: selection
.. index::
   single: misc; selection

Many machine learning techniques generate a set different solutions or have to
choose, as for instance in classification tree induction, between different
features. The most trivial solution is to iterate through the candidates,
compare them and remember the optimal one. The problem occurs, however, when
there are multiple candidates that are equally good, and the naive approaches
would select the first or the last one, depending upon the formulation of
the if-statement.

:class:`Orange.misc.selection` provides a class that makes a random choice
in such cases. Each new candidate is compared with the currently optimal
one; it replaces the optimal if it is better, while if they are equal,
one is chosen by random. The number of competing optimal candidates is stored,
so in this random choice the probability to select the new candidate (over the
current one) is 1/w, where w is the current number of equal candidates,
including the present one. One can easily verify that this gives equal
chances to all candidates, independent of the order in which they are presented.

.. automodule:: Orange.misc.selection
  :members:

Example
--------

The following snippet loads the data set lymphography and prints out the
feature with the highest information gain.

part of `misc-selection-bestonthefly.py`_ (uses `lymphography.tab`_)

.. literalinclude:: code/misc-selection-bestonthefly.py
  :lines: 7-16

Our candidates are tuples gain ratios and features, so we set
:obj:`callCompareOn1st` to make the compare function compare the first element
(gain ratios). We could achieve the same by initializing the object like this:

part of `misc-selection-bestonthefly.py`_ (uses `lymphography.tab`_)

.. literalinclude:: code/misc-selection-bestonthefly.py
  :lines: 18-18


The other way to do it is through indices.

`misc-selection-bestonthefly.py`_ (uses `lymphography.tab`_)

.. literalinclude:: code/misc-selection-bestonthefly.py
  :lines: 25-

.. _misc-selection-bestonthefly.py: code/misc-selection-bestonthefly.py.py
.. _lymphography.tab: code/lymphography.tab

Here we only give gain ratios to :obj:`bestOnTheFly`, so we don't have to specify a
special compare operator. After checking all features we get the index of the 
optimal one by calling :obj:`winnerIndex`.

==================
Server files
==================

.. index:: server files

.. automodule:: Orange.misc.serverfiles

"""

import counters
import selection
import render
import serverfiles

__all__ = ["counters", "selection", "render", "serverfiles",
           "deprecated_members", "deprecated_keywords",
           "deprecated_attribute", "deprecation_warning"]

import random, types, sys
import time

def getobjectname(x, default=""):
    if type(x)==types.StringType:
        return x
      
    for i in ["name", "shortDescription", "description", "func_doc", "func_name"]:
        if getattr(x, i, ""):
            return getattr(x, i)

    if hasattr(x, "__class__"):
        r = repr(x.__class__)
        if r[1:5]=="type":
            return str(x.__class__)[7:-2]
        elif r[1:6]=="class":
            return str(x.__class__)[8:-2]
    return default


def demangleExamples(x):
    if type(x)==types.TupleType:
        return x
    else:
        return x, 0


def frange(*argw):
    start, stop, step = 0.0, 1.0, 0.1
    if len(argw)==1:
        start=step=argw[0]
    elif len(argw)==2:
        stop, step = argw
    elif len(argw)==3:
        start, stop, step = argw
    elif len(argw)>3:
        raise AttributeError, "1-3 arguments expected"

    stop+=1e-10
    i=0
    res=[]
    while 1:
        f=start+i*step
        if f>stop:
            break
        res.append(f)
        i+=1
    return res

verbose = 0

def printVerbose(text, *verb):
    if len(verb) and verb[0] or verbose:
        print text

class ConsoleProgressBar(object):
    def __init__(self, title="", charwidth=40, step=1, output=sys.stderr):
        self.title = title + " "
        self.charwidth = charwidth
        self.step = step
        self.currstring = ""
        self.state = 0
        self.output = output

    def clear(self, i=-1):
        try:
            if hasattr(self.output, "isatty") and self.output.isatty():
                self.output.write("\b" * (i if i != -1 else len(self.currstring)))
            else:
                self.output.seek(-i if i != -1 else -len(self.currstring), 2)
        except Exception: ## If for some reason we failed 
            self.output.write("\n")

    def getstring(self):
        progchar = int(round(float(self.state) * (self.charwidth - 5) / 100.0))
        return self.title + "=" * (progchar) + ">" + " " * (self.charwidth\
            - 5 - progchar) + "%3i" % int(round(self.state)) + "%"

    def printline(self, string):
        try:
            self.clear()
            self.output.write(string)
            self.output.flush()
        except Exception:
            pass
        self.currstring = string

    def __call__(self, newstate=None):
        if newstate == None:
            newstate = self.state + self.step
        if int(newstate) != int(self.state):
            self.state = newstate
            self.printline(self.getstring())
        else:
            self.state = newstate

    def finish(self):
        self.__call__(100)
        self.output.write("\n")

def progressBarMilestones(count, iterations=100):
    return set([int(i*count/float(iterations)) for i in range(iterations)])

def lru_cache(maxsize=100):
    """ A least recently used cache function decorator.
    (Similar to the functools.lru_cache in python 3.2)
    """
    
    def decorating_function(func):
        import functools
        cache = {}
        
        functools.wraps(func)
        def wrapped(*args, **kwargs):
            key = args + tuple(sorted(kwargs.items()))
            if key not in cache:
                res = func(*args, **kwargs)
                cache[key] = (time.time(), res)
                if len(cache) > maxsize:
                    key, (_, _) = min(cache.iteritems(), key=lambda item: item[1][0])
                    del cache[key]
            else:
                _, res = cache[key]
                cache[key] = (time.time(), res) # update the time
                
            return res
        
        def clear():
            cache.clear()
        
        wrapped.clear = clear
        
        return wrapped
    return decorating_function


"""\
Deprecation utility functions.

"""

import warnings
def deprecation_warning(old, new, stacklevel=-2):
    warnings.warn("'%s' is deprecated. Use '%s' instead!" % (old, new), DeprecationWarning, stacklevel=stacklevel)
   
# We need to get the instancemethod type 
class _Foo():
    def bar(self):
        pass
instancemethod = type(_Foo.bar)
del _Foo

function = type(lambda: None)

class universal_set(set):
    """ A universal set, pretends it contains everything.
    """
    def __contains__(self, value):
        return True
    
from functools import wraps

def deprecated_members(name_map, wrap_methods="all", in_place=True):
    """ Decorate a class with properties for accessing attributes, and methods
    with deprecated names. In addition methods from the `wrap_methods` list
    will be wrapped to receive mapped keyword arguments.
    
    :param name_map: A dictionary mapping old into new names.
    :type name_map: dict
    
    :param wrap_methods: A list of method names to wrap. Wrapped methods will
        be called with mapped keyword arguments (by default all methods will
        be wrapped).
    :type wrap_methods: list
    
    Example ::
            
        >>> @deprecated_members({"fooBar": "foo_bar", "setFooBar":"set_foo_bar"},
        ...                    wrap_methods=["set_foo_bar", "__init__"])
        ... class A(object):
        ...     def __init__(self, foo_bar="bar"):
        ...         self.set_foo_bar(foo_bar)
        ...     
        ...     def set_foo_bar(self, foo_bar="bar"):
        ...         self.foo_bar = foo_bar
        ...         
        ...
        >>> a = A(fooBar="foo")
        __main__:1: DeprecationWarning: 'fooBar' is deprecated. Use 'foo_bar' instead!
        >>> print a.fooBar, a.foo_bar
        foo foo
        >>> a.setFooBar("FooBar!")
        __main__:1: DeprecationWarning: 'setFooBar' is deprecated. Use 'set_foo_bar' instead!
        
    """
    def is_wrapped(method):
        """ Is member method already wrapped.
        """
        if getattr(method, "_deprecate_members_wrapped", False):
            return True
        elif hasattr(method, "im_func"):
            im_func = method.im_func
            return getattr(im_func, "_deprecate_members_wrapped", False)
        else:
            return False
        
    if wrap_methods == "all":
        wrap_methods = universal_set()
    elif not wrap_methods:
        wrap_methods = set()
        
    def wrapper(cls):
        cls_names = {}
        # Create properties for accessing deprecated members
        for old_name, new_name in name_map.items():
            cls_names[old_name] = deprecated_attribute(old_name, new_name)
            
        # wrap member methods to map keyword arguments
        for key, value in cls.__dict__.items():
            if isinstance(value, (instancemethod, function)) \
                and not is_wrapped(value) and key in wrap_methods:
                
                wrapped = deprecated_keywords(name_map)(value)
                wrapped._deprecate_members_wrapped = True # A flag indicating this function already maps keywords
                cls_names[key] = wrapped
        if in_place:
            for key, val in cls_names.items():
                setattr(cls, key, val)
            return cls
        else:
            return type(cls.__name__, (cls,), cls_names)
        
    return wrapper

def deprecated_keywords(name_map):
    """ Deprecates the keyword arguments of the function.
    
    Example ::
    
        >>> @deprecated_keywords({"myArg": "my_arg"})
        ... def my_func(my_arg=None):
        ...     print my_arg
        ...
        ...
        >>> my_func(myArg="Arg")
        __main__:1: DeprecationWarning: 'myArg' is deprecated. Use 'my_arg' instead!
        Arg
        
    """
    def decorator(func):
        @wraps(func)
        def wrap_call(*args, **kwargs):
            kwargs = dict(kwargs)
            for name in name_map:
                if name in kwargs:
                    deprecation_warning(name, name_map[name], stacklevel=3)
                    kwargs[name_map[name]] = kwargs[name]
                    del kwargs[name]
            return func(*args, **kwargs)
        return wrap_call
    return decorator

def deprecated_attribute(old_name, new_name):
    """ Return a property object that accesses an attribute named `new_name`
    and raises a deprecation warning when doing so.
    
    Example ::
    
        >>> class A(object):
        ...     def __init__(self):
        ...         self.my_attr = "123"
        ...     myAttr = deprecated_attribute("myAttr", "my_attr")
        ...
        ...
        >>> a = A()
        >>> print a.myAttr
        __main__:1: DeprecationWarning: 'myAttr' is deprecated. Use 'my_attr' instead!
        123
        
    """
    def fget(self):
        deprecation_warning(old_name, new_name, stacklevel=3)
        return getattr(self, new_name)
    
    def fset(self, value):
        deprecation_warning(old_name, new_name, stacklevel=3)
        setattr(self, new_name, value)
    
    def fdel(self):
        deprecation_warning(old_name, new_name, stacklevel=3)
        delattr(self, new_name)
    
    prop = property(fget, fset, fdel,
                    doc="A deprecated member '%s'. Use '%s' instead." % (old_name, new_name))
    return prop
    
def _test():
    import doctest
    doctest.testmod()
    
if __name__ == "__main__":
    _test()