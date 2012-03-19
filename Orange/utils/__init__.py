"""
.. index:: utils

Orange.utils contains developer utilities.

------------------
Reporting progress
------------------

.. autoclass:: Orange.utils.ConsoleProgressBar
    :members:

-----------------------------
Deprecation utility functions
-----------------------------

.. autofunction:: Orange.utils.deprecation_warning

.. autofunction:: Orange.utils.deprecated_members

.. autofunction:: Orange.utils.deprecated_keywords

.. autofunction:: Orange.utils.deprecated_attribute

.. autofunction:: Orange.utils.deprecated_function_name

----------------
Other submodules
----------------

..automodule:: Orange.utils.environ

"""

"""
__all__ = ["deprecated_members", "deprecated_keywords",
           "deprecated_attribute", "deprecation_warning",
           "deprecated_function_name"]
"""

import environ

import warnings
def deprecation_warning(old, new, stacklevel=-2):
    """ Raise a deprecation warning of an obsolete attribute access.
    
    :param old: Old attribute name (used in warning message).
    :param new: New attribute name (used in warning message).
    
    """
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
    
    :param in_place: If True the class will be modified in place, otherwise
        it will be subclassed (default True).
    :type in_place: bool
    
    Example ::
            
        >>> class A(object):
        ...     def __init__(self, foo_bar="bar"):
        ...         self.set_foo_bar(foo_bar)
        ...     
        ...     def set_foo_bar(self, foo_bar="bar"):
        ...         self.foo_bar = foo_bar
        ...
        ... A = deprecated_members(
        ... {"fooBar": "foo_bar", 
        ...  "setFooBar":"set_foo_bar"},
        ... wrap_methods=["set_foo_bar", "__init__"])(A)
        ... 
        ...
        >>> a = A(fooBar="foo")
        __main__:1: DeprecationWarning: 'fooBar' is deprecated. Use 'foo_bar' instead!
        >>> print a.fooBar, a.foo_bar
        foo foo
        >>> a.setFooBar("FooBar!")
        __main__:1: DeprecationWarning: 'setFooBar' is deprecated. Use 'set_foo_bar' instead!
        
    .. note:: This decorator does nothing if \
        :obj:`Orange.utils.environ.orange_no_deprecated_members` environment \
        variable is set to `True`.
        
    """
    if environ.orange_no_deprecated_members:
        return lambda cls: cls
    
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
        
    .. note:: This decorator does nothing if \
        :obj:`Orange.utils.environ.orange_no_deprecated_members` environment \
        variable is set to `True`.
        
    """
    if environ.orange_no_deprecated_members:
        return lambda func: func
    for name in name_map.values():
        if name in name_map:
            raise ValueError("Deprecation keys and values overlap; this could"
                             " cause trouble!")
    
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

    ..

        >>> sys.stderr = sys.stdout
    
    Example ::
    
        >>> class A(object):
        ...     def __init__(self):
        ...         self.my_attr = "123"
        ...     myAttr = deprecated_attribute("myAttr", "my_attr")
        ...
        ...
        >>> a = A()
        >>> print a.myAttr
        ...:1: DeprecationWarning: 'myAttr' is deprecated. Use 'my_attr' instead!
        123
        
    .. note:: This decorator does nothing and returns None if \
        :obj:`Orange.utils.environ.orange_no_deprecated_members` environment \
        variable is set to `True`.
        
    """
    if environ.orange_no_deprecated_members:
        return None
    
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

class class_property(object):
    def __init__(self, fget=None, fset=None, fdel=None, doc="class property"):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc
        
    def __get__(self, instance, owner):
        if instance is None:
            return self.fget(owner)
        else:
            return self.fget(instance)                
            
def deprecated_class_attribute(old_name, new_name):
    """ Return a property object that accesses an class attribute
    named `new_name` and raises a deprecation warning when doing so.
    
    """
    if environ.orange_no_deprecated_members:
        return None
    
    def fget(self):
        deprecation_warning(old_name, new_name, stacklevel=3)
        return getattr(self, new_name)
        
    prop = class_property(fget,
                    doc="A deprecated class member '%s'. Use '%s' instead." % (old_name, new_name))
    return prop

def deprecated_function_name(func):
    """ Return a wrapped function that raises an deprecation warning when
    called. This should be used for deprecation of module level function names. 
    
    Example ::
    
        >>> def func_a(arg):
        ...    print "This is func_a  (used to be named funcA) called with", arg
        ...
        ...
        >>> funcA = deprecated_function_name(func_a)
        >>> funcA(None)
          
    
    .. note:: This decorator does nothing and if \
        :obj:`Orange.utils.environ.orange_no_deprecated_members` environment \
        variable is set to `True`.
        
    """
    if environ.orange_no_deprecated_members:
        return func
    
    @wraps(func)
    def wrapped(*args, **kwargs):
        warnings.warn("Deprecated function name. Use %r instead!" % func.__name__,
                      DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapped
    
class ConsoleProgressBar(object):
    """ A class to for printing progress bar reports in the console.
    
    Example ::
    
        >>> import sys, time
        >>> progress = ConsoleProgressBar("Example", output=sys.stdout)
        >>> for i in range(100):
        ...    progress.advance()
        ...    # Or progress.set_state(i)
        ...    time.sleep(0.01)
        ...
        ...
        Example ===================================>100%
        
    """
    def __init__(self, title="", charwidth=40, step=1, output=None):
        """ Initialize the progress bar.
        
        :param title: The title for the progress bar.
        :type title: str
        :param charwidth: The maximum progress bar width in characters.
        
            .. todo:: Get the console width from the ``output`` if the
                information can be retrieved. 
                
        :type charwidth: int
        :param step: A default step used if ``advance`` is called without
            any  arguments
        
        :type step: int
        :param output: The output file. If None (default) then ``sys.stderr``
            is used.
            
        :type output: An file like object to print the progress report to.
         
        """
        self.title = title + " "
        self.charwidth = charwidth
        self.step = step
        self.currstring = ""
        self.state = 0
        if output is None:
            output = sys.stderr
        self.output = output

    def clear(self, i=-1):
        """ Clear the current progress line indicator string.
        """
        try:
            if hasattr(self.output, "isatty") and self.output.isatty():
                self.output.write("\b" * (i if i != -1 else len(self.currstring)))
            else:
                self.output.seek(-i if i != -1 else -len(self.currstring), 2)
        except Exception: ## If for some reason we failed 
            self.output.write("\n")

    def getstring(self):
        """ Return the progress indicator string.
        """
        progchar = int(round(float(self.state) * (self.charwidth - 5) / 100.0))
        return self.title + "=" * (progchar) + ">" + " " * (self.charwidth\
            - 5 - progchar) + "%3i" % int(round(self.state)) + "%"

    def printline(self, string):
        """ Print the ``string`` to the output file.
        """
        try:
            self.clear()
            self.output.write(string)
            self.output.flush()
        except Exception:
            pass
        self.currstring = string

    def __call__(self, newstate=None):
        """ Set the ``newstate`` as the current state of the progress bar.
        ``newstate`` must be in the interval [0, 100].
        
        .. note:: ``set_state`` is the prefered way to set a new steate. 
        
        :param newstate: The new state of the progress bar.
        :type newstate: float
         
        """
        if newstate is None:
            self.advance()
        else:
            self.set_state(newstate)
            
    def set_state(self, newstate):
        """ Set the ``newstate`` as the current state of the progress bar.
        ``newstate`` must be in the interval [0, 100]. 
        
        :param newstate: The new state of the progress bar.
        :type newstate: float
        
        """
        if int(newstate) != int(self.state):
            self.state = newstate
            self.printline(self.getstring())
        else:
            self.state = newstate
            
    def advance(self, step=None):
        """ Advance the current state by ``step``. If ``step`` is None use
        the default step as set at class initialization.
          
        """
        if step is None:
            step = self.step
            
        newstate = self.state + step
        self.set_state(newstate)

    def finish(self):
        """ Finish the progress bar (i.e. set the state to 100 and
        print the final newline to the ``output`` file).
        """
        self.__call__(100)
        self.output.write("\n")

def progress_bar_milestones(count, iterations=100):
    return set([int(i*count/float(iterations)) for i in range(iterations)])

progressBarMilestones = deprecated_function_name(progress_bar_milestones)
