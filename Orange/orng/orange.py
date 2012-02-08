
try:
    import sys as __sys    
    import os as __os
    import imp as __imp
    from distutils.sysconfig import get_config_var as __get_config_var
    __ORANGE_SO__FILE__ = __os.path.dirname(__os.path.dirname(__os.path.abspath(__file__)))
    __ORANGE_SO__FILE__ = __os.path.join(__ORANGE_SO__FILE__, "orange" + __get_config_var("SO"))
    if __ORANGE_SO__FILE__ != __sys.modules["orange"].__file__:
        __ORANGE = __imp.load_dynamic("orange", __ORANGE_SO__FILE__)
        __sys.modules['orange'] = __ORANGE
except Exception, ex:
    from Orange.orange import *
