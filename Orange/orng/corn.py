
try:
    import sys as __sys    
    import os as __os
    import imp as __imp
    from distutils.sysconfig import get_config_var as __get_config_var
    __CORN_SO__FILE__ = __os.path.dirname(__os.path.dirname(__os.path.abspath(__file__)))
    __CORN_SO__FILE__ = __os.path.join(__CORN_SO__FILE__, "corn" + __get_config_var("SO"))
    if __CORN_SO__FILE__ != __sys.modules["corn"].__file__:
        __CORN = __imp.load_dynamic("corn", __CORN_SO__FILE__)
        __sys.modules['corn'] = __CORN
except Exception, ex:
    from Orange.corn import *