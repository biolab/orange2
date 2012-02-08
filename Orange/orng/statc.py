
try:
    import sys as __sys    
    import os as __os
    import imp as __imp
    from distutils.sysconfig import get_config_var as __get_config_var
    __STATC_SO__FILE__ = __os.path.dirname(__os.path.dirname(__os.path.abspath(__file__)))
    __STATC_SO__FILE__ = __os.path.join(__STATC_SO__FILE__, "statc" + __get_config_var("SO"))
    if __STATC_SO__FILE__ != __sys.modules["statc"].__file__:
        __STATC = __imp.load_dynamic("statc", __STATC_SO__FILE__)
        __sys.modules['statc'] = __STATC
except Exception, ex:
    from Orange.statc import *