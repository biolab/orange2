
try:
    import sys as __sys    
    import os as __os
    import imp as __imp
    from distutils.sysconfig import get_config_var as __get_config_var
    __ORANGENE_SO__FILE__ = __os.path.dirname(__os.path.dirname(__os.path.abspath(__file__)))
    __ORANGENE_SO__FILE__ = __os.path.join(__ORANGENE_SO__FILE__, "orangene" + __get_config_var("SO"))
    if __ORANGENE_SO__FILE__ != __sys.modules["orangene"].__file__:
        __ORANGENE = __imp.load_dynamic("orangene", __ORANGENE_SO__FILE__)
        __sys.modules['orangene'] = __ORANGENE
except Exception, ex:
    from Orange.orangene import *