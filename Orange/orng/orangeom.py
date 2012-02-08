
try:
    import sys as __sys    
    import os as __os
    import imp as __imp
    from distutils.sysconfig import get_config_var as __get_config_var
    __ORANGEOM_SO__FILE__ = __os.path.dirname(__os.path.dirname(__os.path.abspath(__file__)))
    __ORANGEOM_SO__FILE__ = __os.path.join(__ORANGEOM_SO__FILE__, "orangeom" + __get_config_var("SO"))
    if __ORANGEOM_SO__FILE__ != __sys.modules["orangeom"].__file__:
        __ORANGEOM = __imp.load_dynamic("orangeom", __ORANGEOM_SO__FILE__)
        __sys.modules['orangeom'] = __ORANGEOM
except Exception, ex:
    from Orange.orangeom import *