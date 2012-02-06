#
# OWTools.py
# tools for Visual Orange
#

TRUE=1
FALSE=0

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from random import *

try:
    import win32api, win32con
    t = win32api.RegOpenKey(win32con.HKEY_LOCAL_MACHINE, "SOFTWARE\\Python\\PythonCore\\%i.%i\\PythonPath\\Orange" % sys.version_info[:2], 0, win32con.KEY_READ)
    t = win32api.RegQueryValueEx(t, "")[0]
    orangedir = t[:t.find("orange")] + "orange"
except:
    import os
    orangedir = os.getcwd()
    if orangedir[-12:] == "OrangeCanvas":
        orangedir = orangedir[:-13]

if not os.path.exists(orangedir+"/orngStat.py"):
    orangedir = None

def getHtmlCompatibleString(strVal):
    return strVal.replace("<=", "&#8804;").replace(">=","&#8805;").replace("<", "&#60;").replace(">","&#62;").replace("=\\=", "&#8800;")



def domainPurger(examples, purgeClasses):
    import orange
    newDomain = orange.RemoveUnusedValues(removeOneValued=True)(examples, 0, True, purgeClasses)
    if newDomain != examples.domain:
        return orange.ExampleTable(newDomain, examples)
    return examples
