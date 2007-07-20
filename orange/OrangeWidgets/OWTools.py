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
    strVal = strVal.replace("<", "&#60;")
    strVal = strVal.replace(">", "&#62;")
    return strVal


#A 10X10 single color pixmap
class ColorPixmap (QIcon):
    def __init__(self,color=QColor(Qt.white), size = 10):
        "Creates a single-color pixmap"
        p = QPixmap(size,size)
        p.fill(color)
        self.color = color
        QIcon.__init__(self, p)



