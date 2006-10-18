# 
# OWTools.py
# tools for Visual Orange
#

TRUE=1
FALSE=0

from qt import *
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
class ColorPixmap (QPixmap):
    def __init__(self,color=Qt.white, size = 10):
        "Creates a single-color pixmap"
        QPixmap.__init__(self,size,size)
        self.color = color
        self.fill(color)

  
class QPointFloat:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class QRectFloat:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        

#A dynamic tool tip class      
class TooltipManager:
    # Creates a new dynamic tool tip.
    def __init__(self, qwtplot):
        self.qwtplot = qwtplot
        self.positions=[]
        self.texts=[]

    # Adds a tool tip. If a tooltip with the same name already exists, it updates it instead of adding a new one.
    def addToolTip(self,x, y,text, customX = 0, customY = 0):
        self.positions.append((x,y, customX, customY))
        self.texts.append(text)

    #Decides whether to pop up a tool tip and which text to pop up
    def maybeTip(self, x, y):
        if len(self.positions) == 0: return ("", -1, -1)
        dists = [abs(x-position[0]) + abs(y-position[1]) for position in self.positions]
        nearestIndex = dists.index(min(dists))

        intX = abs(self.qwtplot.transform(self.qwtplot.xBottom, x) - self.qwtplot.transform(self.qwtplot.xBottom, self.positions[nearestIndex][0]))
        intY = abs(self.qwtplot.transform(self.qwtplot.yLeft, y) - self.qwtplot.transform(self.qwtplot.yLeft, self.positions[nearestIndex][1]))
        if self.positions[nearestIndex][2] == 0 and self.positions[nearestIndex][3] == 0:   # if we specified no custom range then assume 6 pixels
            if intX + intY < 6:  return (self.texts[nearestIndex], self.positions[nearestIndex][0], self.positions[nearestIndex][1])
            else:                return ("", None, None)
        else:
            if abs(self.positions[nearestIndex][0] - x) <= self.positions[nearestIndex][2] and abs(self.positions[nearestIndex][1] - y) <= self.positions[nearestIndex][3]:
                return (self.texts[nearestIndex], x, y)
            else:
                return ("", None, None)
                
    def removeAll(self):
        self.positions = []
        self.texts = []
