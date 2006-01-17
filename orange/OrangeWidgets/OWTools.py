# 
# OWTools.py
# tools for Visual Orange
#

TRUE=1
FALSE=0

from qt import *
from random import *


def getHtmlCompatibleString(strVal):
    strVal = strVal.replace("<", "&#60;")
    strVal = strVal.replace(">", "&#62;")
    return strVal


#A 10X10 single color pixmap
class ColorPixmap (QPixmap):
    def __init__(self,color=Qt.white):
        "Creates a single-color pixmap"
        QPixmap.__init__(self,10,10)
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
    def addToolTip(self,x, y,text):
        self.positions.append((x,y))
        self.texts.append(text)

    #Decides whether to pop up a tool tip and which text to pop up
    def maybeTip(self, x, y):
        if len(self.positions) == 0: return ("", -1, -1)
        dists = [abs(x-position[0]) + abs(y-position[1]) for position in self.positions]
        nearestIndex = dists.index(min(dists))

        intX = abs(self.qwtplot.transform(self.qwtplot.xBottom, x) - self.qwtplot.transform(self.qwtplot.xBottom, self.positions[nearestIndex][0]))
        intY = abs(self.qwtplot.transform(self.qwtplot.xBottom, y) - self.qwtplot.transform(self.qwtplot.xBottom, self.positions[nearestIndex][1]))
        if intX + intY < 6:
            return (self.texts[nearestIndex], self.positions[nearestIndex][0], self.positions[nearestIndex][1])
        else:
            return ("", None, None)
                
    def removeAll(self):
        self.positions = []
        self.texts = []
