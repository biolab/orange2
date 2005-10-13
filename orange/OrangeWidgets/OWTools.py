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
        nearestIndex = 0
        dist = abs(x-self.positions[0][0]) + abs(y-self.positions[0][1])
        for i in range(1, len(self.positions)):
            ithDist = abs(x-self.positions[i][0]) + abs(y-self.positions[i][1])
            if ithDist < dist:
                nearestIndex = i
                dist = ithDist

        intX = abs(self.qwtplot.transform(self.qwtplot.xBottom, x) - self.qwtplot.transform(self.qwtplot.xBottom, self.positions[nearestIndex][0]))
        intY = abs(self.qwtplot.transform(self.qwtplot.xBottom, y) - self.qwtplot.transform(self.qwtplot.xBottom, self.positions[nearestIndex][1]))
        if intX + intY < 6:
            return (self.texts[nearestIndex], self.positions[nearestIndex][0], self.positions[nearestIndex][1])
        else:
            return ("", None, None)
                
    def removeAll(self):
        self.positions = []
        self.texts = []
