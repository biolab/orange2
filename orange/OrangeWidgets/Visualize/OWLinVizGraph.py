#
# OWRadvizGraph.py
#
# the base for all parallel graphs

from OWVisGraph import *
from copy import copy
import time
from operator import add
import orange, orngDimRed
import math, Numeric, LinearAlgebra

###########################################################################################
##### CLASS : OWRADVIZGRAPH
###########################################################################################
class OWLinVizGraph(OWVisGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)

    # ####################################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, labels, **args):
        self.removeCurves()
        self.removeMarkers()
        #self.tips.removeAll()

        self.__dict__.update(args)

        self.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
        self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        scaleDraw = self.axisScaleDraw(QwtPlot.xBottom)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
        scaleDraw = self.axisScaleDraw(QwtPlot.yLeft)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
                
        self.setAxisScale(QwtPlot.xBottom, -1.22, 1.22, 1)
        self.setAxisScale(QwtPlot.yLeft, -1.13, 1.13, 1)

        length = len(labels)
        xs = []
        self.dataMap = {}
        indices = []
        self.anchorData = []

               

    def onMouseMoved(self, e):
        """
        for key in self.tooltipCurveKeys:  self.removeCurve(key)
        for marker in self.tooltipMarkers: self.removeMarker(marker)
        self.tooltipCurveKeys = []
        self.tooltipMarkers = []
        
        x = self.invTransform(QwtPlot.xBottom, e.x())
        y = self.invTransform(QwtPlot.yLeft, e.y())
        dictValue = "%.1f-%.1f"%(x, y)
        if self.dataMap.has_key(dictValue):
            points = self.dataMap[dictValue]
            dist = 100.0
            nearestPoint = ()
            for (x_i, y_i, color, data) in points:
                if abs(x-x_i)+abs(y-y_i) < dist:
                    dist = abs(x-x_i)+abs(y-y_i)
                    nearestPoint = (x_i, y_i, color, data)
           
            if dist < 0.05:
                x_i = nearestPoint[0]; y_i = nearestPoint[1]; color = nearestPoint[2]; data = nearestPoint[3]
                for (xAnchor,yAnchor,label) in self.anchorData:
                    # draw lines
                    key = self.addCurve("Tooltip curve", color, color, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [x_i, xAnchor], yData = [y_i, yAnchor])
                    self.tooltipCurveKeys.append(key)

                    # draw text
                    marker = self.addMarker(str(data[self.attributeNames.index(label)].value), (x_i + xAnchor)/2.0, (y_i + yAnchor)/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
                    font = self.markerFont(marker)
                    font.setPointSize(12)
                    self.setMarkerFont(marker, font)

                    self.tooltipMarkers.append(marker)

        """
        OWVisGraph.onMouseMoved(self, e)
        #self.update()

if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWLinVizGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
