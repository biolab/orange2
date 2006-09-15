"""
graph for correspondence analysis
"""

BROWSE_RECTANGLE = 4
BROWSE_CIRCLE= 5


from OWGraph import *
from math import sqrt
from numpy import arange, sign
import operator

class OWCorrAnalysisGraph(OWGraph):
    def __init__(self, parent = None, name = "None"):
        OWGraph.__init__(self, parent, name)
        
        self.browseKey = None
        self.browseCurve = None
        
        self.radius  = 0.5
        self.pointWidth = 5
        
        self.showAxisScale = 1
        self.showXaxisTitle= 1
        self.showYLaxisTitle = 1
        self.showLegend = 1
##        self.showClusters = 0
        self.showFilledSymbols = 1
        
##        self.tooltipKind = 1
        
        self.markLines = []
        
        self.connect(self, SIGNAL("plotMouseMoved(const QMouseEvent &)"), self.onMouseMoved)
        self.connect(self, SIGNAL('plotMousePressed(const QMouseEvent&)'), self.onMousePressed)
        self.connect(self, SIGNAL('plotMouseReleased(const QMouseEvent&)'),self.onMouseReleased)        
        
    def activateBrowsing(self, activate):
        if activate:
            self.removeBrowsingCurve()
            self.markLines = []
            self.state = BROWSE_RECTANGLE
            self.browseCurve = SelectionCurve(self, pen = Qt.DashLine)
            self.browseKey = self.insertCurve(self.browseCurve)    
            
            self.__fixAxes()
        else:
            self.__backToZoom()
            
    def activateBrowsingCircle(self, activate):
        if activate:
            self.removeBrowsingCurve()
            self.markLines = []
            self.state = BROWSE_CIRCLE
            self.browseCurve = SelectionCurve(self, pen = Qt.DashLine)
            self.browseKey = self.insertCurve(self.browseCurve)      
     
            self.__fixAxes()
        else:
            self.__backToZoom()
    
    def __backToZoom(self):
        self.state = ZOOMING
        self.removeBrowsingCurve()
        for line in self.markLines:
            self.removeCurve(line)    
        self.markLines = []
        self.setAxisAutoScale(QwtPlot.xBottom)
        self.setAxisAutoScale(QwtPlot.xTop)
        self.setAxisAutoScale(QwtPlot.yLeft)
        self.setAxisAutoScale(QwtPlot.yRight)
        self.replot()        
        
    def __fixAxes(self):
        self.setAxisScale(QwtPlot.xBottom, self.axisScale(QwtPlot.xBottom).lBound(), self.axisScale(QwtPlot.xBottom).hBound())
        self.setAxisScale(QwtPlot.xTop, self.axisScale(QwtPlot.xTop).lBound(), self.axisScale(QwtPlot.xTop).hBound())
        self.setAxisScale(QwtPlot.yLeft, self.axisScale(QwtPlot.yLeft).lBound(), self.axisScale(QwtPlot.yLeft).hBound())
        self.setAxisScale(QwtPlot.yRight, self.axisScale(QwtPlot.yRight).lBound(), self.axisScale(QwtPlot.yRight).hBound())         
            
    def removeBrowsingCurve(self):
        if self.browseKey: self.removeCurve(self.browseKey)
        self.browseCurve = None
        self.browseKey = None
     
    def onMouseMoved(self, e):
        if self.state == BROWSE_RECTANGLE:
            xFloat = self.invTransform(QwtPlot.xBottom, e.x())
            yFloat = self.invTransform(QwtPlot.yLeft, e.y())  
            
            self.createCurve(xFloat, yFloat, 0)
            
            ##
            self.removeMarkers()
            for line in self.markLines:
                self.removeCurve(line)
            self.markLines = []
            cor = [(x, y, self.tips.texts[i]) for (i,(x,y)) in enumerate(self.tips.positions) if abs(xFloat - x)  <= self.radius and abs(yFloat - y) <= self.radius]
            self.addMarkers(cor, xFloat, yFloat, self.radius)
##            for x, y, text in cor:
##                self.addMarker(text, x, y)
            
            ##
            
            self.replot()

            self.event(e)            
        elif self.state == BROWSE_CIRCLE:
            xFloat = self.invTransform(QwtPlot.xBottom, e.x())
            yFloat = self.invTransform(QwtPlot.yLeft, e.y())     
            self.createCurve(xFloat, yFloat, 1)
                
            #
            self.removeMarkers()
            for line in self.markLines:
                self.removeCurve(line)
            self.markLines = []            
            cor = [(x, y, self.tips.texts[i]) for (i,(x,y)) in enumerate(self.tips.positions) if ((xFloat - x)*(xFloat - x) + (yFloat - y)*(yFloat - y) <= self.radius * self.radius)]
            self.addMarkers(cor, xFloat, yFloat, self.radius)
##            for x, y, text in cor:
##                self.addMarker(text, x, y)
    
            #
            self.replot()   
            
            self.event(e) 
        else:   
            OWGraph.onMouseMoved(self, e)
            
            
    def onMousePressed(self, e):        
        if self.state == BROWSE_RECTANGLE or self.state == BROWSE_CIRCLE:   
            self.event(e)            
        else:
            OWGraph.onMousePressed(self, e)
            
    def onMouseReleased(self, e):
        
        if self.state == BROWSE_RECTANGLE or self.state == BROWSE_CIRCLE:            
            self.event(e)            
        else:
            OWGraph.onMouseReleased(self, e)
            
    def createCurve(self, x, y, circle = 0):
        if not circle:
            self.browseCurve.setData([x - self.radius, x + self.radius, x + self.radius, x - self.radius, x - self.radius], [y - self.radius, y - self.radius, y + self.radius, y + self.radius, y - self.radius])
        else:
            xDataU = arange(x - self.radius + 0.0002, x + self.radius - 0.0002, 0.0001).tolist()
            xDataD = xDataU[:]
            xDataD.reverse()
            yDataU = [(y  + sqrt(self.radius*self.radius - (t - x)*(t - x))) for t in xDataU]
            yDataD = [(y  - sqrt(self.radius*self.radius - (t - x)*(t - x))) for t in xDataD]
            xDataU.extend(xDataD)
            yDataU.extend(yDataD)
            self.browseCurve.setData(xDataU, yDataU)
            
    def activateZooming(self):
##        self.browseButton.setOn(0)
##        self.browseButtonCircle.setOn(0)
        self.__backToZoom()
        self.state = ZOOMING
        if self.tempSelectionCurve: self.removeLastSelection()

    def activateRectangleSelection(self):
##        self.browseButton.setOn(0)
##        self.browseButtonCircle.setOn(0)
        self.__backToZoom()
        self.state = SELECT_RECTANGLE
        if self.tempSelectionCurve: self.removeLastSelection()

    def activatePolygonSelection(self):
##        self.browseButton.setOn(0)
##        self.browseButtonCircle.setOn(0)
        self.__backToZoom()
        self.state = SELECT_POLYGON
        if self.tempSelectionCurve: self.removeLastSelection()       

    def _sort(self, x, y):
        return int(sign((y[1] - x[1] and [y[1] - x[1]] or [x[0] - y[0]])[0]))
    def addMarkers(self, cor, x, y, r, bold = 0):
        if not len(cor):
            return

        cor.sort(self._sort)
        
        top = y + r
        top = self.transform(QwtPlot.yLeft, top)
        left = x - r
##        left = self.transform(QwtPlot.xBottom, left) - 35
##        left = self.invTransform(QwtPlot.xBottom, left) 
        right = x + r
##        right = self.transform(QwtPlot.xBottom, right) + 20
##        right = self.invTransform(QwtPlot.xBottom, right) 
        
        newMark = []
        for i, (x, y, text) in zip(range(len(cor)), cor):
            side = left
            if not (i & 1):
                if self.checkPerc(left) > 0:
                    newMark.append((left, self.invTransform(QwtPlot.yLeft, top), text, Qt.AlignLeft, x, y))
                else:
                    newMark.append((right, self.invTransform(QwtPlot.yLeft, top), text, Qt.AlignRight, x, y))
                    top = top + 10                    
            else:
                if self.checkPerc(right) < 70:
                    newMark.append((right, self.invTransform(QwtPlot.yLeft, top), text, Qt.AlignRight, x, y))
                else:
                    top = top + 10
                    newMark.append((left, self.invTransform(QwtPlot.yLeft, top), text, Qt.AlignLeft, x, y))
                top = top + 10
            
        for x, y, text, al, x1, y1 in newMark:
            self.addMarker(text, x, y, alignment = al, color = QColor())
            self.markLines.append(self.addCurve("", QColor("black"), QColor("black"), 1, QwtCurve.Lines, xData = [x, x1], yData = [y, y1] ))

            
    def checkPerc(self, x):
        div = self.axisScale(QwtPlot.xBottom)
        if x < div.lBound():
            return -1
        elif x > div.hBound():
            return 101
        else:
            return (x - div.lBound()) / (div.hBound() - div.lBound())
