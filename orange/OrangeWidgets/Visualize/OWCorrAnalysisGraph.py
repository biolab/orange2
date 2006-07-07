"""
graph for correspondence analysis
"""

BROWSE_RECTANGLE = 4
BROWSE_CIRCLE= 5


from OWGraph import *

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
        
        
        
        self.connect(self, SIGNAL("plotMouseMoved(const QMouseEvent &)"), self.onMouseMoved)
        self.connect(self, SIGNAL('plotMousePressed(const QMouseEvent&)'), self.onMousePressed)
        self.connect(self, SIGNAL('plotMouseReleased(const QMouseEvent&)'),self.onMouseReleased)        
        
    def activateBrowsing(self, activate):
        if activate:
            self.removeBrowsingCurve()
            self.state = BROWSE_RECTANGLE
            self.browseCurve = SelectionCurve(self, pen = Qt.DashLine)
            self.browseKey = self.insertCurve(self.browseCurve)    
            
            self.__fixAxes()
        else:
            self.__backToZoom()
            
    def activateBrowsingCircle(self, activate):
        if activate:
            self.removeBrowsingCurve()
            self.state = BROWSE_CIRCLE
            self.browseCurve = SelectionCurve(self, pen = Qt.DashLine)
            self.browseKey = self.insertCurve(self.browseCurve)      
     
            self.__fixAxes()
        else:
            self.__backToZoom()
    
    def __backToZoom(self):
        self.state = ZOOMING
        self.removeBrowsingCurve()
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
            
            self.createCurve(xFloat, yFloat)
            self.replot()

            self.event(e)            
        elif self.state == BROWSE_CIRCLE:
            xFloat = self.invTransform(QwtPlot.xBottom, e.x())
            yFloat = self.invTransform(QwtPlot.yLeft, e.y())     
            self.createCurve(xFloat, yFloat)
            self.replot()   
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
        if circle:
            self.browseCurve.setData([x - self.radius, x + self.radius, x + self.radius, x - self.radius, x - self.radius], [y - self.radius, y - self.radius, y + self.radius, y + self.radius, y - self.radius])
        else:
            self.browseCurve.setData([x - self.radius, x + self.radius, x + self.radius, x - self.radius, x - self.radius], [y - self.radius, y - self.radius, y + self.radius, y + self.radius, y - self.radius]) ## TODO: change to circle
            
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
        
        
