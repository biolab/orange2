#
# OWHist.py
#
# the base for all histograms

import OWGUI
from OWWidget import *
from OWGraph import *

class OWHist(OWGraph):
    def __init__(self, parent, type=0):
        OWGraph.__init__(self, None, "Histogram")
        self.parent = parent
        self.type = type
        
        self.enableXaxis(1)
        self.enableYLaxis(1)
        
        self.xData = []
        self.yData = []
        
        self.minValue = 0
        self.maxValue = 0
        self.lowerBoundary = 0
        self.upperBoundary = 0
        self.lowerBoundaryKey = None
        self.upperBoundaryKey = None
        
        self.enableGridXB(False)
        self.enableGridYL(False)

    def setValues(self, values):
        nBins = 100
        if len(values) < 100:
            nBins = len(values)

        (self.yData, self.xData) = numpy.histogram(values, bins=100)
        
        self.minx = min(self.xData)
        self.maxx = max(self.xData)
        self.miny = min(self.yData)
        self.maxy = max(self.yData)
        
        self.minValue = self.minx
        self.maxValue = self.maxx
        
        self.updateData()
        self.replot()
        
    def setBoundary(self, lower, upper):
        self.lowerBoundary = lower
        self.upperBoundary = upper
        maxy = max(self.yData)
        
        self.lowerBoundaryKey.setData([self.lowerBoundary, self.lowerBoundary], [0, maxy])
        self.upperBoundaryKey.setData([self.upperBoundary, self.upperBoundary], [0, maxy])
        self.replot()
            
    def updateData(self):
        self.removeDrawingCurves(removeLegendItems = 0)
                    
        self.key = self.addCurve("histogramCurve", Qt.blue, Qt.blue, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Steps, xData = self.xData, yData = self.yData)
        
        maxy = self.maxy
        self.lowerBoundaryKey = self.addCurve("lowerBoundaryCurve", Qt.red, Qt.red, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Lines, xData = [self.lowerBoundary, self.lowerBoundary], yData = [0, maxy])
        self.upperBoundaryKey = self.addCurve("upperBoundaryCurve", Qt.red, Qt.red, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Lines, xData = [self.upperBoundary, self.upperBoundary], yData = [0, maxy])

        minx = self.minx
        maxx = self.maxx
        miny = self.miny

        self.setAxisScale(QwtPlot.xBottom, minx - (0.05 * (maxx - minx)), maxx + (0.05 * (maxx - minx)))
        self.setAxisScale(QwtPlot.yLeft, miny - (0.05 * (maxy - miny)), maxy + (0.05 * (maxy - miny)))
