"""
<name>Network from Distances</name>
<description>Costructs Graph object by connecting nodes from ExampleTable where distance between them is between given threshold.</description>
<icon>icons/Outlier.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>3013</priority>
"""

#
# OWNetworkFromDistances.py
#

import OWGUI
from OWWidget import *
from OWGraph import *
from orngNetwork import * 
from orangeom import Network

class Hist(OWGraph):
    def __init__(self, master, parent = None):
        OWGraph.__init__(self, parent, "Histogram")
        self.master = master
        self.parent = parent

        self.enableXaxis(1)
        self.enableYLaxis(1)
        self.state = NOTHING  #default je rocno premikanje
        
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
        self.minValue = min(values)
        self.maxValue = max(values)
        
        boxes = 100
        box_size = (self.maxValue - self.minValue) / boxes
        
        if box_size > 0:
            self.xData = []
            self.yData = [0] * boxes
            for i in range(boxes):
                self.xData.append(self.minValue + i * box_size + box_size / 2)
                 
            for value in values:
                box = int((value - self.minValue) / box_size)
                if box >= len(self.yData):
                    box = boxes - 1
                n = self.yData[box]
                self.yData[box] = n + 1
                
            #print values
            #print self.xData
            #print self.yData 
            
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
        
        maxy = max(self.yData)
        self.lowerBoundaryKey = self.addCurve("lowerBoundaryCurve", Qt.red, Qt.red, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Lines, xData = [self.lowerBoundary, self.lowerBoundary], yData = [0, maxy])
        self.upperBoundaryKey = self.addCurve("upperBoundaryCurve", Qt.red, Qt.red, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Lines, xData = [self.upperBoundary, self.upperBoundary], yData = [0, maxy])

        print self.lowerBoundary
        print self.upperBoundary
        self.setAxisScale(QwtPlot.xBottom, min(self.xData), max(self.xData))
        self.setAxisScale(QwtPlot.yLeft, min(self.yData), maxy)
#    def setAxisAutoScaled(self):
#        self.setAxisAutoScale(self.xBottom)
#        self.setAxisAutoScale(self.yLeft)
            
class OWNetworkFromDistances(OWWidget):
    settingsList=["threshold", "spinLowerThreshold", "spinUpperThreshold"]
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Network from Distances")
        
        self.inputs = [("Distance Matrix", orange.SymMatrix, self.cdata, Default)]
        self.outputs = [("Network", Network), ("Examples", ExampleTable)]

        self.spinLowerThreshold = 0
        self.spinLowerChecked = False
        self.spinUpperThreshold = 0
        self.spinUpperChecked = False
    
        # set default settings
        self.data = None
        self.threshold = 0.2
        # get settings from the ini file, if they exist
        self.loadSettings()
        
        # GUI
        # general settings
        boxHistogram = OWGUI.widgetBox(self.mainArea, box = "Distance histogram")
        self.histogram = Hist(self, boxHistogram)
        boxHistogram.layout().addWidget(self.histogram)

        boxHistogram.setMinimumWidth(500)
        boxHistogram.setMinimumHeight(300)
        
        boxGeneral = OWGUI.widgetBox(self.controlArea, box = "Distance boundaries")
        OWGUI.separator(self.controlArea)
        #cb, self.spinLower = OWGUI.checkWithSpin(boxGeneral, self, "Lower:", 0, 100000, "spinLowerChecked", "spinLowerThreshold", step=0.01, spinCallback=self.changeSpin)
        #cb, self.spinUpper = OWGUI.checkWithSpin(boxGeneral, self, "Upper:", 0, 100000, "spinUpperChecked", "spinUpperThreshold", step=0.01, spinCallback=self.changeSpin)
        
        OWGUI.lineEdit(boxGeneral, self, "spinLowerThreshold", "Lower:", callback=self.changeLowerSpin, valueType=float)
        OWGUI.lineEdit(boxGeneral, self, "spinUpperThreshold", "Upper:", callback=self.changeUpperSpin, valueType=float)
        
        # options
        boxOptions = OWGUI.widgetBox(self.controlArea, box = "Options")
        self.excludeUnconnected = 0
        self.attrColor = ""
        #box = OWGUI.widgetBox(self.GeneralTab, " Color Attribute")
        OWGUI.checkBox(boxOptions, self, 'excludeUnconnected', 'Exclude unconnected nodes', disabled = 1)#, callback = self.updateGraph)
        # info
        boxInfo = OWGUI.widgetBox(self.controlArea, box = "Network info")
        self.infoa = OWGUI.widgetLabel(boxInfo, "No data loaded.")
        self.infob = OWGUI.widgetLabel(boxInfo, '')
        self.infoc = OWGUI.widgetLabel(boxInfo, '')
        
        self.resize(700, 322)

    def cdata(self, data):
        if data == None:
            return
        
        self.data = data
        
        # draw histogram
        values = []
        for i in range(data.dim):
            for j in range(i):
                values.append(data[i][j])
        
        self.histogram.setValues(values)
        #print maxValue
        #self.spinLower.setMaxValue(maxValue)
        #self.spinUpper.setMaxValue(maxValue)
        
        self.generateGraph()
        
    def changeLowerSpin(self):
        if self.spinLowerThreshold >= self.spinUpperThreshold:
            self.spinLowerThreshold = self.spinUpperThreshold
        elif self.spinLowerThreshold < self.histogram.minValue:
            self.spinLowerThreshold = self.histogram.minValue
        elif self.spinLowerThreshold > self.histogram.maxValue:
            self.spinLowerThreshold = self.histogram.maxValue
            
        self.generateGraph()
        
    def changeUpperSpin(self):
        if self.spinUpperThreshold <= self.spinLowerThreshold:
            self.spinUpperThreshold = self.spinLowerThreshold
        elif self.spinUpperThreshold < self.histogram.minValue:
            self.spinUpperThreshold = self.histogram.minValue
        elif self.spinUpperThreshold > self.histogram.maxValue:
            self.spinUpperThreshold = self.histogram.maxValue
        
        self.generateGraph()
        
    def generateGraph(self):
        if self.data == None:
            self.infoa.setText("No data loaded.")
            self.infob.setText("")
            return

        graph = Network(self.data.dim, 0)
        graph.setattr("items", self.data.items)
            
        # set the threshold
        # set edges where distance is lower than threshold
        n = 0
        nedges = 0
        #print self.spinLowerThreshold
        #print self.spinUpperThreshold
        for i in range(self.data.dim):
            oldn = n
            for j in range(i):
                if self.spinLowerThreshold < self.data[i][j] and self.data[i][j] < self.spinUpperThreshold:
                    n += 1
                    graph[i,j] = 1
            if n > oldn:
                nedges += 1
          
        self.graph = graph
        self.infoa.setText("%d vertices" % self.data.dim)
        self.infob.setText("%d connected (%3.1f%%)" % (nedges, nedges / float(self.data.dim) * 100))
        self.infoc.setText("%d edges (%d average)" % (n, n / float(self.data.dim)))
        self.send("Network", graph)
        self.send("Examples", graph.items)
        self.histogram.setBoundary(self.spinLowerThreshold, self.spinUpperThreshold)
    
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWNetworkFromDistances()
    owf.activateLoadedSettings()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()