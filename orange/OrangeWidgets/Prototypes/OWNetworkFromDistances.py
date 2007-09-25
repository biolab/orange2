"""
<name>Network from Distances</name>
<description>Costructs Graph object by connecting nodes from ExampleTable where distance between them is between given threshold.</description>
<icon>icons/Outlier.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>2030</priority>
"""

#
# OWNetworkFromDistances.py
#

import OWGUI
from OWWidget import *
from orange import Graph
from OWGraph import *
from OWNetworkCanvas import *

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
        
        self.setCurveData(self.lowerBoundaryKey, [self.lowerBoundary, self.lowerBoundary], [0, maxy])
        self.setCurveData(self.upperBoundaryKey, [self.upperBoundary, self.upperBoundary], [0, maxy])
        self.replot()
            
    def updateData(self):
        self.removeDrawingCurves(removeLegendItems = 0)
                    
        key = self.addCurve("histogramCurve", Qt.blue, Qt.blue, 6, style = QwtCurve.Steps, symbol = QwtSymbol.None, xData = self.xData, yData = self.yData, showFilledSymbols = False)
        
        maxy = max(self.yData)
        self.lowerBoundaryKey = self.addCurve("lowerBoundaryCurve", Qt.red, Qt.red, 6, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [self.lowerBoundary, self.lowerBoundary], yData = [0, maxy], showFilledSymbols = False)
        self.upperBoundaryKey = self.addCurve("upperBoundaryCurve", Qt.red, Qt.red, 6, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [self.upperBoundary, self.upperBoundary], yData = [0, maxy], showFilledSymbols = False)

#    def setAxisAutoScaled(self):
#        self.setAxisAutoScale(self.xBottom)
#        self.setAxisAutoScale(self.yLeft)
            
class OWNetworkFromDistances(OWWidget):
    settingsList=["threshold"]
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Network from Distances")
        
        self.inputs = [("Distance Matrix", orange.SymMatrix, self.cdata, Default)]
        self.outputs = [("Graph with ExampleTable", Graph), ("Examples", ExampleTable)]

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
#        boxHistogram = QVGroupBox("Distance histogram", self.mainArea)
#        self.histogram = Hist(self, boxHistogram)      
#        self.box = QVBoxLayout(boxHistogram)
#        self.box.addWidget(self.histogram)

        boxHistogram = QVGroupBox("Distance histogram", self.mainArea)
        self.histogram = Hist(self, boxHistogram)      
        self.box = QVBoxLayout(boxHistogram)
        self.box.addWidget(self.histogram)
        boxHistogram.setMinimumWidth(500)
        boxHistogram.setMinimumHeight(300)
        
        boxGeneral = QVGroupBox("Distance boundaries", self.controlArea)
        OWGUI.separator(self.controlArea)
        #cb, self.spinLower = OWGUI.checkWithSpin(boxGeneral, self, "Lower:", 0, 100000, "spinLowerChecked", "spinLowerThreshold", step=0.01, spinCallback=self.changeSpin)
        #cb, self.spinUpper = OWGUI.checkWithSpin(boxGeneral, self, "Upper:", 0, 100000, "spinUpperChecked", "spinUpperThreshold", step=0.01, spinCallback=self.changeSpin)
        
        OWGUI.lineEdit(boxGeneral, self, "spinLowerThreshold", "Lower:", callback=self.changeLowerSpin, valueType=float)
        OWGUI.lineEdit(boxGeneral, self, "spinUpperThreshold", "Upper:", callback=self.changeUpperSpin, valueType=float)
        
        # options
        boxOptions = QVGroupBox("Options", self.controlArea)
        self.includeNotConnected = 1
        self.attrColor = ""
        #box = OWGUI.widgetBox(self.GeneralTab, " Color Attribute")
        OWGUI.checkBox(boxOptions, self, 'includeNotConnected', 'Include not connected nodes', disabled = 1)#, callback = self.updateGraph)
        # info
        boxInfo = QVGroupBox("Network info", self.controlArea)
        self.infoa = QLabel("No data loaded.", boxInfo)
        self.infob = QLabel('', boxInfo)
        self.infoc = QLabel('', boxInfo)
        
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

        graph = orange.GraphAsList(self.data.dim, 0)
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
        self.send("Graph with ExampleTable", graph)
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