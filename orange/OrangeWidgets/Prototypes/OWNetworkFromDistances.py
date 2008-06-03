"""
<name>Network from Distances</name>
<description>Costructs Graph object by connecting nodes from ExampleTable where distance between them is between given threshold.</description>
<icon>icons/NetworkFromDistances.png</icon>
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
from OWHist import *
            
class OWNetworkFromDistances(OWWidget):
    settingsList=["threshold", "spinLowerThreshold", "spinUpperThreshold", "largestComponent", "excludeUnconnected"]
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Network from Distances")
        
        self.inputs = [("Distance Matrix", orange.SymMatrix, self.cdata, Default)]
        self.outputs = [("Network", Network), ("Examples", ExampleTable)]

        self.spinLowerThreshold = 0
        self.spinLowerChecked = False
        self.spinUpperThreshold = 0
        self.spinUpperChecked = False
        self.largestComponent = 0
        self.excludeUnconnected = 0
        
        # set default settings
        self.data = None
        self.threshold = 0.2
        # get settings from the ini file, if they exist
        self.loadSettings()
        
        # GUI
        # general settings
        boxHistogram = OWGUI.widgetBox(self.mainArea, box = "Distance histogram")
        self.histogram = OWHist(self, boxHistogram)
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
        
        self.attrColor = ""
        #box = OWGUI.widgetBox(self.GeneralTab, " Color Attribute")
        OWGUI.checkBox(boxOptions, self, 'excludeUnconnected', 'Exclude unconnected vertices', callback = self.generateGraph)
        OWGUI.checkBox(boxOptions, self, 'largestComponent', 'Largest connected component only', callback = self.generateGraph)
        
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
        t1 = time.time()
        data.matrixType = data.Lower
        values = data.getValues()
        t2 = time.time()
        self.histogram.setValues(values)
        t3 = time.time()
        #print maxValue
        low = min(values)
        upp = max(values)
        self.spinLowerThreshold = self.spinUpperThreshold = low - (0.03 * (upp - low))
        print self.spinLowerThreshold
        t4 = time.time()
        self.generateGraph()
        t5 = time.time()
        #print t1-t2,t2-t3,t3-t4,t4-t5
        
    def changeLowerSpin(self):
        if self.spinLowerThreshold < self.histogram.minValue:
            self.spinLowerThreshold = self.histogram.minValue
        elif self.spinLowerThreshold > self.histogram.maxValue:
            self.spinLowerThreshold = self.histogram.maxValue
            
        if self.spinLowerThreshold >= self.spinUpperThreshold:
            self.spinUpperThreshold = self.spinLowerThreshold
            
        self.generateGraph()
        
    def changeUpperSpin(self):
        if self.spinUpperThreshold < self.histogram.minValue:
            self.spinUpperThreshold = self.histogram.minValue
        elif self.spinUpperThreshold > self.histogram.maxValue:
            self.spinUpperThreshold = self.histogram.maxValue
            
        if self.spinUpperThreshold <= self.spinLowerThreshold:
            self.spinLowerThreshold = self.spinUpperThreshold
        
        self.generateGraph()
        
    def generateGraph(self):
        if self.data == None:
            self.infoa.setText("No data loaded.")
            self.infob.setText("")
            return

        graph = Network(self.data.dim, 0)
        
        if hasattr(self.data, "items"):
            graph.setattr("items", self.data.items)
            
        # set the threshold
        # set edges where distance is lower than threshold
        nedges = graph.fromSymMatrix(self.data, self.spinLowerThreshold, self.spinUpperThreshold)
        n = len(graph.getEdges())
        
        if self.largestComponent:
            components = graph.getConnectedComponents()[0]
            if len(components) > 1:
                self.graph = Network(graph.getSubGraph(components))
            else:
                self.graph = None
        elif self.excludeUnconnected:
            components = [x for x in graph.getConnectedComponents() if len(x) > 1]
            
            if len(components) > 1:
                include = reduce(lambda x,y: x+y, components)
                
                if len(include) > 1:
                    self.graph = Network(graph.getSubGraph(include))
                else:
                    self.graph = None
            else:
                self.graph = None
        else:
            self.graph = graph
    
        self.infoa.setText("%d vertices" % self.data.dim)
        self.infob.setText("%d connected (%3.1f%%)" % (nedges, nedges / float(self.data.dim) * 100))
        self.infoc.setText("%d edges (%d average)" % (n, n / float(self.data.dim)))
        
        self.send("Network", self.graph)
        if self.graph == None:
             self.send("Examples", None)
        else:
            self.send("Examples", self.graph.items)
        
        self.histogram.setBoundary(self.spinLowerThreshold, self.spinUpperThreshold)
    
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWNetworkFromDistances()
    owf.activateLoadedSettings()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()