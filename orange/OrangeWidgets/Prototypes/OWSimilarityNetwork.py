"""
<name>Similarity Network</name>
<description>Costructs Graph object by connecting nodes from ExampleTable where distance between them is between given threshold.</description>
<icon>icons/Outlier.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>2030</priority>
"""

#
# OWGraphConnectByEuclid.py
#

import OWGUI
from OWWidget import *
from orange import Graph

class OWSimilarityNetwork(OWWidget):
    settingsList=["threshold"]
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Similarity Network")
        
        self.inputs = [("Distance Matrix", orange.SymMatrix, self.cdata, Default)]
        self.outputs = [("Graph with ExampleTable", Graph), ("Examples", ExampleTable)]

        self.spinLowerThreshold = 0
        self.spinLowerChecked = False
        self.spinUpperThreshold = 0
        self.spinUpperChecked = False
    
        # GUI
        # general settings
        boxGeneral = QVGroupBox("Thresholding", self.controlArea)
        #cb, self.spinLower = OWGUI.checkWithSpin(boxGeneral, self, "Lower:", 0, 100000, "spinLowerChecked", "spinLowerThreshold", step=0.01, spinCallback=self.changeSpin)
        #cb, self.spinUpper = OWGUI.checkWithSpin(boxGeneral, self, "Upper:", 0, 100000, "spinUpperChecked", "spinUpperThreshold", step=0.01, spinCallback=self.changeSpin)
        
        OWGUI.lineEdit(boxGeneral, self, "spinLowerThreshold", "Lower:", callback=self.changeSpin, valueType=float)
        OWGUI.lineEdit(boxGeneral, self, "spinUpperThreshold", "Upper:", callback=self.changeSpin, valueType=float)
        
        # info
        boxInfo = QVGroupBox("Info", self.controlArea)
        self.infoa = QLabel("No data loaded.", boxInfo)
        self.infob = QLabel('', boxInfo)
        
        self.resize(230,80)

        # set default settings
        self.data = None
        self.threshold = 0.2
        # get settings from the ini file, if they exist
        self.loadSettings()
        
    def cdata(self, data):
        self.data = data
        
        maxValue = max([max(r) for r in data])
        #print maxValue
        #self.spinLower.setMaxValue(maxValue)
        #self.spinUpper.setMaxValue(maxValue)
        
        self.generateGraph()
        
    def changeSpin(self):
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
        self.infoa.setText("%d vertices, " % self.data.dim + "%d (%3.2f) not connected" % (nedges, nedges/float(self.data.dim)))
        self.infob.setText("%d edges (%d average)" % (n, n/float(self.data.dim)))
        self.send("Graph with ExampleTable", graph)
        self.send("Examples", graph.items)
    
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWSimilarityNetwork()
    owf.activateLoadedSettings()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()