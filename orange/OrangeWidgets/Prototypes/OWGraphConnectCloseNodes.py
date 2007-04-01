"""
<name>Connect Close Nodes</name>
<description>Costructs Graph object by connecting nodes from ExampleTable where distance between them is less than given threshold.</description>
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

class OWGraphConnectByEuclid(OWWidget):
    settingsList=["threshold"]
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Connect Nodes by Euclid")
        
        self.inputs = [("Distance Matrix", orange.SymMatrix, self.cdata, Default)]
        self.outputs = [("Graph with ExampleTable", Graph), ("Examples", ExampleTable)]

        self.threshold = 0
    
        # GUI
        # general settings
        boxGeneral = QVGroupBox("General Settings", self.controlArea)
        self.spin = OWGUI.doubleSpin(boxGeneral, self, "threshold", 0, 100000, step=0.01, label="Threshold:", callback=self.changeSpin)
        self.spin.setMinimumWidth(250)
        
        # info
        boxInfo = QVGroupBox("Info", self.controlArea)
        self.infoa = QLabel("No data loaded.", boxInfo)
        self.infob = QLabel('', boxInfo)
        
        self.resize(150,80)

        # set default settings
        self.data = None
        self.threshold = 0.2
        # get settings from the ini file, if they exist
        self.loadSettings()
        
    def cdata(self, data):
        self.data = data
        self.generateGraph()
        
    def changeSpin(self):
        self.generateGraph()
        
    def generateGraph(self):
        if self.data == None:
            self.infoa.setText("No data loaded.")
            self.infob.setText("")
            return
        
        # construct the function to measure the distances
        dist = data
        
        nedges = [] 
        for i in range(len(self.data)):
           n = 0
           for j in range(len(self.data)):
              if i == j: continue
              if dist(self.data[i], self.data[j]) < self.threshold:
                 n += 1
           nedges.append(n)
        n = 0; idid = []
        for i in range(len(self.data)):
           idid.append(n)
           if nedges[i]:
              n += 1

        self.infoa.setText("%d vertices, " % len(self.data) + "%d (%3.2f) not connected" % (nedges.count(0), nedges.count(0)/float(len(self.data))))

        graph = orange.GraphAsList(len(self.data), 0)
        graph.setattr("items", self.data)
        
        # set the threshold
        # set edges where distance is lower than threshold
        n = 0
        for i in range(len(self.data)):
           for j in range(i+1):
              if i == j: 
                  continue
              if dist(self.data[i], self.data[j]) < self.threshold:
                  n += 1
                  graph[i,j] = 1
          
        self.graph = graph
        self.infob.setText("%d edges (%d average)" % (n, n/float(len(self.data))))
        self.send("Graph with ExampleTable", graph)
        self.send("Examples", graph.items)
    
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWGraphConnectByEuclid()
    owf.activateLoadedSettings()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()