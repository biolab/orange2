"""
THIS WIDGET IS OBSOLETE; USE OWNxFromDistances.py
"""

#
# OWNetworkFromDistances.py
#

import OWGUI
import orange
import orngNetwork
import copy, random

from OWNetworkHist import *
from OWWidget import *
from OWGraph import *
from OWHist import *

class OWNetworkFromDistances(OWWidget, OWNetworkHist):
    settingsList=["spinLowerThreshold", "spinUpperThreshold", "netOption", "dstWeight", "kNN", "andor", "excludeLimit"]
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Network from Distances")
        OWNetworkHist.__init__(self)
        
        self.inputs = [("Distance Matrix", orange.SymMatrix, self.setMatrix)]
        self.outputs = [("Network", orngNetwork.Network), ("Examples", ExampleTable), ("Distance Matrix", orange.SymMatrix)]

        self.addHistogramControls()
        
        # get settings from the ini file, if they exist
        self.loadSettings()
        
        # GUI
        # general settings
        boxHistogram = OWGUI.widgetBox(self.mainArea, box = "Distance histogram")
        self.histogram = OWHist(self, boxHistogram)
        boxHistogram.layout().addWidget(self.histogram)

        boxHistogram.setMinimumWidth(500)
        boxHistogram.setMinimumHeight(300)
        
        # info
        boxInfo = OWGUI.widgetBox(self.controlArea, box = "Network info")
        self.infoa = OWGUI.widgetLabel(boxInfo, "No data loaded.")
        self.infob = OWGUI.widgetLabel(boxInfo, '')
        self.infoc = OWGUI.widgetLabel(boxInfo, '')
        
        OWGUI.rubber(self.controlArea)
        
        self.resize(700, 100)

    def sendReport(self):
        self.reportSettings("Settings",
                            [("Edge thresholds", "%.5f - %.5f" % (self.spinLowerThreshold, self.spinUpperThreshold)),
                             ("Selected vertices", ["All", "Without isolated vertices", "Largest component", "Connected with vertex"][self.netOption]),
                             ("Weight", ["Distance", "1 - Distance"][self.dstWeight])])
        self.reportSection("Histogram")
        self.reportImage(self.histogram.saveToFileDirect, QSize(400,300))
        self.reportSettings("Output graph",
                            [("Vertices", self.matrix.dim),
                             ("Edges", self.nedges),
                             ("Connected vertices", "%i (%.1f%%)" % (self.pconnected, self.pconnected / max(1, float(self.matrix.dim))*100)),
                             ])
        
    def sendSignals(self):
        if self.graph != None:
            #setattr(matrix, "items", self.graph.items)
            self.matrix.items = self.graph.items
        
        self.send("Network", self.graph)
        
        if self.matrix:
            self.send("Distance Matrix", self.matrix)
            
        if self.graph == None:
            self.send("Examples", None)
        else:
            self.send("Examples", self.graph.items)
                                                                     
if __name__ == "__main__":    
    appl = QApplication(sys.argv)
    ow = OWNetworkFromDistances()
    ow.show()
    appl.exec_()
