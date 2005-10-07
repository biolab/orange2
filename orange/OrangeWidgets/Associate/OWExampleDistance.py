"""
<name>Example Distance</name>
<description>Computes a distance matrix from a set of data examples.</description>
<icon>icons/ExampleDistance.png</icon>
<priority>1050</priority>
"""

import orange, math
import OWGUI
from qt import *
from qtcanvas import *
from OWWidget import *
import random

##############################################################################
# main class

class OWExampleDistance(OWWidget):	
    settingsList = ["Metrics"]

    def __init__(self, parent=None, signalManager = None):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, 'ExampleDistance') 

        self.inputs = [("Examples", ExampleTable, self.dataset)]
        self.outputs = [("Distance Matrix", orange.SymMatrix)]

        self.Metrics = 0
        self.loadSettings()
        self.data = None

        self.metrics = [("Euclidean", orange.ExamplesDistanceConstructor_Euclidean),
                   ("Manhattan", orange.ExamplesDistanceConstructor_Manhattan),
                   ("Hamiltonian", orange.ExamplesDistanceConstructor_Hamiltonian),
                   ("Relief", orange.ExamplesDistanceConstructor_Relief)]

        #set default settings
        items = [x[0] for x in self.metrics]
        OWGUI.comboBox(self.controlArea, self, "Metrics", box="Distance Metrics", items=items,
            tooltip="Choose metrics to measure pairwise distance between examples.",
            callback=self.computeMatrix)
        self.labelCombo = OWGUI.comboBox(self.controlArea, self, "Label", box="Example Label",
            items=[],
            tooltip="Choose attribute which will be used as a label of the example.",
            callback=self.setLabel, sendSelectedValue = 1)
        self.labelCombo.setDisabled(1)
        self.resize(100,100)

    ##############################################################################
    # callback functions

    def computeMatrix(self):
        if not self.data:
            return
        data = self.data
        dist = self.metrics[self.Metrics][1](data)
        matrix = orange.SymMatrix(len(data))
        matrix.setattr('items', data)
        for i in range(len(data)):
            for j in range(i+1):
                matrix[i, j] = dist(data[i], data[j])
        self.send("Distance Matrix", matrix)

    def setLabel(self):
        for e in self.data:
            e.name = str(e[self.Label])

    ##############################################################################
    # input signal management

    def setLabelComboItems(self):
        d = self.data
        self.labelCombo.clear()
        self.labelCombo.setDisabled(0)
        labels = [m.name for m in d.domain.getmetas().values()] + [a.name for a in d.domain.variables]
        for l in labels:
            self.labelCombo.insertItem(l)
        # here we would need to use the domain dependent setting of the label id
        self.labelCombo.setCurrentItem(0); self.Label = labels[0]
        self.setLabel()

    def dataset(self, data):
        if data and len(data.domain.attributes):
            self.data = data
            self.setLabelComboItems()
            self.computeMatrix()
        else:
            self.send("Distance Matrix", None)

##################################################################################################
# test script

if __name__=="__main__":
    import os
    if os.path.isfile(r'../../doc/datasets/glass'):
        data = orange.ExampleTable(r'../../doc/datasets/glass')
    else:
        data = orange.ExampleTable('glass')
    a = QApplication(sys.argv)
    ow = OWExampleDistance()
    a.setMainWidget(ow)
    ow.show()
    ow.dataset(data)
    a.exec_loop()
    ow.saveSettings()
