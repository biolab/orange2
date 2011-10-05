"""
<name>Example Distance</name>
<description>Computes a distance matrix from a set of data examples.</description>
<icon>icons/ExampleDistance.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>1300</priority>
"""
import orange, math
import OWGUI
from OWWidget import *
import random
import orngClustering
import orngMisc

##############################################################################
# main class

class OWExampleDistance(OWWidget):
    settingsList = ["Metrics", "Normalize"]
    contextHandlers = {"": DomainContextHandler("", ["Label"])}

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, 'ExampleDistance', wantMainArea = 0, resizingEnabled = 0)

        self.inputs = [("Examples", ExampleTable, self.dataset)]
        self.outputs = [("Distance Matrix", orange.SymMatrix)]

        self.Metrics = 0
        self.Normalize = True
        self.Label = ""
        self.loadSettings()
        self.data = None
        self.matrix = None

        self.metrics = [
            ("Euclidean", orange.ExamplesDistanceConstructor_Euclidean),
            ("Pearson Correlation", orngClustering.ExamplesDistanceConstructor_PearsonR),
            ("Spearman Rank Correlation", orngClustering.ExamplesDistanceConstructor_SpearmanR),
            ("Manhattan", orange.ExamplesDistanceConstructor_Manhattan),
            ("Hamming", orange.ExamplesDistanceConstructor_Hamming),
            ("Relief", orange.ExamplesDistanceConstructor_Relief),
            ]

        cb = OWGUI.comboBox(self.controlArea, self, "Metrics", box="Distance Metrics",
            items=[x[0] for x in self.metrics],
            tooltip="Choose metrics to measure pairwise distance between examples.",
            callback=self.distMetricChanged, valueType=str)
        cb.setMinimumWidth(170)
        
        OWGUI.separator(self.controlArea)
        
        box = OWGUI.widgetBox(self.controlArea, "Normalization", 
                              addSpace=True)
        self.normalizeCB = OWGUI.checkBox(box, self, "Normalize", "Normalize data", 
                                          callback=self.computeMatrix)
        
        self.normalizeCB.setEnabled(self.Metrics in [0, 3])
        
        self.labelCombo = OWGUI.comboBox(self.controlArea, self, "Label", box="Example Label",
            items=[],
            tooltip="Attribute used for example labels",
            callback=self.setLabel, sendSelectedValue = 1)

        self.labelCombo.setDisabled(1)
        
        OWGUI.rubber(self.controlArea)

    def sendReport(self):
        self.reportSettings("Settings",
                            [("Metrics", self.metrics[self.Metrics][0]),
                             ("Label", self.Label)])
        self.reportData(self.data)

    def distMetricChanged(self):
        self.normalizeCB.setEnabled(self.Metrics in [0, 3])
        self.computeMatrix()

    def computeMatrix(self):
        if not self.data:
            return
        data = self.data
        constructor = self.metrics[self.Metrics][1]()
        constructor.normalize = self.Normalize
        dist = constructor(data)
        self.error(0)
        try:
            self.matrix = orange.SymMatrix(len(data))
        except orange.KernelException, ex:
            self.error(0, "Could not create distance matrix! %s" % str(ex))
            self.matrix = None
            self.send("Distance Matrix", None)
            return
        self.matrix.setattr('items', data)
        pb = OWGUI.ProgressBar(self, 100)
        milestones  = orngMisc.progressBarMilestones(len(data)*(len(data)-1)/2, 100)
        count = 0
        for i in range(len(data)):
            for j in range(i+1):
                self.matrix[i, j] = dist(data[i], data[j])
                if count in milestones:
                    pb.advance()
                count += 1
        pb.finish()
        self.send("Distance Matrix", self.matrix)

    def setLabel(self):
        for d in self.data:
            d.name = str(d[str(self.Label)])
        self.send("Distance Matrix", self.matrix)

    def setLabelComboItems(self):
        d = self.data
        self.labelCombo.clear()
        self.labelCombo.setDisabled(0)
        labels = [m.name for m in d.domain.getmetas().values()] + \
                 [a.name for a in d.domain.variables]
        self.labelCombo.addItems(labels)
        # here we would need to use the domain dependent setting of the label id
        self.labelCombo.setCurrentIndex(0); self.Label = labels[0]
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
    data = orange.ExampleTable(r'../../doc/datasets/glass')
    data = orange.ExampleTable('glass')
    a = QApplication(sys.argv)
    ow = OWExampleDistance()
    ow.show()
    ow.dataset(data)
    a.exec_()
    ow.saveSettings()
