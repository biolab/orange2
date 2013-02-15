"""
<name>Example Distance</name>
<description>Computes a distance matrix from a set of data examples.</description>
<icon>icons/Distance.svg</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>1300</priority>
"""

import OWGUI
from OWWidget import *

import Orange

from Orange import distance
from Orange.utils import progress_bar_milestones


class OWExampleDistance(OWWidget):
    settingsList = ["Metrics", "Normalize"]
    contextHandlers = {"": DomainContextHandler("", ["Label"])}

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'ExampleDistance',
                          wantMainArea=False, resizingEnabled=False)

        self.inputs = [("Data", Orange.data.Table, self.dataset)]
        self.outputs = [("Distances", Orange.misc.SymMatrix)]

        self.Metrics = 0
        self.Normalize = True
        self.Absolute = False
        self.Label = ""
        self.loadSettings()
        self.data = None
        self.matrix = None

        self.metrics = [
            ("Euclidean", distance.Euclidean),
            ("Pearson Correlation", distance.PearsonR),
            ("Spearman Rank Correlation", distance.SpearmanR),
            ("Manhattan", distance.Manhattan),
            ("Hamming", distance.Hamming),
            ("Relief", distance.Relief),
            ]

        cb = OWGUI.comboBox(
            self.controlArea, self, "Metrics", box="Distance Metrics",
            items=[x[0] for x in self.metrics],
            tooltip=("Choose metrics to measure pairwise distance between "
                     "examples."),
            callback=self.distMetricChanged,
            valueType=str
        )

        cb.setMinimumWidth(170)

        OWGUI.separator(self.controlArea)

        box = OWGUI.widgetBox(self.controlArea, "Settings",
                              addSpace=True)

        self.normalizeCB = OWGUI.checkBox(box, self, "Normalize",
                                          "Normalize data",
                                          callback=self.computeMatrix)

        self.normalizeCB.setEnabled(self.Metrics in [0, 3])

        self.absoluteCB = OWGUI.checkBox(
            box, self, "Absolute",
            "Absolute correlations",
            tooltip=("Use absolute correlations "
                     "for distances."),
            callback=self.computeMatrix
        )

        self.absoluteCB.setEnabled(self.Metrics in [1, 2])

        self.labelCombo = OWGUI.comboBox(
            self.controlArea, self, "Label",
            box="Example Label",
            items=[],
            tooltip="Attribute used for example labels",
            callback=self.setLabel,
            sendSelectedValue=True
        )

        self.labelCombo.setDisabled(True)

        OWGUI.rubber(self.controlArea)

    def sendReport(self):
        metric = self.metrics[self.Metrics][0]
        if self.Metrics in [0, 3] and self.Normalize:
            metric = "Normalized " + metric
        elif self.Metrics in [1, 2] and self.Absolute:
            metric = "Absolute " + metric

        self.reportSettings("Settings",
                            [("Metrics", metric),
                             ("Label", self.Label)])
        self.reportData(self.data)

    def distMetricChanged(self):
        self.normalizeCB.setEnabled(self.Metrics in [0, 3])
        self.absoluteCB.setEnabled(self.Metrics in [1, 2])
        self.computeMatrix()

    def computeMatrix(self):
        if not self.data:
            return

        data = self.data
        if self.Metrics in [1, 2] and self.Absolute:
            if self.Metrics == 1:
                constructor = distance.PearsonRAbsolute()
            else:
                constructor = distance.SpearmanRAbsolute()
        else:
            constructor = self.metrics[self.Metrics][1]()
            constructor.normalize = self.Normalize

        self.error(0)
        self.progressBarInit()
        try:
            matrix = distance.distance_matrix(data, constructor,
                                              self.progressBarSet)
        except Orange.core.KernelException, ex:
            self.error(0, "Could not create distance matrix! %s" % str(ex))
            matrix = None

        self.progressBarFinished()

        if matrix:
            matrix.setattr('items', data)

        self.matrix = matrix
        self.send("Distances", self.matrix)

    def setLabel(self):
        for d in self.data:
            d.name = str(d[str(self.Label)])
        self.send("Distances", self.matrix)

    def setLabelComboItems(self):
        d = self.data
        self.labelCombo.clear()
        self.labelCombo.setDisabled(0)
        labels = [m.name for m in d.domain.getmetas().values()] + \
                 [a.name for a in d.domain.variables]
        self.labelCombo.addItems(labels)

        # here we would need to use the domain dependent setting of the
        # label id
        self.labelCombo.setCurrentIndex(0)
        self.Label = labels[0]
        self.setLabel()

    def dataset(self, data):
        if data and len(data.domain.attributes):
            self.data = data
            self.setLabelComboItems()
            self.computeMatrix()
        else:
            self.data = None
            self.matrix = None
            self.labelCombo.clear()
            self.send("Distances", None)


if __name__ == "__main__":
    data = Orange.data.Table('glass')
    a = QApplication(sys.argv)
    ow = OWExampleDistance()
    ow.show()
    ow.dataset(data)
    a.exec_()
    ow.saveSettings()
