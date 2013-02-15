"""
<name>SOM</name>
<description>Self organizing maps learner.</description>
<icon>icons/SOM.svg</icon>
<contact>Ales Erjavec (ales.erjevec(@at@)fri.uni.lj.si)</contact>
<priority>5010</priority>
"""

import orange
import orngSOM
from OWWidget import *
import OWGUI


class OWSOM(OWWidget):
    settingsList = ["xdim", "ydim", "neighborhood", "topology", "alphaType",
                    "iterations1", "iterations2", "radius1", "radius2",
                    "alpha1", "alpha2", "initialization", "eps"]

    def __init__(self, parent=None, signalManager=None, name="SOM"):
        OWWidget.__init__(self, parent, signalManager, name,
                          wantMainArea=False)

        self.inputs = [("Data", ExampleTable, self.setData)]
        self.outputs = [("Classifier", orange.Classifier),
                        ("Learner", orange.Learner),
                        ("SOM", orngSOM.SOMMap)]

        self.LearnerName = "SOM Map"
        self.xdim = 5
        self.ydim = 10
        self.initialization = orngSOM.InitializeLinear
        self.neighborhood = 0
        self.topology = 0
        self.alphaType = 0
        self.iterations1 = 100
        self.iterations2 = 10000
        self.radius1 = 3
        self.radius2 = 1
        self.eps = 1e-5
        self.alpha1 = 0.05
        self.alpha2 = 0.01
        self.loadSettings()

        self.TopolMap = [orngSOM.HexagonalTopology,
                         orngSOM.RectangularTopology]

        self.NeighMap = [orngSOM.NeighbourhoodGaussian,
                         orngSOM.NeighbourhoodBubble]

        self.learnerName = OWGUI.lineEdit(
            self.controlArea, self, "LearnerName",
            box="Learner/Classifier Name",
            tooltip=("Name to be used by other widgets to identify your "
                     "Learner/Classifier")
        )

        box = OWGUI.radioButtonsInBox(
            self.controlArea, self, "topology",
            ["Hexagonal topology", "Rectangular topology"],
            box="Topology"
        )

        OWGUI.spin(box, self, "xdim", 4, 1000,
                   orientation="horizontal",
                   label="Columns")

        OWGUI.spin(box, self, "ydim", 4, 1000,
                   orientation="horizontal",
                   label="Rows")

        OWGUI.radioButtonsInBox(self.controlArea, self, "initialization",
                                ["Linear", "Random"],
                                box="Map Initialization")

        OWGUI.radioButtonsInBox(self.controlArea, self, "neighborhood",
                                ["Gaussian neighborhood",
                                 "Bubble neighborhood"],
                                box="Neighborhood")

        b = OWGUI.widgetBox(self.controlArea, "Radius")

        OWGUI.spin(b, self, "radius1", 2, 50,
                   orientation="horizontal", label="Initial radius")

        OWGUI.spin(b, self, "radius2", 1, 50,
                   orientation="horizontal", label="Final radius")

        b = OWGUI.widgetBox(self.controlArea, "Stopping Conditions")
        OWGUI.spin(b, self, "iterations1", 10, 10000, label="Iterations")

        OWGUI.button(self.controlArea, self, "&Apply",
                     callback=self.ApplySettings,
                     default=True)

        OWGUI.rubber(self.controlArea)

        self.data = None

        self.resize(100, 100)

    def dataWithDefinedValues(self, data):
        self.warning(1235)
        self.warning(1236)
        exclude = []
        for attr in data.domain.variables:
            if not any(not ex[attr].isSpecial() for ex in data):
                exclude.append(attr)

        if exclude:
            self.warning(1235,
                         "Excluding attributes with all unknown "
                         "values: %s." % \
                         ", ".join(attr.name for attr in exclude))

            exclude_class = data.domain.classVar in exclude
            if exclude_class:
                self.warning(1236,
                             "Excluding class attribute: %s" % \
                             data.domain.classVar.name)

            domain = orange.Domain(
                [attr for attr in data.domain.variables
                 if attr not in exclude],
                data.domain.classVar if not exclude_class else False
            )

            domain.addmetas(data.domain.getmetas())
            data = orange.ExampleTable(domain, data)

        return data

    def setData(self, data=None):
        self.data = data
        if data:
            self.data = self.dataWithDefinedValues(data)
            self.ApplySettings()
        else:
            self.send("Classifier", None)
            self.send("SOM", None)
            self.send("Learner", None)

    def ApplySettings(self):
        topology = self.TopolMap[self.topology]
        neigh = self.NeighMap[self.neighborhood]

        self.learner = orngSOM.SOMLearner(
            name=self.LearnerName,
            map_shape=(self.xdim, self.ydim),
            topology=topology,
            neighbourhood=neigh,
            epochs=self.iterations1,
            eps=self.eps,
            initialize=self.initialization,
            radius_ini=self.radius1,
            radius_fin=self.radius2
        )

        self.send("Learner", self.learner)

        if self.data:
            self.progressBarInit()
            self.classifier = self.learner(
                self.data, progressCallback=self.progressBarSet
            )

            self.progressBarFinished()
            self.classifier.name = self.LearnerName
            self.classifier.setattr("data", self.data)
            if self.data.domain.classVar:
                self.send("Classifier", self.classifier)
            self.send("SOM", self.classifier)

    def sendReport(self):
        self.reportSettings(
            "Topology",
            [("Shape", ["hexagonal", "rectangular"][self.topology]),
             ("Size", "%i columns, %i rows" % (self.xdim, self.ydim))]
        )

        self.reportSettings(
            "Optimization",
            [("Initialization", ["linear", "random"][self.initialization]),
             ("Neighborhood", ["Gaussian", "bubble"][self.neighborhood]),
             ("Radius", "initial: %i, final: %i" % \
              (self.radius1, self.radius2)),
             ("Number of iterations", self.iterations1)
            ])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWSOM()
    w.show()
    data = orange.ExampleTable("../../doc/datasets/iris.tab")

    w.setData(data)
    app.exec_()
