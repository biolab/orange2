"""
<name>SOM</name>
<description>Self organizing maps learner.</description>
<icon>icons/SOM.png</icon>
<contact>Ales Erjavec (ales.erjevec(@at@)fri.uni.lj.si)</contact> 
<priority>5010</priority>
"""

import orange
import orngSOM
from OWWidget import *
import OWGUI

#class orange_input(object):
#    def __init__(self, input_type, check=None):
#        pass
#    
#    def __get__(self, instance):
#        return self.__call__        

class OWSOM(OWWidget):
    settingsList=["xdim", "ydim", "neighborhood", "topology", "alphaType", "iterations1", "iterations2",
                  "radius1", "radius2", "alpha1", "alpha2", "initialization", "eps"]
    def __init__(self ,parent=None , signalManager=None, name="SOM"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea=False)

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Classifier", orange.Classifier), ("Learner", orange.Learner), ("SOMMap", orngSOM.SOMMap)]

        self.LearnerName="SOM Map"
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
##        self.TopolMap=[orangeom.SOMLearner.HexagonalTopology, orangeom.SOMLearner.RectangularTopology]
        self.TopolMap = [orngSOM.HexagonalTopology, orngSOM.RectangularTopology]
##        self.NeighMap=[orangeom.SOMLearner.BubbleNeighborhood, orangeom.SOMLearner.GaussianNeighborhood]
        self.NeighMap = [orngSOM.NeighbourhoodGaussian, orngSOM.NeighbourhoodBubble]
##        self.AlphaMap=[orangeom.SOMLearner.LinearFunction, orangeom.SOMLearner.InverseFunction]
##        self.AlphaMap=[1, 2]
        self.learnerName = OWGUI.lineEdit(self.controlArea, self, "LearnerName", box="Learner/Classifier Name", tooltip="Name to be used by other widgets to identify yor Learner/Classifier")
##        self.learnerName.setText("SOM Classifier")
##        box = OWGUI.widgetBox(self.controlArea, self, "Dimensions")
        box = OWGUI.radioButtonsInBox(self.controlArea, self, "topology", ["Hexagonal topology", "Rectangular topology"], box="Topology")
        OWGUI.spin(box, self, "xdim", 4, 1000, orientation="horizontal", label="Columns")
        OWGUI.spin(box, self, "ydim", 4, 1000, orientation="horizontal", label="Rows")
        OWGUI.radioButtonsInBox(self.controlArea, self, "initialization", ["Linear", "Random"], box="Map Initialization")
        OWGUI.radioButtonsInBox(self.controlArea, self, "neighborhood", ["Gaussian neighborhood", "Bubble neighborhood"], box="Neighborhood")
##        OWGUI.radioButtonsInBox(self.controlArea, self, "alphaType", ["Linear function", "Inverse function"], box="Alpha Function Type")
        b = OWGUI.widgetBox(self.controlArea, "Radius")
        OWGUI.spin(b, self, "radius1", 2,50, orientation="horizontal", label="Initial radius")
        OWGUI.spin(b, self, "radius2", 1,50, orientation="horizontal", label="Final radius")

        b = OWGUI.widgetBox(self.controlArea , "Stoping Conditions")
        OWGUI.spin(b, self, "iterations1", 10, 10000, label="Iterations")
##        OWGUI.doubleSpin(b, self, "eps", 0.0, 1.0, 1e-5, label="Eps")
##        tabW=OWGUI.tabWidget(self.controlArea)
##        b1=OWGUI.createTabPage(tabW, "Step 1")
##        b2=OWGUI.createTabPage(tabW, "Step 2")
##        
##        OWGUI.spin(b1, self, "iterations1", 10, 100000, orientation="horizontal", label="Num. iter.")
##        OWGUI.spin(b2, self, "iterations2", 10, 100000, orientation="horizontal", label="Num. iter.")
##        OWGUI.spin(b1, self, "radius1", 2,1000, orientation="horizontal", label="Radius")
##        OWGUI.spin(b2, self, "radius2", 2,1000, orientation="horizontal", label="Radius")
##        OWGUI.doubleSpin(b1, self, "alpha1", 0.0, 1.0, 0.01, orientation="horizontal", label="Alpha")
##        OWGUI.doubleSpin(b2, self, "alpha2", 0.0, 1.0, 0.01, orientation="horizontal", label="Alpha")
##        self.alpha1=self.alpha1
##        self.alpha2=self.alpha2
        OWGUI.button(self.controlArea, self,  "&Apply", callback=self.ApplySettings)

        OWGUI.rubber(self.controlArea)
        
        self.data = None        
        
        self.resize(100, 100)

    def dataWithDefinedValues(self, data, ):
        self.warning(1235)
        self.warning(1236)
        exclude = []
        for attr in data.domain.variables:
            if not any(not ex[attr].isSpecial() for ex in data):
                exclude.append(attr)
        if exclude:
            self.warning(1235, "Excluding attributes with all unknown values: %s." % ", ".join(attr.name for attr in exclude))
            exclude_class = data.domain.classVar in exclude
            if exclude_class:
                self.warning(1236, "Excluding class attribute: %s" % data.domain.classVar.name) 
            domain = orange.Domain([attr for attr in data.domain.variables if attr not in exclude],
                                   data.domain.classVar if not exclude_class else False)
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
            self.send("SOMMap", None)
            self.send("Learner", None)
        

    def ApplySettings(self):
        topology = self.TopolMap[self.topology]
        neigh = self.NeighMap[self.neighborhood]
##        alphaT=self.AlphaMap[self.alphaType]
##        params=[{"iterations":self.iterations1, "radius":self.radius1, "alpha":self.alpha1},
##                {"iterations":self.iterations2, "radius":self.radius2, "alpha":self.alpha2}]
        self.learner = orngSOM.SOMLearner(name=self.LearnerName, map_shape=(self.xdim, self.ydim), topology=topology, neighborhood=neigh,
                                        epochs=self.iterations1, eps=self.eps, initialize=self.initialization,
                                        radius_ini=self.radius1, radius_fin=self.radius2) #alphaType=alphaT, parameters=params)

        self.send("Learner", self.learner)
        if self.data:
            self.progressBarInit()
            self.classifier = self.learner(self.data, progressCallback=self.progressBarSet)
            self.progressBarFinished()
            self.classifier.name = self.LearnerName
            self.classifier.setattr("data", self.data)
            if self.data.domain.classVar:
                self.send("Classifier", self.classifier)
            self.send("SOMMap", self.classifier)
        

    def sendReport(self):
        self.reportSettings("Topology",
                            [("Shape", ["hexagonal", "rectangular"][self.topology]),
                             ("Size", "%i columns, %i rows" % (self.xdim, self.ydim))
                            ])
        self.reportSettings("Optimization",
                            [("Initialization", ["linear", "random"][self.initialization]),
                             ("Neighborhood", ["Gaussian", "bubble"][self.neighborhood]),
                             ("Radius", "initial: %i, final: %i" % (self.radius1, self.radius2)),
                             ("Number of iterations", self.iterations1)
                            ])
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWSOM()
##    app.setMainWidget(w)
    w.show()
    data = orange.ExampleTable("../../doc/datasets/iris.tab")
    
    w.setData(data)
    app.exec_()
