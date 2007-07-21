"""
<name>SOM</name>
<description>Self organizing maps</description>
<icon>SOM.png</icon>
<contact>Ales Erjavec (ales.erjevec324(@at@)email.si)</contact> 
<priority>5010</priority>
"""

import orange, orangeom
import orngSOM
from OWWidget import *
import OWGUI

class OWSOM(OWWidget):
    settingsList=["xdim", "ydim", "neighborhood", "topology", "alphaType", "iterations1", "iterations2",
                  "radius1", "radius2", "alpha1", "alpha2"]
    def __init__(self ,parent=None , signalManager=None, name="SOM"):
        OWWidget.__init__(self, parent, signalManager, name)

        self.inputs=[("Examples", ExampleTable, self.data)]
        self.outputs=[("Classifier", orange.Classifier), ("Learner", orange.Learner), ("SOMMap", orangeom.SOMMap)]

        self.LearnerName=""
        self.xdim=5
        self.ydim=10
        self.neighborhood=0
        self.topology=0
        self.alphaType=0
        self.iterations1=1000
        self.iterations2=10000
        self.radius1=5
        self.radius2=5
        self.alpha1=0.05
        self.alpha2=0.01
        self.loadSettings()
        self.TopolMap=[orangeom.SOMLearner.HexagonalTopology, orangeom.SOMLearner.RectangularTopology]
        self.NeighMap=[orangeom.SOMLearner.BubbleNeighborhood, orangeom.SOMLearner.GaussianNeighborhood]
        self.AlphaMap=[orangeom.SOMLearner.LinearFunction, orangeom.SOMLearner.InverseFunction]
        self.learnerName=OWGUI.lineEdit(self.controlArea, self, "LearnerName", box="Learner/Classifier Name", tooltip="Name to be used by other widgets to identify yor Learner/Classifier")
        self.learnerName.setText("SOM Classifier")
        box=OWGUI.widgetBox(self.controlArea, self, "Dimensions")
        OWGUI.spin(box, self, "xdim", 4, 1000, orientation="horizontal", label="Columns")
        OWGUI.spin(box, self, "ydim", 4, 1000, orientation="horizontal", label="Rows")
        OWGUI.radioButtonsInBox(self.controlArea, self, "topology", ["Hexagonal topology", "Rectangular topology"], box="Topology")
        OWGUI.radioButtonsInBox(self.controlArea, self, "neighborhood", ["Bubble neighborhood","Gaussian neighborhood"], box="Neighborhood")
        OWGUI.radioButtonsInBox(self.controlArea, self, "alphaType", ["Linear function", "Inverse function"], box="Alpha Function Type")
        tabW=QTabWidget(self.controlArea)
        b1=OWGUI.widgetBox(self.controlArea)
        b2=OWGUI.widgetBox(self.controlArea)
        tabW.addTab(b1,"Step 1")
        tabW.addTab(b2,"Step 2")
        OWGUI.spin(b1, self, "iterations1", 10, 100000, orientation="horizontal", label="Num. iter.")
        OWGUI.spin(b2, self, "iterations2", 10, 100000, orientation="horizontal", label="Num. iter.")
        OWGUI.spin(b1, self, "radius1", 2,1000, orientation="horizontal", label="Radius")
        OWGUI.spin(b2, self, "radius2", 2,1000, orientation="horizontal", label="Radius")
        OWGUI.doubleSpin(b1, self, "alpha1", 0.0, 1.0, 0.01, orientation="horizontal", label="Alpha")
        OWGUI.doubleSpin(b2, self, "alpha2", 0.0, 1.0, 0.01, orientation="horizontal", label="Alpha")
        self.alpha1=self.alpha1
        self.alpha2=self.alpha2
        OWGUI.button(self.controlArea, self,  "&Apply", callback=self.ApplySettings)
        
        self.resize(100,100)

        
    def data(self, data=None):
        self.data=data
        if data:
            self.ApplySettings()
        else:
            self.send("Classifier", None)
            self.send("SOMMap", None)
            self.send("Learner", None)
        

    def ApplySettings(self):
        topology=self.TopolMap[self.topology]
        neigh=self.NeighMap[self.neighborhood]
        alphaT=self.AlphaMap[self.alphaType]
        params=[{"iterations":self.iterations1, "radius":self.radius1, "alpha":self.alpha1},
                {"iterations":self.iterations2, "radius":self.radius2, "alpha":self.alpha2}]
        self.learner=orngSOM.SOMLearner(xDim=self.xdim, yDim=self.ydim, topology=topology, neighborhood=neigh, alphaType=alphaT, parameters=params)
        
        self.classifier=self.learner(self.data)
        self.classifier.name=self.LearnerName
        self.classifier.setattr("data", self.data)
        if self.data.domain.classVar:
            self.send("Classifier", self.classifier)
        self.send("SOMMap", self.classifier)
        self.send("Learner", self.learner)
        
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWSOM()
    app.setMainWidget(w)
    w.show()
    data=orange.ExampleTable("../../doc/datasets/iris.tab")
    
    w.data(data)
    app.exec_loop()
