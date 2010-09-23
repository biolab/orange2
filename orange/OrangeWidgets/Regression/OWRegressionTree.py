"""
<name>Regression Tree</name>
<description>Constructs a tree regression learner and given data a regression tree classifier
</description>
<icon>RegressionTree.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>100</priority>
"""

import orange
import orngTree
import OWGUI
from OWWidget import *
import sys

from orngWrap import PreprocessedLearner

class OWRegressionTree(OWWidget):
    settingsList=["Name","MinInstCheck", "MinInstVal", "MinNodeCheck", "MinNodeVal,"
                  "MaxMajCheck", "MaxMajVal", "PostMaj", "PostMPCheck", "PostMPVal", "Bin"]
    callbackDeposit=[]
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Regression Tree", wantMainArea = 0, resizingEnabled = 0)
        self.Name="Regression Tree"
        self.MinInstCheck=1
        self.MinInstVal=5
        self.MinNodeCheck=1
        self.MinNodeVal=10
        self.MaxMajCheck=1
        self.MaxMajVal=70
        self.PostMaj=1
        self.PostMPCheck=1
        self.PostMPVal=5
        self.Bin=1
        self.loadSettings()

        self.data=None
        self.preprocessor = None

        self.inputs=[("Example Table",ExampleTable,self.dataset), ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        self.outputs=[("Learner",orange.Learner),("Regressor",orange.Classifier),("Regression Tree",orange.TreeClassifier)]

        ##
        #GUI
        ##
        OWGUI.lineEdit(self.controlArea, self, "Name", box="Learner/Classifier name")

        OWGUI.separator(self.controlArea)
        OWGUI.checkBox(self.controlArea, self, "Bin", label="Binarization", box ="Tree structure")

        OWGUI.separator(self.controlArea)
        self.prePBox=OWGUI.widgetBox(self.controlArea, "Pre-Pruning")

        #OWGUI.checkWithSpin(self.prePBox, self, "Min. instances in leaves: ", 1, 1000,
        #                    "MinInstCheck", "MinInstVal")

        OWGUI.checkWithSpin(self.prePBox, self, "Do not split nodes with less instances than", 1, 1000,
                            "MinNodeCheck", "MinNodeVal")

        #OWGUI.checkWithSpin(self.prePBox, self, "Stop splitting nodes with ", 1, 100,
        #                    "MaxMajCheck", "MaxMajVal", "% of majority class")

        #OWGUI.checkBox(self.postPBox, self, 'PostMaj', 'Recursively merge leaves with same majority class')

        OWGUI.separator(self.controlArea)
        self.postPBox=OWGUI.widgetBox(self.controlArea, "Post-Pruning")
        OWGUI.checkWithSpin(self.postPBox, self, "Pruning with m-estimate, m:", 0, 1000, 'PostMPCheck', 'PostMPVal')

        OWGUI.button(self.controlArea, self, "&Apply settings",callback=self.setLearner)
        
        OWGUI.rubber(self.controlArea)
        self.setLearner()
        self.resize(100,100)

    def sendReport(self):
        self.reportSettings("Learning parameters",
                            [("Binarization", OWGUI.YesNo[self.Bin]),
                             ("Minimal number of instances in internal nodes", self.MinNodeVal if self.MinNodeCheck else "Not limited"),
                             ("Pruning with m-estimate", "m=%i" % self.PostMPVal if self.PostMPCheck else "No")])
        self.reportData(self.data)
        

    def setLearner(self):
        learner=orngTree.TreeLearner(measure="retis",
                         binarization=self.Bin,
                         mForPruning=self.PostMPCheck and self.PostMPVal,
                         minExamples=self.MinNodeCheck and self.MinNodeVal,
                         storeExamples=1)
        if self.preprocessor:
            learner = self.preprocessor.wrapLearner(learner) 
        learner.name=self.Name
        self.send("Learner",learner)
        self.error()

        classifier = None
        
        if self.data:
            try:
                classifier=learner(self.data)
                classifier.name=self.Name
            except orange.KernelException, (errValue):
                self.error(str(errValue))
                classifier = None
        
        self.send("Regressor", classifier)
        self.send("Regression Tree", classifier)
        #orngTree.printTxt(classifier)

    def dataset(self, data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Continuous, checkMissing=True) and data or None
#        if data and self.isDataWithClass(data, orange.VarTypes.Continuous, checkMissing=True):
        self.setLearner()
#        else:
#            self.data = None
#            self.send("Learner",None)
#            self.send("Regressor",None)
#            self.send("Regression Tree", None)

    def setPreprocessor(self, pp):
        self.preprocessor = pp
        self.setLearner()

if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWRegressionTree()
    #data=orange.ExampleTable("../../doc/datasets/housing.tab")
    #w.dataset(data)
    w.show()
    app.exec_()
