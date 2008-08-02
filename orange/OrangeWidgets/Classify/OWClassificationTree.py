"""
<name>Classification Tree</name>
<description>Classification tree learner/classifier.</description>
<icon>icons/ClassificationTree.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>30</priority>
"""
import orngOrangeFoldersQt4
from OWWidget import *
import orngTree, OWGUI
from exceptions import Exception

import warnings
warnings.filterwarnings("ignore", ".*this class is not optimized for 'candidates' list and can be very slow.*", orange.KernelWarning, ".*orngTree", 34)



class OWClassificationTree(OWWidget):
    settingsList = ["name",
                    "estim", "relK", "relM",
                    "bin", "subset",
                    "preLeafInst", "preNodeInst", "preNodeMaj",
                    "preLeafInstP", "preNodeInstP", "preNodeMajP",
                    "postMaj", "postMPruning", "postM"]

    measures = (("Information Gain", "infoGain"), ("Gain Ratio", "gainRatio"), ("Gini Index", "gini"), ("ReliefF", "relief"))

    def __init__(self, parent=None, signalManager = None, name='Classification Tree'):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Learner", orange.TreeLearner),("Classification Tree", orange.TreeClassifier)]

        self.name = 'Classification Tree'
        self.estim = 0; self.relK = 5; self.relM = 100; self.limitRef = True
        self.bin = 0; self.subset = 0
        self.preLeafInstP = 2; self.preNodeInstP = 5; self.preNodeMajP = 95
        self.preLeafInst = 1; self.preNodeInst = 0; self.preNodeMaj = 0
        self.postMaj = 1; self.postMPruning = 1; self.postM = 2.0

        self.loadSettings()

        self.data = None
        self.preprocessor = None
        self.setLearner()

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)

        qBox = OWGUI.widgetBox(self.controlArea, 'Attribute selection criterion')

        self.qMea = OWGUI.comboBox(qBox, self, "estim", items = [m[0] for m in self.measures], callback = self.measureChanged)

        b1 = OWGUI.widgetBox(qBox, orientation = "horizontal")
        OWGUI.separator(b1, 16, 0)
        b2 = OWGUI.widgetBox(b1)
        self.cbLimitRef, self.hbxRel1 = OWGUI.checkWithSpin(b2, self, "Limit the number of reference examples to ", 1, 1000, "limitRef", "relM")
        OWGUI.separator(b2)
        self.hbxRel2 = OWGUI.spin(b2, self, "relK", 1, 50, orientation="horizontal", label="Number of neighbours in ReliefF  ")
        OWGUI.separator(self.controlArea)

        OWGUI.radioButtonsInBox(self.controlArea, self, 'bin', ["No binarization", "Exhaustive search for optimal split", "One value against others"], "Binarization")
        OWGUI.separator(self.controlArea)

        self.measureChanged()

        self.pBox = OWGUI.widgetBox(self.controlArea, 'Pre-Pruning')

        self.preLeafInstBox, self.preLeafInstPBox = OWGUI.checkWithSpin(self.pBox, self, "Min. instances in leaves ", 1, 1000, "preLeafInst", "preLeafInstP")
        self.preNodeInstBox, self.preNodeInstPBox = OWGUI.checkWithSpin(self.pBox, self, "Stop splitting nodes with less instances than ", 1, 1000, "preNodeInst", "preNodeInstP")
        self.preNodeMajBox, self.preNodeMajPBox = OWGUI.checkWithSpin(self.pBox, self, "Stop splitting nodes with a majority class of (%)", 1, 100, "preNodeMaj", "preNodeMajP")

        OWGUI.separator(self.controlArea)
        self.mBox = OWGUI.widgetBox(self.controlArea, 'Post-Pruning')

        OWGUI.checkBox(self.mBox, self, 'postMaj', 'Recursively merge leaves with same majority class')
        self.postMPruningBox, self.postMPruningPBox = OWGUI.checkWithSpin(self.mBox, self, "Pruning with m-estimate, m=", 0, 1000, 'postMPruning', 'postM')

        OWGUI.separator(self.controlArea)
        self.btnApply = OWGUI.button(self.controlArea, self, "&Apply", callback = self.setLearner, disabled=0)
        self.resize(200,200)


    def setLearner(self):
        if hasattr(self, "btnApply"):
            self.btnApply.setFocus()
        self.learner = orngTree.TreeLearner(measure = self.measures[self.estim][1],
            reliefK = self.relK, reliefM = self.limitRef and self.relM or -1,
            binarization = self.bin,
            minExamples = self.preNodeInst and self.preNodeInstP,
            minSubset = self.preLeafInst and self.preLeafInstP,
            maxMajority = self.preNodeMaj and self.preNodeMajP/100.0 or 1.0,
            sameMajorityPruning = self.postMaj,
            mForPruning = self.postMPruning and self.postM,
            storeExamples = 1)

        self.learner.name = self.name
        self.send("Learner", self.learner)

        self.error()
        if self.data:
            try:
                self.classifier = self.learner(self.data)
                self.classifier.name = self.name
            except Exception, (errValue):
                self.error(str(errValue))
                self.classifier = None
        else:
            self.classifier = None

        self.send("Classification Tree", self.classifier)


    def measureChanged(self):
        relief = self.estim == 3
        self.hbxRel1.setEnabled(relief and self.limitRef)
        self.hbxRel2.setEnabled(relief)
        self.cbLimitRef.setEnabled(relief)

    def setData(self,data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete) and data or None
        self.setLearner()


##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWClassificationTree()

    #d = orange.ExampleTable('adult_sample')
    #ow.setData(d)

    ow.show()
    a.exec_()
    ow.saveSettings()
