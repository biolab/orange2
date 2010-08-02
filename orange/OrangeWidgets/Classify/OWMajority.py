"""
<name>Majority</name>
<description>Majority class learner/classifier.</description>
<icon>icons/Majority.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>20</priority>
"""
from OWWidget import *
import OWGUI
from exceptions import Exception

from orngWrap import PreprocessedLearner
class OWMajority(OWWidget):
    settingsList = ["name"]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, 'Majority', wantMainArea = 0, resizingEnabled = 0)

        self.callbackDeposit = []

        self.inputs = [("Examples", ExampleTable, self.setData), ("Preprocessing", PreprocessedLearner, self.setPreprocessor)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier)]

        self.name = 'Majority'
        
        self.data = None
        self.preprocessor = None

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
                 tooltip='Name to be used by other widgets to identify your learner/classifier.')

        OWGUI.separator(self.controlArea)

        OWGUI.button(self.controlArea, self, "&Apply", callback = self.setLearner, disabled=0)
        
        OWGUI.rubber(self.controlArea)

        self.learner = orange.MajorityLearner()
        self.setLearner()
        self.resize(100,100)

    def setLearner(self):
        if self.preprocessor:
            self.learner = self.preprocessor.wrapLearner(orange.MajorityLearner())
        self.learner.name = self.name
        self.send("Learner", self.learner)

    def setData(self,data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete, checkMissing=True) and data or None

        if self.data:
            try:
                self.classifier = self.learner(self.data)
                self.classifier.name = self.name
                self.error(1)
            except Exception, (errValue):
                self.classifier = None
                self.error(1, str(errValue))
        else:
            self.classifier = None
        self.send("Classifier", self.classifier)
        
    def setPreprocessor(self, pp):
        self.preprocessor = pp
        self.setLearner()
        self.setData(self.data)

    def sendReport(self):
        self.reportData(self.data)

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWMajority()

##    dataset = orange.ExampleTable('adult_sample')
##    ow.setData(dataset)

    ow.show()
    a.exec_()
    ow.saveSettings()
