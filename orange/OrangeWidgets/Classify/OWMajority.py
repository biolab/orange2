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

class OWMajority(OWWidget):
    settingsList = ["name"]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, 'Majority')
        
        self.callbackDeposit = []

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier)]

        self.name = 'Majority'

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
                 tooltip='Name to be used by other widgets to identify your learner/classifier.')

        OWGUI.separator(self.controlArea)
        
        OWGUI.button(self.controlArea, self, "&Apply Setting", callback = self.setLearner, disabled=0)

        self.learner = orange.MajorityLearner()
        self.setLearner()
        self.resize(100,100)

    def setLearner(self):
        self.learner.name = self.name
        self.send("Learner", self.learner)

    def cdata(self,data):
        self.error(0)
        self.error(1)
        if data and not data.domain.classVar:
            self.error(0, "This data set has no class")
            data = None

        self.data = data
        if data:
            try:
                self.classifier = self.learner(data)
                self.classifier.name = self.name
            except Exception, (errValue):
                self.classifier = None
                self.error(1, str(errValue))
        else:
            self.classifier = None
        self.send("Classifier", self.classifier)
            
 
##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWMajority()
    a.setMainWidget(ow)

##    dataset = orange.ExampleTable('adult_sample')
##    ow.cdata(dataset)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
