"""
<name>Majority</name>
<description>Constructs a learner that always predicts the majority class</description>
<category>Classification</category>
<icon>icons/Majority.png</icon>
<priority>50</priority>
"""

from OWWidget import *
import OWGUI

class OWMajority(OWWidget):
    settingsList = ["name"]

    def __init__(self, parent=None, name='Majority'):
        OWWidget.__init__(self, parent, name, "Constructs a learner that always predicts the majority class", icon="Majority.png")
        
        self.callbackDeposit = []

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1)]
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
        self.data = data
        if data:
            print "learning"
            self.classifier = self.learner(data)
            self.classifier.name = self.name
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