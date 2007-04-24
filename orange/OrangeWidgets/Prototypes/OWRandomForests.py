"""
<name>Random Forests</name>
<description>Random Forests learner/classifier.</description>
<icon>icons/RandomForests.png</icon>
<contact>Marko toplak (marko.toplak(@at@)gmail.com)</contact>
<priority>6060</priority>
"""

from OWWidget import *
import orngTree, OWGUI
import orngEnsemble
from exceptions import Exception

class OWRandomForests(OWWidget):
    settingsList = ["name", "trees", "attributes", "attributesP", "preNodeInst", "preNodeInstP", "limitDepth", "limitDepthP", "rseed" ]

    def __init__(self, parent=None, signalManager = None, name='Random Forests'):
        OWWidget.__init__(self, parent, signalManager, name)

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Learner", orange.Learner),("Random Forests Classifier", orange.Classifier)]

        self.name = 'Random Forests'
	self.trees = 10
	self.attributes = 0
	self.attributesP = 5
	self.preNodeInst = 1
	self.preNodeInstP = 5
	self.limitDepth = 0
	self.limitDepthP = 3
	self.rseed = 0

        self.loadSettings()

        self.data = None
        self.preprocessor = None
        self.setLearner()

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', tooltip='Name to be used by other widgets to identify your learner/classifier.')

        OWGUI.separator(self.controlArea)

        self.bBox = QVGroupBox(self.controlArea)
        self.bBox.setTitle('Basic properties')

        self.treesBox = OWGUI.spin(self.bBox, self, "trees", 1, 10000, orientation="horizontal", label="Number of trees in forest  ")
	self.attributesBox, self.attributesPBox = OWGUI.checkWithSpin(self.bBox, self, "Consider exactly", 1, 10000, "attributes", "attributesP", " random attributes at each split.")
        self.rseedBox = OWGUI.spin(self.bBox, self, "rseed", 0, 100000, orientation="horizontal", label="Seed for random generator ")

        OWGUI.separator(self.controlArea)

        self.pBox = QVGroupBox(self.controlArea)
        self.pBox.setTitle('Growth Control')

	self.limitDepthBox, self.limitDepthPBox = OWGUI.checkWithSpin(self.pBox, self, "Maximum depth of individual trees  ", 1, 1000, "limitDepth", "limitDepthP", "")
        self.preNodeInstBox, self.preNodeInstPBox = OWGUI.checkWithSpin(self.pBox, self, "Stop splitting nodes with ", 1, 1000, "preNodeInst", "preNodeInstP", " or fewer instances")
        OWGUI.separator(self.controlArea)
        self.btnApply = OWGUI.button(self.controlArea, self, "&Apply Changes", callback = self.setLearner, disabled=0)

        self.resize(100,200)

    def pbchange(self, val):
	self.progressBarSet(val*100)

    def setLearner(self):

	self.progressBarInit()

        if hasattr(self, "btnApply"):
            self.btnApply.setFocus()

	#assemble learner
	rand = random.Random(self.rseed)
	
        attrs = None
        if self.attributes: attrs = self.attributesP

	self.learner = orngEnsemble.RandomForestLearner(trees = self.trees, rand=rand, attributes=attrs, callback=self.pbchange)
	
	if self.preNodeInst: self.learner.learner.stop.minExamples = self.preNodeInstP 
        else: self.learner.learner.stop.minExamples = 0

	if self.limitDepth: self.learner.learner.maxDepth = self.limitDepthP

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


	self.progressBarFinished()

        self.send("Random Forests Classifier", self.classifier)

    def setData(self,data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete) and data or None
        self.setLearner()


##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRandomForests()
    a.setMainWidget(ow)

    d = orange.ExampleTable('adult_sample')
    ow.setData(d)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
