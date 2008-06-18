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
    settingsList = ["name", "trees", "attributes", "attributesP", "preNodeInst", "preNodeInstP", "limitDepth", "limitDepthP", "rseed", "outtree" ]

    def __init__(self, parent=None, signalManager = None, name='Random Forests'):
        OWWidget.__init__(self, parent, signalManager, name)

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Learner", orange.Learner),("Random Forests Classifier", orange.Classifier),("Choosen Tree", orange.TreeClassifier) ]

        self.name = 'Random Forests'
	self.trees = 10
	self.attributes = 0
	self.attributesP = 5
	self.preNodeInst = 1
	self.preNodeInstP = 5
	self.limitDepth = 0
	self.limitDepthP = 3
	self.rseed = 0
	self.outtree = 0


	self.maxTrees = 10000

        self.loadSettings()

        self.data = None
        self.preprocessor = None

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', tooltip='Name to be used by other widgets to identify your learner/classifier.')

        OWGUI.separator(self.controlArea)

        self.bBox = QVGroupBox(self.controlArea)
        self.bBox.setTitle('Basic Properties')

        self.treesBox = OWGUI.spin(self.bBox, self, "trees", 1, self.maxTrees, orientation="horizontal", label="Number of trees in forest  ")
	self.attributesBox, self.attributesPBox = OWGUI.checkWithSpin(self.bBox, self, "Consider exactly", 1, 10000, "attributes", "attributesP", " random attributes at each split.")
        self.rseedBox = OWGUI.spin(self.bBox, self, "rseed", 0, 100000, orientation="horizontal", label="Seed for random generator ")

        OWGUI.separator(self.controlArea)

        self.pBox = QVGroupBox(self.controlArea)
        self.pBox.setTitle('Growth Control')

	self.limitDepthBox, self.limitDepthPBox = OWGUI.checkWithSpin(self.pBox, self, "Maximum depth of individual trees  ", 1, 1000, "limitDepth", "limitDepthP", "")
        self.preNodeInstBox, self.preNodeInstPBox = OWGUI.checkWithSpin(self.pBox, self, "Stop splitting nodes with ", 1, 1000, "preNodeInst", "preNodeInstP", " or fewer instances")

        OWGUI.separator(self.controlArea)

        #self.sBox = QVGroupBox(self.controlArea)
        #self.sBox.setTitle('Single Tree Output')

	self.streesBox = OWGUI.spin(self.controlArea, self, "outtree", -1, self.maxTrees, orientation="horizontal", label="Index of tree on the output ", callback=[self.period, self.extree])
	#self.streesBox.setDisabled(True)
	self.streeEnabled(False)

	OWGUI.separator(self.controlArea)

	self.btnApply = OWGUI.button(self.controlArea, self, "&Apply Changes", callback = self.setLearner, disabled=0)

        self.resize(100,200)

	self.setLearner()

    def period(self):
	if self.outtree == -1: self.outtree = self.claTrees-1
	elif self.outtree >= self.claTrees: self.outtree = 0

    def extree(self):
	self.send("Choosen Tree", self.classifier.classifiers[self.outtree])

    def streeEnabled(self, status):
	if status:
		self.claTrees = self.trees
		self.streesBox.setDisabled(False)
		self.period()
		self.extree()
	else:
		#a = 1
		self.streesBox.setDisabled(True)

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

	self.learner.learner.storeExamples = 1
	self.learner.learner.storeNodeClassifier = 1
	self.learner.learner.storeContigencies = 1
	self.learner.learner.storeDistributions = 1

	if self.limitDepth: self.learner.learner.maxDepth = self.limitDepthP

        self.learner.name = self.name
        self.send("Learner", self.learner)

        self.error()

        if self.data:
            try:
                self.classifier = self.learner(self.data)
                self.classifier.name = self.name
		self.streeEnabled(True)
            except Exception, (errValue):
                self.error(str(errValue))
                self.classifier = None
		self.streeEnabled(False)
        else:
		self.classifier = None
		self.streeEnabled(False)

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

    d = orange.ExampleTable('adult_sample')
    ow.setData(d)

    ow.show()
    a.exec_()
    ow.saveSettings()
