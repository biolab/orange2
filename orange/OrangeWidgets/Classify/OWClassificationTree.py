"""
<name>Classification Tree</name>
<description>Classification tree learner/classifier.</description>
<icon>icons/ClassificationTree.png</icon>
<priority>30</priority>
"""

from OWWidget import *
import orngTree
import OWGUI

class OWClassificationTree(OWWidget):
    settingsList = ["name",
                    "estim", "relK", "relM",
                    "bin", "subset",
                    "preLeafInst", "preNodeInst", "preNodeMaj",
                    "preLeafInstP", "preNodeInstP", "preNodeMajP",
                    "postMaj", "postMPruning", "postM"]

    # If you change this, you need to change measureChanged as well,
    # because it enables/disables two widgets when ReliefF is chosen
    measures = (("Information Gain", "infoGain"),
                ("Gain Ratio", "gainRatio"),
                ("Gini Index", "gini"),
                ("ReliefF", "relief"))
    
    def __init__(self, parent=None, signalManager = None, name='Classification Tree'):
        OWWidget.__init__(self, parent, signalManager, name)

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.dataset)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier),("Classification Tree", orange.TreeClassifier)]

        # Settings
        self.name = 'Classification Tree'
        self.estim = 0; self.relK = 5; self.relM = 100
        self.bin = 0; self.subset = 0
        self.preLeafInstP = 2; self.preNodeInstP = 5; self.preNodeMajP = 95
        self.preLeafInst = 1; self.preNodeInst = 0; self.preNodeMaj = 0
        self.postMaj = 1; self.postMPruning = 1; self.postM = 2.0
        
        self.loadSettings()
        
        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        self.setLearner()                   # this just sets the learner, no data
                                            # has come to the input yet
        
        # GUI
        # name
        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
                 tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)
        
        # attribute quality estimation
        qBox = QVGroupBox(self.controlArea)
        qBox.setTitle('Attribute Quality Estimation')

        self.qMea = QComboBox(qBox)
        for m in self.measures:
            self.qMea.insertItem(m[0])
        self.qMea.setCurrentItem(self.estim)
        self.connect(self.qMea, SIGNAL("activated(int)"), self.measureChanged)
        
        self.hbxRel1 = OWGUI.spin(qBox, self, "relM", 1, 1000, 10, label="Relief's reference examples: ")
        self.hbxRel2 = OWGUI.spin(qBox, self, "relK", 1, 50, label="Relief's neighbours")
        OWGUI.separator(self.controlArea)
        
        # structure of the tree
        self.cbBin = OWGUI.checkBox(self.controlArea, self, 'bin', 'Binarization', box='Tree Structure')
        OWGUI.separator(self.controlArea)

        self.measureChanged(self.estim)

        # prepruning
        self.pBox = QVGroupBox(self.controlArea)
        self.pBox.setTitle('Pre-Pruning')

        self.preLeafInstBox, self.preLeafInstPBox = \
          OWGUI.checkWithSpin(self.pBox, self, "Min. instances in leaves: ", 1, 1000, "preLeafInst", "preLeafInstP")
        self.preNodeInstBox, self.preNodeInstPBox = \
          OWGUI.checkWithSpin(self.pBox, self, "Stop splitting nodes with ", 1, 1000, "preNodeInst", "preNodeInstP", " or fewer instances")
        self.preNodeMajBox, self.preNodeMajPBox = \
          OWGUI.checkWithSpin(self.pBox, self, "Stop splitting nodes with ", 1, 100, "preNodeMaj", "preNodeMajP", "% of majority class")
        
        OWGUI.separator(self.controlArea)
        self.mBox = QVGroupBox(self.controlArea)

        # post-pruning
        self.mBox.setTitle('Post-Pruning')
        OWGUI.checkBox(self.mBox, self, 'postMaj', 'Recursively merge leaves with same majority class')
        self.postMPruningBox, self.postMPruningPBox = \
          OWGUI.checkWithSpin(self.mBox, self, "m for m-error pruning ", 0, 1000, 'postMPruning', 'postM')

        # apply button
        OWGUI.separator(self.controlArea)
        self.btnApply = OWGUI.button(self.controlArea, self, "&Apply Changes", callback = self.setLearner, disabled=0)

        self.resize(100,400)

    # main part:         

    def setLearner(self):
        if hasattr(self, "btnApply"):
            self.btnApply.setFocus()
        self.learner = orngTree.TreeLearner(measure = self.measures[self.estim][1],
            reliefK = self.relK, reliefM = self.relM,
            binarization = self.bin,
            sameMajorityPruning = self.postMaj,
            storeExamples = 1)
        if self.preNodeInst:
            self.learner.minExamples = self.preNodeInstP
        if self.preLeafInst:
            self.learner.minSubset = self.preLeafInstP
        if self.preNodeMaj:
            self.learner.maxMajority = self.preNodeMajP / 100.0
        if self.postMPruning:
            self.learner.mForPruning = self.postM

        self.learner.name = self.name
        self.send("Learner", self.learner)
        if self.data <> None:
            self.classifier = self.learner(self.data)
            self.classifier.name = self.name
            self.send("Classifier", self.classifier)
            self.send("Classification Tree", self.classifier)

    def measureChanged(self, idx):
        self.estim = idx
        self.hbxRel1.setEnabled(idx == 3)
        self.hbxRel2.setEnabled(idx == 3)
        self.cbBin.setEnabled(idx != 3)
        if idx==3:
            self.prevBinState = self.bin
            self.bin = 0
        else:
            if hasattr(self, "prevBinState"):
                self.bin = self.prevBinState
        
    # handle input signals        

    def dataset(self,data):
        self.data = data
        if self.data:
            self.setLearner()
        else:
            self.send("Classifier", None)
            self.send("Classification Tree", None)

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWClassificationTree()
    a.setMainWidget(ow)

    d = orange.ExampleTable('adult_sample')
    ow.dataset(d)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
