"""
<name>Classification Tree</name>
<description>Classification Tree widget can construct a classification tree learner or (if given
a data set) a classification tree classifier.</description>
<category>Classification</category>
<icon>icons/ClassificationTree.png</icon>
<priority>50</priority>
"""

from OWWidget import *
from orngTreeLearner import TreeLearner
from OWGUI import *

class OWClassificationTree(OWWidget):
    settingsList = ["name",
                    "estim", "relK", "relM",
                    "bin", "subset",
                    "preLeafInst", "preNodeInst", "preNodeMaj",
                    "preLeafInstP", "preNodeInstP", "preNodeMajP",
                    "postMaj", "postMPrunning", "postM"]

    # If you change this, you need to change measureChanged as well,
    # because it enables/disables two widgets when ReliefF is chosen
    measures = (("Information Gain", "infoGain"),
                ("Gain Ratio", "gainRatio"),
                ("Gini Index", "gini"),
                ("ReliefF", "relief"))
    
    def __init__(self, parent=None, name='Classification Tree'):
        OWWidget.__init__(self,
        parent,
        name,
        """ClassificationTree widget can either \nconstruct a classification tree leraner, or,
if given a data set, a classification tree classifier. \nIt can also be combined with
preprocessors to filter/change the data.
""",
        FALSE,
        FALSE)
        
        self.callbackDeposit = []

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier),("Classification Tree", orange.TreeClassifier)]

        # Settings
        self.name = 'Classification Tree'
        self.estim = 0; self.relK = 5; self.relM = 100
        self.bin = 0; self.subset = 0
        self.preLeafInstP = 2; self.preNodeInstP = 5; self.preNodeMajP = 95
        self.preLeafInst = 1; self.preNodeInst = 0; self.preNodeMaj = 0
        self.postMaj = 0; self.postMPrunning = 0; self.postM = 2.0
        
        self.loadSettings()
        
        
        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        self.setLearner()                   # this just sets the learner, no data
                                            # has come to the input yet
        
        # GUI
        # name

        self.nameBox = QVGroupBox(self.controlArea)
        self.nameBox.setTitle('Learner/Classifier Name')
        QToolTip.add(self.nameBox,"Name to be used by other widgets to identify your learner/classifier.")
        lineEditOnly(self.nameBox, self, '', 'name')
        QWidget(self.controlArea).setFixedSize(0, 16)
        
        # attribute quality estimation
        self.qBox = QVGroupBox(self.controlArea)
        self.qBox.setTitle('Attribute Quality Estimation')

        self.qMea = QComboBox(self.qBox)
        for m in self.measures:
            self.qMea.insertItem(m[0])
        self.qMea.setCurrentItem(self.estim)
        self.connect(self.qMea, SIGNAL("activated(int)"), self.measureChanged)
        
        self.hbxRel1 = labelWithSpin_hb(self.qBox, self, "Relief's reference examples: ", 1, 1000, "relM", 10)
        self.hbxRel2 = labelWithSpin_hb(self.qBox, self, "Relief's neighbours", 1, 50, "relK")
        QWidget(self.controlArea).setFixedSize(0, 16)
        self.measureChanged(self.estim)

        # structure of the tree
        self.sBox = QVGroupBox(self.controlArea)
        self.sBox.setTitle('Tree Structure')
        checkOnly(self.sBox, self, 'Binarization', 'bin')
        QWidget(self.controlArea).setFixedSize(0, 16)

        # preprunning
        self.pBox = QVGroupBox(self.controlArea)
        self.pBox.setTitle('Pre-Prunning')

        self.preLeafInstBox, self.preLeafInstPBox = \
          checkWithSpin(self.pBox, self, "Min. instances in leaves: ", 1, 1000, "preLeafInst", "preLeafInstP")
        self.preNodeInstBox, self.preNodeInstPBox = \
          checkWithSpin(self.pBox, self, "Stop splitting nodes with ", 1, 1000, "preNodeInst", "preNodeInstP", " or fewer instances")
        self.preNodeMajBox, self.preNodeMajPBox = \
          checkWithSpin(self.pBox, self, "Stop splitting nodes with ", 1, 100, "preNodeMaj", "preNodeMajP", "% of majority class")
        
        QWidget(self.controlArea).setFixedSize(0, 16)

        self.mBox = QVGroupBox(self.controlArea)

        # post-prunning
        self.mBox.setTitle('Post-Prunning')
        checkOnly(self.mBox, self, 'Recursively merge leaves with same majority class', 'postMaj')
        self.postMPrunningBox, self.postMPrunningPBox = \
          checkWithSpin(self.mBox, self, "m for m-error prunning ", 0, 1000, 'postMPrunning', 'postM')

        QWidget(self.controlArea).setFixedSize(0, 16)

        # apply button
        self.applyBtn = QPushButton("&Apply Changes", self.controlArea)
        self.connect(self.applyBtn, SIGNAL("clicked()"), self.setLearner)

        self.resize(100,550)

    # main part:         

    def setLearner(self):
        #print 'MinEx', self.preNodeInst, self.preNodeInstP, '|', self.preLeafInst, self.preLeafInstP
        self.learner = TreeLearner(measure = self.measures[self.estim][1],
                                   reliefK = self.relK, reliefM = self.relM,
                                   binarization = self.bin,
                                   minExamples = self.preNodeInst and self.preNodeInstP,
                                   minSubset = self.preLeafInst and self.preLeafInstP,
                                   maxMajority = self.preNodeMaj and self.preNodeMajP/100.0,
                                   sameMajorityPruning = self.postMaj,
                                   mForPrunning = self.postMPrunning and self.postM,
                                   storeExamples = 1)
                                   
        self.learner.name = self.name
        self.send("Learner", self.learner)
        if self.data <> None:
            self.classifier = self.learner(self.data)
            self.classifier.name = self.name
            self.send("Classifier", self.classifier)
            self.send("Classification Tree", self.classifier)

    # slots: handle input signals        
    def measureChanged(self, idx):
        self.estim = idx
        self.hbxRel1.setEnabled(idx == 3)
        self.hbxRel2.setEnabled(idx == 3)
        
        
    def cdata(self,data):
        self.data=data
        self.setLearner()

    def pp():
        pass
        # include preprocessing!!!

    # signal processing

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWClassificationTree()
    a.setMainWidget(ow)

    dataset = orange.ExampleTable('../adult_sample')
    ow.cdata(dataset)

    ow.show()
    a.exec_loop()
    ow.saveSettings()