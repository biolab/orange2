"""
<name>Test learners</name>
<description>TestLearners widget can take learners on the input and tests them on a single data set. Alternatively, it can
evaluate already build classifiers. For testing learners, it implements different sampling techniques.
It reports on variety of statistics, including accuracy, sensitivity, specificity, information score,
AUC (area under ROC curve), and Brier score.</description>
<category>Evaluation</category>
<icon>icons/TestLearners.png</icon>
<priority>200</priority>
"""
#
# OWTestLearners.py
#

from qttable import *
from OData import *
from OWWidget import *
import orngTest, orngStat

##############################################################################

class OWTestLearners(OWWidget):
    settingsList = ["sampleMethod", "nFolds", "pLearning", "useStat", "pRepeat", "precision"]

    stat = ( ('Classification Accuracy', 'CA', 'CA(res)'),
             ('Sensitivity', 'Sens', 'sens(cm)'),
             ('Specificity', 'Spec', 'spec(cm)'),
             ('Area Under ROC Curve', 'AUC', 'AUCFromCDT(cdt)'),
             ('Information Score', 'IS', 'IS(res)'),
             ('Brier Score', 'Brier', 'BrierScore(res)')
           )
    
    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        "TestLearners",
        """TestLearners widget can take learners
on the input and tests them\n on a single data set. Alternatively, it can
evaluate already build classifiers. For testing learners,
it implements different sampling techniques.
It reports on variety of statistics, including accuracy,
sensitivity, specificity, information score,
AUC (area under ROC curve), and Brier score.
""",
        FALSE,
        FALSE)
        
        self.addInput("cdata")
        self.addInput("learner", FALSE)
        self.addInput("classifier", FALSE)

        # Settings
        self.sampleMethod = 0               # cross validation
        self.nFolds = 5                     # cross validation folds
        self.pLearning = 70                 # size of learning set when sampling [%]
        self.useStat = [1] * len(self.stat)
        self.pRepeat = 10
        self.precision = 4
        self.loadSettings()
        
        self.data = None                    # input data set
        self.learnDict = {}
        self.results = {}

        # GUI

        # sampling
        self.s = [None, None, None]
        self.sBox = QVButtonGroup("Sampling", self.controlArea)        
        self.s[0] = QRadioButton('Cross Validation', self.sBox)

        self.h1Box = QHBox(self.sBox)
        QWidget(self.h1Box).setFixedSize(19, 8)
        QLabel("Number of Folds:  ", self.h1Box)
        self.nFoldsSB = QSpinBox(2, 20, 1, self.h1Box)
        self.nFoldsSB.setValue(self.nFolds)

        self.s[1] = QRadioButton('Leave-One-Out', self.sBox)
        self.s[2] = QRadioButton('Random Sampling', self.sBox)

        self.h3Box = QHBox(self.sBox)
        QWidget(self.h3Box).setFixedSize(19, 8)
        QLabel("Repeat train/test:  ", self.h3Box)
        self.pRepeatBox = QSpinBox(1, 100, 1, self.h3Box)
        self.pRepeatBox.setValue(self.pRepeat)

        self.h2Box = QHBox(self.sBox)
        QWidget(self.h2Box).setFixedSize(19, 8)
        QLabel("Relative Training Set Size:", self.h2Box)
        self.h3Box = QHBox(self.sBox)
        QWidget(self.h3Box).setFixedSize(19, 8)
        self.pLearnSlider = QSlider(10, 100, 1, self.pLearning, QSlider.Horizontal, self.h3Box)
        self.pLearnSlider.setTickmarks(QSlider.Below)
        self.pLearnSlider.setTickInterval(10)
        self.pLabel = QLabel("   " + str(self.pLearning) + "%", self.h3Box)

        QWidget(self.sBox).setFixedSize(0, 8)
        self.applyBtn = QPushButton("&Apply", self.sBox)
        self.applyBtn.setDisabled(TRUE)

        self.s[self.sampleMethod].setChecked(TRUE)        
        QWidget(self.controlArea).setFixedSize(0, 16)
        
        # statistics
        self.statBox = QVGroupBox(self.controlArea)
        self.statBox.setTitle('Statistics')
        self.statBtn = []
        for i in range(len(self.stat)):
            self.statBtn.append(QCheckBox(self.stat[i][0], self.statBox))
            self.statBtn[i].setChecked(self.useStat[i])

        # table with results
        self.layout=QVBoxLayout(self.mainArea)
        self.g = QVGroupBox(self.mainArea)
        self.g.setTitle('Evaluation Results')

        self.tab=QTable(self.g)
        self.tab.setSelectionMode(QTable.NoSelection)
        self.layout.add(self.g)

        self.lab = QLabel(self.g)
            
        # signals
        self.connect(self.applyBtn, SIGNAL("clicked()"), self.test)
        self.connect(self.pLearnSlider, SIGNAL("valueChanged(int)"), self.setPLearn)
        self.connect(self.pRepeatBox, SIGNAL("valueChanged(int)"), self.pRepeatChanged)
        self.connect(self.nFoldsSB, SIGNAL("valueChanged(int)"), self.nFoldsChanged)
        self.dummy1 = [None]*3
        for i in range(3):
            self.dummy1[i] = lambda x, v=i: self.sChanged(x, v)
            self.connect(self.s[i], SIGNAL("toggled(bool)"), self.dummy1[i])
        self.dummy2 = [None]*len(self.stat)
        for i in range(len(self.stat)):
            self.dummy2[i] = lambda x, v=i: self.statChanged(x, v)
            self.connect(self.statBtn[i], SIGNAL('toggled(bool)'), self.dummy2[i])

        self.resize(500,400)

    # main part:         

    def test(self):
        try:
            self.learners = []
            for k in self.learnDict.keys():
                self.learners.append(self.learnDict[k])

            if self.sampleMethod==0:
                res = orngTest.crossValidation(self.learners, self.data, folds=self.nFolds, strat=1)
            elif self.sampleMethod==1:
                res = orngTest.leaveOneOut(self.learners, self.data)
            elif self.sampleMethod==2:
                res = orngTest.proportionTest(self.learners, self.data, self.pLearning/100., times=self.pRepeat)

            cm = orngStat.computeConfusionMatrices(res, classIndex = self.classindex)
            cdt = orngStat.computeCDT(res, classIndex = self.classindex)
            self.results = {}
            for i in range(len(self.stat)):
                self.results[i] = eval('orngStat.' + self.stat[i][2])

            # orngStat.AROCFromCDT(cdt[i])[7])

            self.setStatTable()
            
        except Exception, msg:
            QMessageBox.critical(self, self.title + ": Execution error", "Error while testing: '%s'" % msg)

    def setStatTable(self):
        self.tab.setNumCols(len(self.stat)+1)
        self.tabHH=self.tab.horizontalHeader()
        self.tabHH.setLabel(0, 'Classifier')
        for i in range(len(self.stat)):
            self.tabHH.setLabel(i+1, self.stat[i][1])

        self.tab.setNumRows(len(self.learners))
        for i in range(len(self.learners)):
            self.tab.setText(i, 0, self.learners[i].name)

        prec="%%.%df" % self.precision

        for i in range(len(self.learners)):
            for j in range(len(self.stat)):
                self.tab.setText(i, j+1, prec % self.results[j][i])

        for i in range(len(self.stat)+1):
            self.tab.adjustColumn(i)

        for i in range(len(self.stat)):
            if not self.useStat[i]:
                self.tab.hideColumn(i+1)
    # slots: handle input signals        
        
    def cdata(self,data):
        self.data = data.table
        self.classindex = data.targetValIndx
        if len(self.learnDict)>0:
            self.applyBtn.setDisabled(FALSE)

    def learner(self, learner, id):
        self.learnDict[id] = learner
        if self.data:
            self.applyBtn.setDisabled(FALSE)

    def classifier(self, classifier, id):
        pass

    # signal processing

    def statChanged(self, value, id):
        self.useStat[id] = value
        if value:
            self.tab.showColumn(id+1)
            self.tab.adjustColumn(id+1)
        else:
            self.tab.hideColumn(id+1)

    def setPLearn(self, value):
        self.pLearning =value
        self.pLabel.setText("   " + str(self.pLearning) + "%")

    def sChanged(self, value, id):
        self.sampleMethod = id

    def nFoldsChanged(self, value):
        self.nFolds = value

    def pRepeatChanged(self, value):
        self.pRepeat = value
        
##############################################################################
# Test the widget, run from DOS prompt

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWTestLearners()
    a.setMainWidget(ow)

    dataset = orange.ExampleTable('test')
    od = OrangeData(dataset)
    ow.cdata(od)

    l1 = orange.BayesLearner()
    l1.name = 'Naive Bayes'
    ow.learner(l1,1)

    l2 = orange.BayesLearner()
    l2.name = 'Naive Bayes(m=10)'
    l2.conditionalEstimatorConstructor = orange.ConstructProbabilityEstimator_m(m=10)
    l2.unconditionalEstimatorConstructor = l2.conditionalEstimatorConstructor
    ow.learner(l2,2)

    ow.show()
    a.exec_loop()
    ow.saveSettings()