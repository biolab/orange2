"""
<name>Test Learners</name>
<description>Estimates the predictive performance of learners on a data set.</description>
<icon>icons/TestLearners.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact> 
<priority>200</priority>
"""
#
# OWTestLearners.py
#

from qttable import *
from OWWidget import *
import orngTest, orngStat, OWGUI
import warnings
warnings.filterwarnings("ignore", "'id' is not a builtin attribute", orange.AttributeWarning)

##############################################################################

class OWTestLearners(OWWidget):
    settingsList = ["sampleMethod", "nFolds", "pLearning", "useStat", "pRepeat", "precision"]
    callbackDeposit = []

    stat = ( ('Classification Accuracy', 'CA', 'CA(res)'),
             ('Sensitivity', 'Sens', 'sens(cm)'),
             ('Specificity', 'Spec', 'spec(cm)'),
             ('Area Under ROC Curve', 'AUC', 'AUC(res)'),
             ('Information Score', 'IS', 'IS(res)'),
             ('Brier Score', 'Brier', 'BrierScore(res)')
           )
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "TestLearners")
        
        self.inputs = [("Data", ExampleTableWithClass, self.cdata, Default), ("Separate Test Data", ExampleTableWithClass, self.testdata), ("Learner", orange.Learner, self.learner, Multiple)]
        self.outputs = [("Evaluation Results", orngTest.ExperimentResults)]

        # Settings
        self.sampleMethod = 0           # cross validation
        self.nFolds = 5                 # cross validation folds
        self.pLearning = 70   # size of learning set when sampling [%]
        self.useStat = [1] * len(self.stat)
        self.pRepeat = 10
        self.precision = 4
        self.loadSettings()
        
        self.data = None                # input data set
        self.testdata = None            # separate test data set
        self.learners = None            # set of learners (input)
        self.results = None             # from orngTest
        self.scores = None              # to be displayed in the table

        # GUI
        self.s = [None] * 5
        self.sBox = QVButtonGroup("Sampling", self.controlArea)        
        self.s[0] = QRadioButton('Cross Validation', self.sBox)

        box = QHBox(self.sBox)
        QWidget(box).setFixedSize(19, 8)
        OWGUI.spin(box, self, 'nFolds', 2, 100, step=1, label='Number of Folds:  ')

        self.s[1] = QRadioButton('Leave-One-Out', self.sBox)
        self.s[2] = QRadioButton('Random Sampling', self.sBox)

        box = QHBox(self.sBox)
        QWidget(box).setFixedSize(19, 8)
        OWGUI.spin(box, self, 'pRepeat', 1, 100, step=1, label='Repeat Train/Test:  ')

        self.h2Box = QHBox(self.sBox)
        QWidget(self.h2Box).setFixedSize(19, 8)
        QLabel("Relative Training Set Size:", self.h2Box)
        box = QHBox(self.sBox)
        QWidget(box).setFixedSize(19, 8)
        OWGUI.hSlider(box, self, 'pLearning', minValue=10, maxValue=100, step=1, ticks=10, labelFormat="   %d%%")        

        self.s[3] = QRadioButton('Test on Train Data', self.sBox)        
        self.s[4] = self.testDataBtn = QRadioButton('Test on Test Data', self.sBox)

        QWidget(self.sBox).setFixedSize(0, 8)
        self.applyBtn = QPushButton("&Apply", self.sBox)
        self.applyBtn.setDisabled(TRUE)

        if self.sampleMethod == 4:
            self.sampleMethod = 0
        self.s[self.sampleMethod].setChecked(TRUE)        
        OWGUI.separator(self.controlArea)
        
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
        self.dummy1 = [None]*len(self.s)
        for i in range(len(self.s)):
            self.dummy1[i] = lambda x, v=i: self.sChanged(x, v)
            self.connect(self.s[i], SIGNAL("toggled(bool)"), self.dummy1[i])
        self.dummy2 = [None]*len(self.stat)
        for i in range(len(self.stat)):
            self.dummy2[i] = lambda x, v=i: self.statChanged(x, v)
            self.connect(self.statBtn[i], SIGNAL('toggled(bool)'), self.dummy2[i])

        self.resize(600,400)

    # test() evaluates the learners on a sigle data set
    # if learner is specified, this is either a new or an oldlearner to
    # be tested. the list in results should either be recomputed or added
    # else, if learner=None, all results are recomputed (user pressed apply button)
    def test(self, learner=None):
        if not self.data:
            return
        if learner:
            learners = [learner]
        else:
            learners = self.learners

        if self.sampleMethod==4 and not self.testdata:
            self.results = None
            self.setStatTable() # makes table with results empty
            return

        pb = None
        if self.sampleMethod==0:
            pb = ProgressBar(self, iterations=self.nFolds)
            res = orngTest.crossValidation(learners, self.data, folds=self.nFolds, strat=orange.MakeRandomIndices.StratifiedIfPossible, callback=pb.advance, storeExamples = True)
        elif self.sampleMethod==1:
            pb = ProgressBar(self, iterations=len(self.data))
            res = orngTest.leaveOneOut(learners, self.data, callback=pb.advance, storeExamples = True)
        elif self.sampleMethod==2:
            pb = ProgressBar(self, iterations=self.pRepeat)
            res = orngTest.proportionTest(learners, self.data, self.pLearning/100., times=self.pRepeat, callback=pb.advance)
        elif self.sampleMethod==3:
            res = orngTest.learnAndTestOnLearnData(learners, self.data)
        elif self.sampleMethod==4:                
            res = orngTest.learnAndTestOnTestData(learners, self.data, self.testdata)

        cm = orngStat.computeConfusionMatrices(res, classIndex = self.classindex)
        cdt = orngStat.computeCDT(res, classIndex = self.classindex)

        # merging of results and scores (if necessary)
        if self.results and learner:
            if learner.id not in [l.id for l in self.learners]:
                # this is a new learner, add new results
                self.results.classifierNames.append(learner.name)
                self.results.numberOfLearners += 1
                for i,r in enumerate(self.results.results):
                    r.classes.append(res.results[i].classes[0])
                    r.probabilities.append(res.results[i].probabilities[0])
                for (i, stat) in enumerate(self.stat):
                    try:
                        self.scores[i].append(eval('orngStat.' + stat[2])[0])
                    except:
                        self.scores[i].append(-1) # handle the exception
                        type, val, traceback = sys.exc_info()
                        sys.excepthook(type, val, traceback)  # print the exception
                        self.error("Caught an exception while evaluating classifiers")
            else:
                # this is an old but updated learner
                indx = [l.id for l in self.learners].index(learner.id)
                self.results.classifierNames[indx] = learner.name
                for i,r in enumerate(self.results.results):
                    r.classes[indx] = res.results[i].classes[0]
                    r.probabilities[indx] = res.results[i].probabilities[0]
                for (i, stat) in enumerate(self.stat):
                    try:
                        self.scores[i][indx] = eval('orngStat.' + stat[2])[0]
                    except:
                        self.scores[i][indx] = -1
                        type, val, traceback = sys.exc_info()
                        sys.excepthook(type, val, traceback)  # print the exception
                        self.error("Caught an exception while evaluating classifiers")
                    
        else: # test on all learners, or on the new learner with no other learners in the memory
            self.results = res
            self.scores = []
            for i in range(len(self.stat)):
                try:
                    self.scores.append(eval('orngStat.' + self.stat[i][2]))
                except:
                    self.scores.append([-1 for c in range(len(self.results.learners))]) # handle the exception
                    type, val, traceback = sys.exc_info()
                    sys.excepthook(type, val, traceback)  # print the exception
                    self.error("Caught an exception while evaluating classifiers")

        # update the tables that show the results
        self.setStatTable()
        self.send("Evaluation Results", self.results)
        if pb: pb.finish()

#        except Exception, msg:
#            QMessageBox.critical(self, self.title + ": Execution error", "Error while testing: '%s'" % msg)

    # slots: handle input signals
    def cdata(self, data):
        if not data:
            self.data = None
            self.results = None
            self.setStatTable()
            return

##        if self.testdata and data.domain <> self.testdata.domain:
##            self.testdata = None
##            self.results = None
##            # self.setStatTable()
        self.data = orange.Filter_hasClassValue(data)
        self.classindex = 0 # data.targetValIndx
        if self.learners:
            self.applyBtn.setDisabled(FALSE)
            self.results = None; self.scores = None
            self.test()

    def testdata(self, data):
        self.testdata = data
        if self.sampleMethod == 4:
            self.test()

    def learner(self, learner, id=None):
        if learner: # a new or updated learner
            # print 'ADD/UPD', learner.name, ", id:", id
            learner.id = id # remember id's of learners
            self.test(learner)
            if self.learners:
                if id not in [l.id for l in self.learners]:
                    self.learners.append(learner)
            else:
                self.learners = [learner]
            self.applyBtn.setDisabled(FALSE)
        else: # remove a learner and corresponding results
            # print 'REMOVE', id, 'FROM', self.learners
            ids = [l.id for l in self.learners]
            if id not in ids:
                return                  # happens if a widget with learner empties the signal first
            indx = ids.index(id)

            if self.results:
                del self.results.classifierNames[indx]
                self.results.numberOfLearners -= 1
                for i, r in enumerate(self.results.results):
                    del r.classes[indx]
                    del r.probabilities[indx]
                for (i, stat) in enumerate(self.stat):
                    del self.scores[i][indx]
                self.setStatTable()
                self.send("Evaluation Results", self.results)

            del self.learners[indx]
            
    # signal processing
    def statChanged(self, value, id):
        self.useStat[id] = value
        if value:
            self.tab.showColumn(id+1)
            self.tab.adjustColumn(id+1)
        else:
            self.tab.hideColumn(id+1)

    def sChanged(self, value, id):
        if self.sampleMethod <> id:
            self.sampleMethod = id
            if self.data:
                self.results = None
                self.test()

    # reporting on evaluation results
    def setStatTable(self):
        if not self.results:
            self.tab.setNumRows(0)
            return
        self.tab.setNumCols(len(self.stat)+1)
        self.tabHH=self.tab.horizontalHeader()
        self.tabHH.setLabel(0, 'Classifier')
        for i in range(len(self.stat)):
            self.tabHH.setLabel(i+1, self.stat[i][1])

        self.tab.setNumRows(self.results.numberOfLearners)
        for i in range(len(self.results.classifierNames)):
            self.tab.setText(i, 0, self.results.classifierNames[i])

        prec="%%.%df" % self.precision

        for i in range(self.results.numberOfLearners):
            for j in range(len(self.stat)):
                if self.scores[j][i] < 1e-8:
                    self.tab.setText(i, j+1, "N/A")
                else:
                    self.tab.setText(i, j+1, prec % self.scores[j][i])

        for i in range(len(self.stat)+1):
            self.tab.adjustColumn(i)

        for i in range(len(self.stat)):
            if not self.useStat[i]:
                self.tab.hideColumn(i+1)

#
class ProgressBar:
    def __init__(self, widget, iterations):
        self.iter = iterations
        self.widget = widget
        self.count = 0
        self.widget.progressBarInit()
    def advance(self):
        self.count += 1
        self.widget.progressBarSet(int(self.count*100/self.iter))
    def finish(self):
        self.widget.progressBarFinished()

##############################################################################
# Test the widget, run from DOS prompt

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWTestLearners()
    a.setMainWidget(ow)

    data = orange.ExampleTable('voting')

    l1 = orange.MajorityLearner(); l1.name = '1 - Majority'

    l2 = orange.BayesLearner()
    l2.estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m=10)
    l2.conditionalEstimatorConstructor = orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m=10))
    l2.name = '2 - NBC (m=10)'

    l3 = orange.BayesLearner(); l3.name = '3 - NBC (default)'

    l4 = orange.MajorityLearner(); l4.name = "4 - Majority"

    testcase = 2

    if testcase == 0: # 1(UPD), 3, 4
        ow.cdata(data)
        ow.learner(l1, 1)
        ow.learner(l2, 2)
        ow.learner(l3, 3)
        l1.name = l1.name + " UPD"
        ow.learner(l1, 1)
        ow.learner(None, 2)
        ow.learner(l4, 4)
    if testcase == 1: # no data, all learners removed
        ow.learner(l1, 1)
        ow.learner(l2, 2)
        ow.learner(None, 2)
        ow.learner(None, 1)
        ow.cdata(data)
    if testcase == 2: # sends data, then learner, then removes the learner
        ow.cdata(data)
        ow.learner(l1, 1)
        ow.learner(None, 1)
        
    ow.show()
    a.exec_loop()
    ow.saveSettings()
