"""
<name>Test Learners</name>
<description>Estimates the predictive performance of learners on a data set.</description>
<icon>icons/TestLearners.png</icon>
<priority>200</priority>
"""
#
# OWTestLearners.py
#

from qttable import *
from OWWidget import *
import orngTest, orngStat, OWGUI

##############################################################################

class OWTestLearners(OWWidget):
    settingsList = ["sampleMethod", "nFolds", "pLearning", "useStat", "pRepeat", "precision"]
    callbackDeposit = []

    stat = ( ('Classification Accuracy', 'CA', 'CA(res)'),
             ('Sensitivity', 'Sens', 'sens(cm)'),
             ('Specificity', 'Spec', 'spec(cm)'),
             ('Area Under ROC Curve', 'AUC', 'AUCFromCDT(cdt)'),
             ('Information Score', 'IS', 'IS(res)'),
             ('Brier Score', 'Brier', 'BrierScore(res)')
           )
    
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "TestLearners")
        
        self.inputs = [("Test Data Set", ExampleTableWithClass, self.cdata), ("Learner", orange.Learner, self.learner, 0)]
        self.outputs = [("Evaluation Results", orngTest.ExperimentResults)]

        # Settings
        self.sampleMethod = 0               # cross validation
        self.nFolds = 5                     # cross validation folds
        self.pLearning = 70                 # size of learning set when sampling [%]
        self.useStat = [1] * len(self.stat)
        self.pRepeat = 10
        self.precision = 4
        self.loadSettings()
        
        self.data = None                    # input data set
        self.learnDict = {}; self.learners = []
        self.results = None; self.scores = None

        # GUI
        self.s = [None, None, None, None]
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

        QWidget(self.sBox).setFixedSize(0, 8)
        self.applyBtn = QPushButton("&Apply", self.sBox)
        self.applyBtn.setDisabled(TRUE)

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

        self.resize(500,400)

    # test() evaluates the learners on a sigle data set
    # if learner is specified, this is either a new or an oldlearner to
    # be tested. the list in results should either be recomputed or added
    # else, if learner=None, all results are recomputed (user pressed apply button)
    def test(self, learner=None):
        pb = ProgressBar(self, iterations=self.nFolds)

        # testing
        if self.results and learner:
            learners = [learner]
        else:
            learners = self.learnDict.values()

        if self.sampleMethod==0:
            res = orngTest.crossValidation(learners, self.data, folds=self.nFolds, strat=orange.MakeRandomIndices.StratifiedIfPossible, callback=pb.advance)
        elif self.sampleMethod==1:
            res = orngTest.leaveOneOut(learners, self.data)
        elif self.sampleMethod==2:
            res = orngTest.proportionTest(learners, self.data, self.pLearning/100., times=self.pRepeat)
        elif self.sampleMethod==3:
            res = orngTest.learnAndTestOnLearnData(learners, self.data)

        cm = orngStat.computeConfusionMatrices(res, classIndex = self.classindex)
        cdt = orngStat.computeCDT(res, classIndex = self.classindex)

        # merging of results and scores (if necessary)
        if self.results and learner:
            if len(self.learners) > len(self.scores[0]):
                # this is a new learner, add new results
                self.results.classifierNames.append(learner.name)
                self.results.numberOfLearners += 1
                for i,r in enumerate(self.results.results):
                    r.classes.append(res.results[i].classes[0])
                    r.probabilities.append(res.results[i].probabilities[0])
                for (i, stat) in enumerate(self.stat):
                    self.scores[i].append(eval('orngStat.' + stat[2])[0])
            else:
                # this is an old but updated learner
                indx = self.learners.index(learner)
                self.results.classifierNames[indx] = learner.name
                for i,r in enumerate(self.results.results):
                    r.classes[indx] = res.results[i].classes[0]
                    r.probabilities[indx] = res.results[i].probabilities[0]
                for (i, stat) in enumerate(self.stat):
                    self.scores[i][indx] = eval('orngStat.' + stat[2])[0]
        else: # test on all learners
            self.results = res
            self.scores = []
            for i in range(len(self.stat)):
                self.scores.append(eval('orngStat.' + self.stat[i][2]))

        # update the tables that show the results
        self.setStatTable()
        self.send("Evaluation Results", self.results)
        pb.finish()

#        except Exception, msg:
#            QMessageBox.critical(self, self.title + ": Execution error", "Error while testing: '%s'" % msg)

    # slots: handle input signals        
    def cdata(self,data):
        if not data:
            return # have to handle this appropriately
        self.data = orange.Filter_hasClassValue(data)
        self.classindex = 0 # data.targetValIndx
        if self.learners:
            self.applyBtn.setDisabled(FALSE)
            self.results = None; self.scores = None
            self.test()

    def learner(self, learner, id=None):
        if not learner: # remove a learner and corresponding results
            indx = self.learners.index(self.learnDict[id])
            for i,r in enumerate(self.results.results):
                del r.classes[indx]
                del r.probabilities[indx]
            del self.results.classifierNames[indx]
            self.results.numberOfLearners -= 1
            for (i, stat) in enumerate(self.stat):
                del self.scores[i][indx]
            del self.learners[indx]
            del self.learnDict[id]
            self.setStatTable()
            self.send("Evaluation Results", self.results)
        else: # a new or updated learner
            if not self.learnDict.has_key(id): # new
                self.learners.append(learner)
            else: # updated
                self.learners[self.learners.index(self.learnDict[id])] = learner
            self.learnDict[id] = learner
            if self.data:
                self.test(learner)
                self.applyBtn.setDisabled(FALSE)
            
    # signal processing
    def statChanged(self, value, id):
        self.useStat[id] = value
        if value:
            self.tab.showColumn(id+1)
            self.tab.adjustColumn(id+1)
        else:
            self.tab.hideColumn(id+1)

    def sChanged(self, value, id):
        self.sampleMethod = id

    # reporting on evaluation results
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
    ow.cdata(data)

    l1 = orange.BayesLearner()
    l1.name = 'Naive Bayes'
    ow.learner(l1, 1)

    l2 = orange.BayesLearner()
    l2.name = 'Naive Bayes (m=10)'
    l2.estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m=10)
    l2.conditionalEstimatorConstructor = orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m=10))
    ow.learner(l2, 2)

    # now we resend the first learner
    l3 = orange.BayesLearner()
    l3.name = 'NB First'
    ow.learner(l3, 1)

    import orngTree
    l4 = orngTree.TreeLearner(minSubset=2)
    l4.name = "Decision Tree"
    ow.learner(l4, 4)

    # and now we kill the first learner    
    ow.learner(None, 1)    

    ow.show()
    a.exec_loop()
    ow.saveSettings()