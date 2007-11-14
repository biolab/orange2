"""
<name>Test Learners</name>
<description>Estimates the predictive performance of
learners on a data set.</description>
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
import time
import warnings
warnings.filterwarnings("ignore", "'id' is not a builtin attribute",
                        orange.AttributeWarning)

##############################################################################

class Learner:
    def __init__(self, learner, id):
        learner.id = id
        self.learner = learner
        self.name = learner.name
        self.id = id
        self.scores = []
        self.results = None
        self.time = time.time() # used to order the learners in the table

class Score:
    def __init__(self, name, label, f, show=True, cmBased=False):
        self.name = name
        self.label = label
        self.f = f
        self.show = show
        self.cmBased = cmBased

class OWTestLearners(OWWidget):
    settingsList = ["sampleMethod", "nFolds", "pLearning", "pRepeat", "precision",
                    "selectedCScores", "selectedRScores", "applyOnAnyChange"]
    contextHandlers = {"": DomainContextHandler("", ["targetClass"])}
    callbackDeposit = []

    cStatistics = [apply(Score,s) for s in [\
        ('Classification accuracy', 'CA', 'CA(res)', True),
        ('Sensitivity', 'Sens', 'sens(cm)', True, True),
        ('Specificity', 'Spec', 'spec(cm)', True, True),
        ('Area under ROC curve', 'AUC', 'AUC(res)', True),
        ('Information score', 'IS', 'IS(res)', False),
        ('F-measure', 'F1', 'F1(cm)', False, True),
        ('Brier score', 'Brier', 'BrierScore(res)', True)]]

    rStatistics = [apply(Score,s) for s in [\
        ("Mean squared error", "MSE", "MSE(res)", False),
        ("Root mean squared error", "RMSE", "RMSE(res)"),
        ("Mean absolute error", "MAE", "MAE(res)", False),
        ("Relative squared error", "RSE", "RSE(res)", False),
        ("Root relative squared error", "RRSE", "RRSE(res)"),
        ("Relative absolute error", "RAE", "RAE(res)", False),
        ("R-squared", "R2", "R2(res)")]]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "TestLearners")

        self.inputs = [("Data", ExampleTable, self.setData, Default), ("Separate Test Data", ExampleTable, self.setTestData), ("Learner", orange.Learner, self.setLearner, Multiple)]
        self.outputs = [("Evaluation Results", orngTest.ExperimentResults)]

        # Settings
        self.sampleMethod = 0           # cross validation
        self.nFolds = 5                 # cross validation folds
        self.pLearning = 70   # size of learning set when sampling [%]
        self.pRepeat = 10
        self.precision = 4
        self.applyOnAnyChange = True
        self.selectedCScores = [i for (i,s) in enumerate(self.cStatistics) if s.show]
        self.selectedRScores = [i for (i,s) in enumerate(self.rStatistics) if s.show]
        self.targetClass=0
        self.loadSettings()

        self.stat = self.cStatistics

        self.data = None                # input data set
        self.testdata = None            # separate test data set
        self.learners = {}              # set of learners (input)
        self.results = None             # from orngTest

        # GUI
        self.s = [None] * 5
        self.sBox = QVButtonGroup("Sampling", self.controlArea)
        self.s[0] = QRadioButton('Cross validation', self.sBox)

        box = QHBox(self.sBox)
        QWidget(box).setFixedSize(19, 8)
        OWGUI.spin(box, self, 'nFolds', 2, 100, step=1,
                   label='Number of folds:  ')

        self.s[1] = QRadioButton('Leave one out', self.sBox)
        self.s[2] = QRadioButton('Random sampling', self.sBox)

        box = QHBox(self.sBox)
        QWidget(box).setFixedSize(19, 8)
        OWGUI.spin(box, self, 'pRepeat', 1, 100, step=1,
                   label='Repeat train/test:  ')

        self.h2Box = QHBox(self.sBox)
        QWidget(self.h2Box).setFixedSize(19, 8)
        QLabel("Relative training set size:", self.h2Box)
        box = QHBox(self.sBox)
        QWidget(box).setFixedSize(19, 8)
        OWGUI.hSlider(box, self, 'pLearning', minValue=10, maxValue=100,
                      step=1, ticks=10, labelFormat="   %d%%")

        self.s[3] = QRadioButton('Test on train data', self.sBox)
        self.s[4] = self.testDataBtn = QRadioButton('Test on test data', self.sBox)
        self.testDataBtn.setDisabled(True)

        OWGUI.separator(self.sBox)
        OWGUI.checkBox(self.sBox, self, 'applyOnAnyChange',
                       label="Apply on any change", callback=self.applyChange)
        self.applyBtn = OWGUI.button(self.sBox, self, "&Apply",
                                     callback=lambda f=True: self.recompute(f))
        self.applyBtn.setDisabled(True)

        if self.sampleMethod == 4:
            self.sampleMethod = 0
        self.s[self.sampleMethod].setChecked(True)
        OWGUI.separator(self.controlArea)

        # statistics
        self.cbox = QVBox(self.controlArea)
        self.cStatLabels = [s.name for s in self.cStatistics]
        self.cstatLB = OWGUI.listBox(self.cbox, self, 'selectedCScores',
                                     'cStatLabels', box = "Performance scores",
                                     selectionMode = QListBox.Multi,
                                     callback=self.newscoreselection)
        OWGUI.separator(self.cbox)
        self.targetCombo=OWGUI.comboBox(self.cbox, self, "targetClass", orientation=0,
                                        callback=[self.changedTarget],
                                        box="Target class")

        self.rStatLabels = [s.name for s in self.rStatistics]
        self.rstatLB = OWGUI.listBox(self.controlArea, self, 'selectedRScores',
                                     'rStatLabels', box = "Performance scores",
                                     selectionMode = QListBox.Multi,
                                     callback=self.newscoreselection)
        self.rstatLB.box.hide()


        # score table
        self.layout=QVBoxLayout(self.mainArea)
        self.g = QVGroupBox(self.mainArea)
        self.g.setTitle('Evaluation results')

        self.tab=QTable(self.g)
        self.tab.setSelectionMode(QTable.NoSelection)
        self.layout.add(self.g)

        self.lab = QLabel(self.g)

        # signals - change of sampling technique
        self.dummy1 = [None]*len(self.s)
        for i in range(len(self.s)):
            self.dummy1[i] = lambda x, v=i: self.newsampling(x, v)
            self.connect(self.s[i], SIGNAL("toggled(bool)"), self.dummy1[i])

        self.resize(600,470)

    # scoring and painting of score table
    def isclassification(self):
        if not self.data:
            return True
        return self.data.domain.classVar.varType == orange.VarTypes.Discrete
        
    def paintscores(self):
        """paints the table with evaluation scores"""
        def adjustcolumns():
            """adjust the width of the score table cloumns"""
            usestat = [self.selectedRScores, self.selectedCScores][self.isclassification()]
            for i in range(len(self.stat)+1):
                self.tab.adjustColumn(i)
            for i in range(len(self.stat)):
                if i not in usestat:
                    self.tab.hideColumn(i+1)

        self.tab.setNumCols(len(self.stat)+1)
        self.tabHH=self.tab.horizontalHeader()
        self.tabHH.setLabel(0, 'Method')
        for (i,s) in enumerate(self.stat):
            self.tabHH.setLabel(i+1, s.label)

        prec="%%.%df" % self.precision

        learners = [(l.time, l) for l in self.learners.values()]
        learners.sort()
        learners = [lt[1] for lt in learners]

        self.tab.setNumRows(len(self.learners))
        for (i, l) in enumerate(learners):
            self.tab.setText(i, 0, l.name)
            
        for (i, l) in enumerate(learners):
            if l.scores:
                for j in range(len(self.stat)):
                    self.tab.setText(i, j+1, prec % l.scores[j])
            else:
                for j in range(len(self.stat)):
                    self.tab.setText(i, j+1, "")
        adjustcolumns()

    def score(self, ids):
        """compute scores for the list of learners"""
        if (not self.data):
            for id in ids:
                self.learners[id].results = None
            return
        # test which learners can accept the given data set
        # e.g., regressions can't deal with classification data
        learners = []
        n = len(self.data.domain.attributes)*2
        new = self.data.selectref([1]*min(n, len(self.data)) +
                                  [0]*(len(self.data) - min(n, len(self.data))))
        for l in [self.learners[id] for id in ids]:
            try:
                predictor = l.learner(new)
                if predictor(new[0]).varType == new.domain.classVar.varType:
                    learners.append(l.learner)
                else:
                    l.scores = []
            except:
                l.scores = []

        if not learners:
            return

        # computation of results (res, and cm if classification)
        pb = None
        if self.sampleMethod==0:
            pb = OWGUI.ProgressBar(self, iterations=self.nFolds)
            res = orngTest.crossValidation(learners, self.data, folds=self.nFolds,
                                           strat=orange.MakeRandomIndices.StratifiedIfPossible,
                                           callback=pb.advance, storeExamples = True)
            pb.finish()
        elif self.sampleMethod==1:
            pb = OWGUI.ProgressBar(self, iterations=len(self.data))
            res = orngTest.leaveOneOut(learners, self.data,
                                       callback=pb.advance, storeExamples = True)
            pb.finish()
        elif self.sampleMethod==2:
            pb = OWGUI.ProgressBar(self, iterations=self.pRepeat)
            res = orngTest.proportionTest(learners, self.data, self.pLearning/100.,
                                          times=self.pRepeat, callback=pb.advance, storeExamples = True)
            pb.finish()
        elif self.sampleMethod==3:
            res = orngTest.learnAndTestOnLearnData(learners, self.data, storeExamples = True)
        elif self.sampleMethod==4:
            if not self.testdata:
                for l in self.learners.values():
                    l.scores = []
                return
            res = orngTest.learnAndTestOnTestData(learners, self.data, self.testdata, storeExamples = True)
        if self.isclassification():
            cm = orngStat.computeConfusionMatrices(res, classIndex = self.targetClass)

        res.learners = learners
        for l in [self.learners[id] for id in ids]:
            if l.learner in learners:
                l.results = res

        self.error()
        try:
            scores = [eval("orngStat." + s.f) for s in self.stat]
            for (i, l) in enumerate(learners):
                self.learners[l.id].scores = [s[i] for s in scores]
        except:
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # output the exception
            self.error("An error occurred while evaluating %s" % \
                       " ".join([l.name for l in learners]))

        self.sendResults()

        #print "SCORES:"
        #for l in [self.learners[id] for id in ids]:
        #    print "%-20s" % l.name + " ".join(["%6.3f" % s for s in l.scores])

    def recomputeCM(self):
        cm = orngStat.computeConfusionMatrices(self.results, classIndex = self.targetClass)
        scores = [(indx, eval("orngStat." + s.f))
                  for (indx, s) in enumerate(self.stat) if s.cmBased]
        for (indx, score) in scores:
            for (i, l) in enumerate([l for l in self.learners.values() if l.scores]):
                l.scores[indx] = score[i]
        self.paintscores()
        

    # handle input signals

    def setData(self, data):
        """handle input train data set"""
        self.closeContext()
        self.data = data
        self.fillClassCombo()
        if not self.data: # remove data
            for l in self.learners.values():
                l.scores = []
            self.send("Evaluation Results", None)
        else:
            self.data = orange.Filter_hasClassValue(self.data)
            if self.isclassification():
                self.rstatLB.box.hide()
                self.cbox.show()
            else:
                self.cbox.hide()
                self.rstatLB.box.show()
            self.stat = [self.rStatistics, self.cStatistics][self.isclassification()]
            
            if self.learners:
                self.score([l.id for l in self.learners.values()])

        self.openContext("", data)
        self.paintscores()

    def setTestData(self, data):
        """handle test data set"""
        self.testdata = data
        if self.testdata:
            if self.sampleMethod == 4:
                if testdata:
                    self.score()
                else:
                    for l in self.learners.values():
                        l.scores = []
                self.paintscores()
        else:
            self.testDataBtn.setDisabled(True)
            if self.sampleMethod == 4:
                self.sampleMethod = 1
                self.s[0].setChecked(True)
                self.s[4].setChecked(False)
                self.recompute()

    def fillClassCombo(self):
        self.targetCombo.clear()
        if not self.data or not self.data.domain.classVar or not self.isclassification():
            return

        domain = self.data.domain
        for v in domain.classVar.values:
            self.targetCombo.insertItem(str(v))
        if self.targetClass<len(domain.classVar.values):
            self.targetCombo.setCurrentItem(self.targetClass)
        else:
            self.targetCombo.setCurrentItem(0)
            self.targetClass=0

    def setLearner(self, learner, id=None):
        """add/remove a learner"""
        if learner: # a new or updated learner
            if id in self.learners: # updated learner
                time = self.learners[id].time
                self.learners[id] = Learner(learner, id)
                self.learners[id].time = time
            else: # new learner
                self.learners[id] = Learner(learner, id)
            self.score([id])
        else: # remove a learner and corresponding results
            if id in self.learners:
                res = self.learners[id].results
                if res and res.numberOfLearners > 1:
                    indx = [l.id for l in res.learners].index(id)
                    res.remove(indx)
                    del res.learners[indx]
                del self.learners[id]
            self.sendResults()
        self.paintscores()

    # handle output signals

    def sendResults(self):
        """commit evaluation results"""
        # for each learner, we first find a list where a result is stored
        # and remember the corresponding index
        valid = [(l.results, [x.id for x in l.results.learners].index(l.id))
                 for l in self.learners.values() if l.scores]
        if not (self.data and len(valid)):
            self.send("Evaluation Results", None)
            return

        # find the result set for a largest number of learners
        # and remove this set from the list of result sets
        rlist = dict([(l.results,1) for l in self.learners.values() if l.scores]).keys()
        rlen = [r.numberOfLearners for r in rlist]
        results = rlist.pop(rlen.index(max(rlen)))
        
        for (i, l) in enumerate(results.learners):
            print "xxx %s" % str(l.id)
            if not l.id in self.learners:
                results.remove(i)
                del results.learners[i]
        for r in rlist:
            for (i, l) in enumerate(r.learners):
                if (r, i) in valid:
                    results.add(r, i)
                    results.learners.append(r.learners[i])
                    self.learners[r.learners[i].id].results = results
        self.send("Evaluation Results", results)
        self.results = results

    # signal processing

    def newsampling(self, value, id):
        """handle change of evaluation method"""
        if not self.applyOnAnyChange:
            self.applyBtn.setDisabled(self.applyOnAnyChange)
        else:
            if self.sampleMethod <> id:
                self.sampleMethod = id
                if self.learners:
                    self.recompute()

    def newscoreselection(self):
        """handle change in set of scores to be displayed"""
        usestat = [self.selectedRScores, self.selectedCScores][self.isclassification()]
        for i in range(len(self.stat)):
            if i in usestat:
                self.tab.showColumn(i+1)
                self.tab.adjustColumn(i+1)
            else:
                self.tab.hideColumn(i+1)

    def recompute(self, forced=False):
        """recompute the scores for all learners"""
        if self.applyOnAnyChange or forced:
            self.score([l.id for l in self.learners.values()])
            self.paintscores()
            self.applyBtn.setDisabled(True)
        if not self.applyOnAnyChange:
            self.applyBtn.setDisabled(False)

    def applyChange(self):
        if self.applyOnAnyChange:
            self.applyBtn.setDisabled(True)
        
    def changedTarget(self):
        self.recomputeCM()

##############################################################################
# Test the widget, run from DOS prompt

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWTestLearners()
    a.setMainWidget(ow)

    data1 = orange.ExampleTable('voting')
    data2 = orange.ExampleTable('golf')
    datar = orange.ExampleTable("auto-mpg")
#    datar = orange.ExampleTable("housing")

    l1 = orange.MajorityLearner(); l1.name = '1 - Majority'

    l2 = orange.BayesLearner()
    l2.estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m=10)
    l2.conditionalEstimatorConstructor = \
        orange.ConditionalProbabilityEstimatorConstructor_ByRows(
        estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m=10))
    l2.name = '2 - NBC (m=10)'

    l3 = orange.BayesLearner(); l3.name = '3 - NBC (default)'

    l4 = orange.MajorityLearner(); l4.name = "4 - Majority"

    import orngRegression as r
    r5 = r.LinearRegressionLearner(name="0 - lin reg")

    testcase = 1

    if testcase == 0: # 1(UPD), 3, 4
        ow.setData(data2)
        ow.setLearner(r5, 5)
        ow.setLearner(l1, 1)
        ow.setLearner(l2, 2)
        ow.setLearner(l3, 3)
        l1.name = l1.name + " UPD"
        ow.setLearner(l1, 1)
        ow.setLearner(None, 2)
        ow.setLearner(l4, 4)
#        ow.setData(data1)
#        ow.setData(datar)
#        ow.setData(data1)
    if testcase == 1: # data, but all learners removed
        ow.setLearner(l1, 1)
        ow.setLearner(l2, 2)
        ow.setLearner(l1, 1)
        ow.setLearner(None, 2)
        ow.setData(data2)
        ow.setLearner(None, 1)
    if testcase == 2: # sends data, then learner, then removes the learner
        ow.setData(data2)
        ow.setLearner(l1, 1)
        ow.setLearner(None, 1)
    if testcase == 3: # regression firs
        ow.setData(datar)
        ow.setLearner(r5, 5)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
