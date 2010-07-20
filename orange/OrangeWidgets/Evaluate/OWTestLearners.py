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
    settingsList = ["nFolds", "pLearning", "pRepeat", "precision",
                    "selectedCScores", "selectedRScores", "applyOnAnyChange",
                    "resampling"]
    contextHandlers = {"": DomainContextHandler("", ["targetClass"])}
    callbackDeposit = []

    cStatistics = [Score(*s) for s in [\
        ('Classification accuracy', 'CA', 'CA(res)', True),
        ('Sensitivity', 'Sens', 'sens(cm)', True, True),
        ('Specificity', 'Spec', 'spec(cm)', True, True),
        ('Area under ROC curve', 'AUC', 'AUC(res)', True),
        ('Information score', 'IS', 'IS(res)', False),
        ('F-measure', 'F1', 'F1(cm)', False, True),
        ('Precision', 'Prec', 'precision(cm)', False, True),
        ('Recall', 'Recall', 'recall(cm)', False, True),
        ('Brier score', 'Brier', 'BrierScore(res)', True),
        ('Matthews correlation coefficient', 'MCC', 'MCC(cm)', False, True)]]

    rStatistics = [Score(*s) for s in [\
        ("Mean squared error", "MSE", "MSE(res)", False),
        ("Root mean squared error", "RMSE", "RMSE(res)"),
        ("Mean absolute error", "MAE", "MAE(res)", False),
        ("Relative squared error", "RSE", "RSE(res)", False),
        ("Root relative squared error", "RRSE", "RRSE(res)"),
        ("Relative absolute error", "RAE", "RAE(res)", False),
        ("R-squared", "R2", "R2(res)")]]

    resamplingMethods = ["Cross-validation", "Leave-one-out", "Random sampling",
                         "Test on train data", "Test on test data"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "TestLearners")

        self.inputs = [("Data", ExampleTable, self.setData, Default), ("Separate Test Data", ExampleTable, self.setTestData), ("Learner", orange.Learner, self.setLearner, Multiple)]
        self.outputs = [("Evaluation Results", orngTest.ExperimentResults)]

        # Settings
        self.resampling = 0             # cross-validation
        self.nFolds = 5                 # cross validation folds
        self.pLearning = 70   # size of learning set when sampling [%]
        self.pRepeat = 10
        self.precision = 4
        self.applyOnAnyChange = True
        self.selectedCScores = [i for (i,s) in enumerate(self.cStatistics) if s.show]
        self.selectedRScores = [i for (i,s) in enumerate(self.rStatistics) if s.show]
        self.targetClass = 0
        self.loadSettings()
        self.resampling = 0             # cross-validation

        self.stat = self.cStatistics

        self.data = None                # input data set
        self.testdata = None            # separate test data set
        self.learners = {}              # set of learners (input)
        self.results = None             # from orngTest

        self.controlArea.layout().setSpacing(8)
        # GUI
        self.sBtns = OWGUI.radioButtonsInBox(self.controlArea, self, "resampling", box="Sampling",
                                             btnLabels=self.resamplingMethods[:1],
                                             callback=self.newsampling)
        ibox = OWGUI.widgetBox(OWGUI.indentedBox(self.sBtns))
        OWGUI.spin(ibox, self, 'nFolds', 2, 100, step=1, label='Number of folds:',
                   callback=lambda p=0: self.conditionalRecompute(p))
        OWGUI.separator(self.sBtns, height = 3)
        
        OWGUI.appendRadioButton(self.sBtns, self, "resampling", self.resamplingMethods[1])      # leave one out
        OWGUI.separator(self.sBtns, height = 3)
        OWGUI.appendRadioButton(self.sBtns, self, "resampling", self.resamplingMethods[2])      # random sampling
                        
        ibox = OWGUI.widgetBox(OWGUI.indentedBox(self.sBtns))
        OWGUI.spin(ibox, self, 'pRepeat', 1, 100, step=1,
                   label='Repeat train/test:',
                   callback=lambda p=2: self.conditionalRecompute(p))
        
        OWGUI.widgetLabel(ibox, "Relative training set size:")
        
        OWGUI.hSlider(ibox, self, 'pLearning', minValue=10, maxValue=100,
                      step=1, ticks=10, labelFormat="   %d%%",
                      callback=lambda p=2: self.conditionalRecompute(p))
        
        OWGUI.separator(self.sBtns, height = 3)
        OWGUI.appendRadioButton(self.sBtns, self, "resampling", self.resamplingMethods[3])  # test on train
        OWGUI.separator(self.sBtns, height = 3)
        OWGUI.appendRadioButton(self.sBtns, self, "resampling", self.resamplingMethods[4])  # test on test

        self.trainDataBtn = self.sBtns.buttons[-2]
        self.testDataBtn = self.sBtns.buttons[-1]
        self.testDataBtn.setDisabled(True)

        box = OWGUI.widgetBox(self.sBtns, orientation='vertical', addSpace=False)
        OWGUI.separator(box)
        OWGUI.checkBox(box, self, 'applyOnAnyChange',
                       label="Apply on any change", callback=self.applyChange)
        self.applyBtn = OWGUI.button(box, self, "&Apply",
                                     callback=lambda f=True: self.recompute(f))
        self.applyBtn.setDisabled(True)

        if self.resampling == 4:
            self.resampling = 3

#        OWGUI.separator(self.controlArea)

        # statistics
        self.statLayout = QStackedLayout()
#        self.cbox = OWGUI.widgetBox(self.controlArea, spacing=8, margin=0)
#        self.cbox.layout().setSpacing(8)
        self.cbox = OWGUI.widgetBox(self.controlArea, addToLayout=False)
        self.cStatLabels = [s.name for s in self.cStatistics]
        self.cstatLB = OWGUI.listBox(self.cbox, self, 'selectedCScores',
                                     'cStatLabels', box = "Performance scores",
                                     selectionMode = QListWidget.MultiSelection,
                                     callback=self.newscoreselection)
#        OWGUI.separator(self.cbox)
        self.cbox.layout().addSpacing(8)
        self.targetCombo = OWGUI.comboBox(self.cbox, self, "targetClass", orientation=0,
                                        callback=[self.changedTarget],
                                        box="Target class")

        self.rStatLabels = [s.name for s in self.rStatistics]
        self.rbox = OWGUI.widgetBox(self.controlArea, "Performance scores", addToLayout=False)
        self.rstatLB = OWGUI.listBox(self.rbox, self, 'selectedRScores', 'rStatLabels',
                                     selectionMode = QListWidget.MultiSelection,
                                     callback=self.newscoreselection)
        
        self.statLayout.addWidget(self.cbox)
        self.statLayout.addWidget(self.rbox)
        self.controlArea.layout().addLayout(self.statLayout)
        
        self.statLayout.setCurrentWidget(self.cbox)
        
#        self.rstatLB.box.hide()


        # score table
        # table with results
        self.g = OWGUI.widgetBox(self.mainArea, 'Evaluation Results')
        self.tab = OWGUI.table(self.g, selectionMode = QTableWidget.NoSelection)

        #self.lab = QLabel(self.g)

        self.resize(680,470)

    # scoring and painting of score table
    def isclassification(self):
        if not self.data or not self.data.domain.classVar:
            return True
        return self.data.domain.classVar.varType == orange.VarTypes.Discrete
        
    def paintscores(self):
        """paints the table with evaluation scores"""

        self.tab.setColumnCount(len(self.stat)+1)
        self.tab.setHorizontalHeaderLabels(["Method"] + [s.label for s in self.stat])
        
        prec="%%.%df" % self.precision

        learners = [(l.time, l) for l in self.learners.values()]
        learners.sort()
        learners = [lt[1] for lt in learners]

        self.tab.setRowCount(len(self.learners))
        for (i, l) in enumerate(learners):
            OWGUI.tableItem(self.tab, i,0, l.name)
            
        for (i, l) in enumerate(learners):
            if l.scores:
                for j in range(len(self.stat)):
                    if l.scores[j] is not None:
                        OWGUI.tableItem(self.tab, i, j+1, prec % l.scores[j])
                    else:
                        OWGUI.tableItem(self.tab, i, j+1, "N/A")
            else:
                for j in range(len(self.stat)):
                    OWGUI.tableItem(self.tab, i, j+1, "")
        
        # adjust the width of the score table cloumns
        self.tab.resizeColumnsToContents()
        self.tab.resizeRowsToContents()
        usestat = [self.selectedRScores, self.selectedCScores][self.isclassification()]
        for i in range(len(self.stat)):
            if i not in usestat:
                self.tab.hideColumn(i+1)


    def sendReport(self):
        exset = []
        if self.resampling == 0:
            exset = [("Folds", self.nFolds)]
        elif self.resampling == 2:
            exset = [("Repetitions", self.pRepeat), ("Proportion of training instances", "%i%%" % self.pLearning)]
        else:
            exset = []
        self.reportSettings("Validation method",
                            [("Method", self.resamplingMethods[self.resampling])]
                            + exset +
                            [("Target class", self.data.domain.classVar.values[self.targetClass])])
        
        self.reportData(self.data)
        
        self.reportSection("Results")
        learners = [(l.time, l) for l in self.learners.values()]
        learners.sort()
        learners = [lt[1] for lt in learners]
        usestat = [self.selectedRScores, self.selectedCScores][self.isclassification()]
        
        res = "<table><tr><th></th>"+"".join("<th><b>%s</b></th>" % hr for hr in [s.label for i, s in enumerate(self.stat) if i in usestat])+"</tr>"
        for i, l in enumerate(learners):
            res += "<tr><th><b>%s</b></th>" % l.name
            if l.scores:
                for j in usestat:
                    scr = l.scores[j]
                    res += "<td>" + ("%.4f" % scr if scr is not None else "") + "</td>"
            res += "</tr>"
        res += "</table>"
        self.reportRaw(res)
            
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
        indices = orange.MakeRandomIndices2(p0=min(n, len(self.data)), stratified=orange.MakeRandomIndices2.StratifiedIfPossible)
        new = self.data.selectref(indices(self.data))
#        new = self.data.selectref([1]*min(n, len(self.data)) +
#                                  [0]*(len(self.data) - min(n, len(self.data))))
        self.warning(0)
        for l in [self.learners[id] for id in ids]:
            try:
                predictor = l.learner(new)
                if predictor(new[0]).varType == new.domain.classVar.varType:
                    learners.append(l.learner)
                else:
                    l.scores = []
            except Exception, ex:
                self.warning(0, "Learner %s ends with exception: %s" % (l.name, str(ex)))
                l.scores = []

        if not learners:
            return

        # computation of results (res, and cm if classification)
        pb = None
        if self.resampling==0:
            pb = OWGUI.ProgressBar(self, iterations=self.nFolds)
            res = orngTest.crossValidation(learners, self.data, folds=self.nFolds,
                                           strat=orange.MakeRandomIndices.StratifiedIfPossible,
                                           callback=pb.advance, storeExamples = True)
            pb.finish()
        elif self.resampling==1:
            pb = OWGUI.ProgressBar(self, iterations=len(self.data))
            res = orngTest.leaveOneOut(learners, self.data,
                                       callback=pb.advance, storeExamples = True)
            pb.finish()
        elif self.resampling==2:
            pb = OWGUI.ProgressBar(self, iterations=self.pRepeat)
            res = orngTest.proportionTest(learners, self.data, self.pLearning/100.,
                                          times=self.pRepeat, callback=pb.advance, storeExamples = True)
            pb.finish()
        elif self.resampling==3:
            pb = OWGUI.ProgressBar(self, iterations=len(learners))
            res = orngTest.learnAndTestOnLearnData(learners, self.data, storeExamples = True, callback=pb.advance)
            pb.finish()
            
        elif self.resampling==4:
            if not self.testdata:
                for l in self.learners.values():
                    l.scores = []
                return
            pb = OWGUI.ProgressBar(self, iterations=len(learners))
            res = orngTest.learnAndTestOnTestData(learners, self.data, self.testdata, storeExamples = True, callback=pb.advance)
            pb.finish()
        if self.isclassification():
            cm = orngStat.computeConfusionMatrices(res, classIndex = self.targetClass)

        res.learners = learners
        for l in [self.learners[id] for id in ids]:
            if l.learner in learners:
                l.results = res

        self.error(range(len(self.stat)))
        scores = []
        for i, s in enumerate(self.stat):
            try:
                scores.append(eval("orngStat." + s.f))
                
            except Exception, ex:
                self.error(i, "An error occurred while evaluating orngStat." + s.f + "on %s due to %s" % \
                           (" ".join([l.name for l in learners]), ex.message))
                scores.append([None] * len(self.learners))
                
        for (i, l) in enumerate(learners):
            self.learners[l.id].scores = [s[i] if s else None for s in scores]
            
        self.sendResults()

    def recomputeCM(self):
        if not self.results:
            return
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
        self.data = self.isDataWithClass(data, checkMissing=True) and data or None
        self.fillClassCombo()
        if not self.data:
            # data was removed, remove the scores
            for l in self.learners.values():
                l.scores = []
                l.results = None
            self.send("Evaluation Results", None)
        else:
            # new data has arrived
            self.data = orange.Filter_hasClassValue(self.data)
            self.statLayout.setCurrentWidget(self.cbox if self.isclassification() else self.rbox)
#            if self.isclassification():
#                self.rstatLB.box.hide()
#                self.cbox.show()
#            else:
#                self.cbox.hide()
#                self.rstatLB.box.show()
            self.stat = [self.rStatistics, self.cStatistics][self.isclassification()]
            
            if self.learners:
                self.score([l.id for l in self.learners.values()])

        self.openContext("", data)
        self.paintscores()

    def setTestData(self, data):
        """handle test data set"""
        if data is None:
            self.testdata = None
        else:
            self.testdata = orange.Filter_hasClassValue(data)
        self.testDataBtn.setEnabled(self.testdata is not None)
        if self.testdata is not None:
            if self.resampling == 4:
                if self.data:
                    self.score([l.id for l in self.learners.values()])
                else:
                    for l in self.learners.values():
                        l.scores = []
                self.paintscores()
        elif self.resampling == 4 and self.data:
            # test data removed, switch to testing on train data
            self.resampling = 3
            self.recompute()

    def fillClassCombo(self):
        """upon arrival of new data appropriately set the target class combo"""
        self.targetCombo.clear()
        if not self.data or not self.data.domain.classVar or not self.isclassification():
            return

        domain = self.data.domain
        self.targetCombo.addItems([str(v) for v in domain.classVar.values])
        
        if self.targetClass<len(domain.classVar.values):
            self.targetCombo.setCurrentIndex(self.targetClass)
        else:
            self.targetCombo.setCurrentIndex(0)
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
            if self.applyBtn.isEnabled():
                self.recompute(True)
            else:
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
                 for l in self.learners.values() if l.scores and l.results]
            
        if not (self.data and len(valid)):
            self.send("Evaluation Results", None)
            return

        # find the result set for a largest number of learners
        # and remove this set from the list of result sets
        rlist = dict([(l.results,1) for l in self.learners.values() if l.scores]).keys()
        rlen = [r.numberOfLearners for r in rlist]
        results = rlist.pop(rlen.index(max(rlen)))
        
        for (i, l) in enumerate(results.learners):
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

    def newsampling(self):
        """handle change of evaluation method"""
        if not self.applyOnAnyChange:
            self.applyBtn.setDisabled(self.applyOnAnyChange)
        else:
            if self.learners:
                self.recompute()

    def newscoreselection(self):
        """handle change in set of scores to be displayed"""
        usestat = [self.selectedRScores, self.selectedCScores][self.isclassification()]
        for i in range(len(self.stat)):
            if i in usestat:
                self.tab.showColumn(i+1)
                self.tab.resizeColumnToContents(i+1)
            else:
                self.tab.hideColumn(i+1)

    def recompute(self, forced=False):
        """recompute the scores for all learners,
           if not forced, will do nothing but enable the Apply button"""
        if self.applyOnAnyChange or forced:
            self.score([l.id for l in self.learners.values()])
            self.paintscores()
            self.applyBtn.setDisabled(True)
        else:
            self.applyBtn.setEnabled(True)

    def conditionalRecompute(self, option):
        """calls recompute only if specific sampling option enabled"""
        if self.resampling == option:
            self.recompute(False)

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
    ow.show()
    a.exec_()

    data1 = orange.ExampleTable(r'../../doc/datasets/voting')
    data2 = orange.ExampleTable(r'../../golf')
    datar = orange.ExampleTable(r'../../auto-mpg')
    data3 = orange.ExampleTable(r'../../sailing-big')
    data4 = orange.ExampleTable(r'../../sailing-test')

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

    testcase = 4

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
    if testcase == 3: # regression first
        ow.setData(datar)
        ow.setLearner(r5, 5)
    if testcase == 4: # separate train and test data
        ow.setData(data3)
        ow.setTestData(data4)
        ow.setLearner(l2, 5)
        ow.setTestData(None)

    ow.saveSettings()
