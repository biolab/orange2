"""
<name>Learning Curve (A)</name>
<description>Takes a data set and a set of learners and shows a learning curve in a table</description>
<icon>icons/LearningCurveA.png</icon>
<priority>1000</priority>
"""

from OWWidget import *
import OWGUI, orngTest, orngStat

class OWLearningCurveA(OWWidget):
    settingsList = ["folds", "steps", "scoringF", "commitOnChange"]
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'LearningCurveA')

        self.inputs = [("Data", ExampleTable, self.dataset), ("Learner", orange.Learner, self.learner, 0)]
        
        self.folds = 5     # cross validation folds
        self.steps = 10    # points in the learning curve
        self.scoringF = 0  # scoring function
        self.commitOnChange = 1 # compute curve on any change of parameters
        self.loadSettings()
        self.setCurvePoints() # sets self.curvePoints, self.steps equidistantpoints from 1/self.steps to 1
        self.scoring = [("Classification Accuracy", orngStat.CA), ("AUC", orngStat.AUC), ("BrierScore", orngStat.BrierScore), ("Information Score", orngStat.IS), ("Sensitivity", orngStat.sens), ("Specificity", orngStat.spec)]
        self.learners = [] # list of current learners from input channel, tuples (id, learner)
        self.data = None   # data on which to construct the learning curve
        self.curves = []   # list of evaluation results (one per learning curve point)
        self.scores = []   # list of current scores, learnerID:[learner scores]

        # GUI
        box = OWGUI.widgetBox(self.controlArea, "Info")
        self.infoa = OWGUI.widgetLabel(box, 'No data on input.')
        self.infob = OWGUI.widgetLabel(box, 'No learners.')

        OWGUI.separator(self.controlArea)
        box = OWGUI.widgetBox(self.controlArea, "Evaluation Scores")
        scoringNames = [x[0] for x in self.scoring]
        OWGUI.comboBox(box, self, "scoringF", items=scoringNames, callback=self.computeScores)

        OWGUI.separator(self.controlArea)
        box = OWGUI.widgetBox(self.controlArea, "Options")
        OWGUI.spin(box, self, 'folds', 2, 100, step=1, label='Cross validation folds:  ',
                   callback=lambda: self.computeCurve(self.commitOnChange))
        OWGUI.spin(box, self, 'steps', 2, 100, step=1, label='Learning curve points:  ',
                   callback=[self.setCurvePoints, lambda: self.computeCurve(self.commitOnChange)])

        OWGUI.checkBox(box, self, 'commitOnChange', 'Apply setting on any change')
        self.commitBtn = OWGUI.button(box, self, "Apply Setting", callback=self.computeCurve, disabled=1)

        # table widget
        self.table = OWGUI.table(self.mainArea, selectionMode=QTableWidget.NoSelection)
                
        self.resize(500,200)

    ##############################################################################    
    # slots: handle input signals        

    def dataset(self, data):
        if data:
            self.infoa.setText('%d instances in input data set' % len(data))
            self.data = data
            if (len(self.learners)):
                self.computeCurve()
        else:
            self.infoa.setText('No data on input.')
            self.curves = []
            self.scores = []
        self.commitBtn.setEnabled(self.data<>None)

    def learner(self, learner, id=None):
        ids = [x[0] for x in self.learners]
        if not learner: # remove a learner and corresponding results
            if not ids.count(id):
                return # no such learner, removed before
            indx = ids.index(id)
            for i in range(self.steps):
                self.curves[i].remove(indx)
            del self.scores[indx]
            del self.learners[indx]
            self.setTable()
        else:
            if ids.count(id): # update (already seen a learner from this source)
                indx = ids.index(id)
                self.learners[indx] = (id, learner)
                if self.data:
                    curve = self.getLearningCurve([learner])
                    score = [self.scoring[self.scoringF][1](x)[0] for x in curve]
                    self.scores[indx] = score
                    for i in range(self.steps):
                        self.curves[i].add(curve[i], 0, replace=indx)
            else: # add new learner
                self.learners.append((id, learner))
                if self.data:
                    curve = self.getLearningCurve([learner])
                    score = [self.scoring[self.scoringF][1](x)[0] for x in curve]
                    self.scores.append(score)
                    if len(self.curves):
                        for i in range(self.steps):
                            self.curves[i].add(curve[i], 0)
                    else:
                        self.curves = curve
        if len(self.learners):
            self.infob.setText("%d learners on input." % len(self.learners))
        else:
            self.infob.setText("No learners.")
        self.commitBtn.setEnabled(len(self.learners))
##        if len(self.scores):
        if self.data:
            self.setTable()

    ##############################################################################    
    # learning curve table, callbacks

    # recomputes the learning curve
    def computeCurve(self, condition=1):
        if condition:
            learners = [x[1] for x in self.learners]
            self.curves = self.getLearningCurve(learners)
            self.computeScores()

    def computeScores(self):            
        self.scores = [[] for i in range(len(self.learners))]
        for x in self.curves:
            for (i,s) in enumerate(self.scoring[self.scoringF][1](x)):
                self.scores[i].append(s)
        self.setTable()

    def getLearningCurve(self, learners):   
        pb = OWGUI.ProgressBar(self, iterations=self.steps*self.folds)
        curve = orngTest.learningCurveN(learners, self.data, folds=self.folds, proportions=self.curvePoints, callback=pb.advance)
        pb.finish()
        return curve

    def setCurvePoints(self):
        self.curvePoints = [(x+1.)/self.steps for x in range(self.steps)]

    def setTable(self):
        self.table.setColumnCount(0)
        self.table.setColumnCount(len(self.learners))
        self.table.setRowCount(self.steps)

        # set the headers
        self.table.setHorizontalHeaderLabels([l.name for i,l in self.learners])
        self.table.setVerticalHeaderLabels(["%4.2f" % p for p in self.curvePoints])

        # set the table contents
        for l in range(len(self.learners)):
            for p in range(self.steps):
                OWGUI.tableItem(self.table, p, l, "%7.5f" % self.scores[l][p])

        for i in range(len(self.learners)):
            self.table.setColumnWidth(i, 80)

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = OWLearningCurveA()
    ow.show()
    
    l1 = orange.BayesLearner()
    l1.name = 'Naive Bayes'
    ow.learner(l1, 1)

    data = orange.ExampleTable('../datasets/iris.tab')
    ow.dataset(data)

    l2 = orange.BayesLearner()
    l2.name = 'Naive Bayes (m=10)'
    l2.estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m=10)
    l2.conditionalEstimatorConstructor = orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m=10))
    ow.learner(l2, 2)

    import orngTree
    l4 = orngTree.TreeLearner(minSubset=2)
    l4.name = "Decision Tree"
    ow.learner(l4, 4)

#    ow.learner(None, 1)
#    ow.learner(None, 2)
#    ow.learner(None, 4)

    appl.exec_()
