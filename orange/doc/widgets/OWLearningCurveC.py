"""
<name>Learning Curve (C)</name>
<description>Takes a data set and a set of learners and plots a learning curve in a table</description>
<icon>icons/LearningCurveC.png</icon>
<priority>1020</priority>
"""

from OWWidget import *
from OWTools import ColorPixmap
import OWGUI, orngTest, orngStat
from qttable import *
from OWGraph import ColorPaletteHSV
from qwt import *
import warnings

class OWLearningCurveC(OWWidget):
    settingsList = ["folds", "steps", "scoringF", "commitOnChange",
                    "graphPointSize", "graphDrawLines", "graphShowGrid"]
    
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, 'LearningCurveA')

        self.inputs = [("Data", ExampleTable, self.dataset), ("Learner", orange.Learner, self.learner, 0)]
        
        self.folds = 5     # cross validation folds
        self.steps = 10    # points in the learning curve
        self.scoringF = 0  # scoring function
        self.commitOnChange = 1 # compute curve on any change of parameters
        self.graphPointSize = 5 # size of points in the graphs
        self.graphDrawLines = 1 # draw lines between points in the graph
        self.graphShowGrid = 1  # show gridlines in the graph
        self.loadSettings()
        
        warnings.filterwarnings("ignore", ".*builtin attribute.*", orange.AttributeWarning)

        self.setCurvePoints() # sets self.curvePoints, self.steps equidistantpoints from 1/self.steps to 1
        self.scoring = [("Classification Accuracy", orngStat.CA), ("AUC", orngStat.AUC), ("BrierScore", orngStat.BrierScore), ("Information Score", orngStat.IS), ("Sensitivity", orngStat.sens), ("Specificity", orngStat.spec)]
        self.learners = [] # list of current learners from input channel, tuples (id, learner)
        self.data = None   # data on which to construct the learning curve
        self.curves = []   # list of evaluation results (one per learning curve point)
        self.scores = []   # list of current scores, learnerID:[learner scores]

        # GUI
        box = QVGroupBox("Info", self.controlArea)
        self.infoa = QLabel('No data on input.', box)
        self.infob = QLabel('No learners.', box)

        ## class selection (classQLB)
        OWGUI.separator(self.controlArea)
        self.cbox = QVGroupBox("Learners", self.controlArea)
        self.llb = QListBox(self.cbox)
        self.llb.setSelectionMode(QListBox.Multi)
        self.llb.setMinimumHeight(50)
        self.connect(self.llb, SIGNAL("selectionChanged()"),
                     self.learnerSelectionChanged)
        self.blockSelectionChanges = 0

        OWGUI.separator(self.controlArea)
        box = QVGroupBox("Evaluation Scores", self.controlArea)
        scoringNames = [x[0] for x in self.scoring]
        OWGUI.comboBox(box, self, "scoringF", items=scoringNames,
                       callback=self.computeScores)

        OWGUI.separator(self.controlArea)
        box = QVGroupBox("Options", self.controlArea)
        OWGUI.spin(box, self, 'folds', 2, 100, step=1,
                   label='Cross validation folds:  ',
                   callback=lambda: self.computeCurve(self.commitOnChange))
        OWGUI.spin(box, self, 'steps', 2, 100, step=1,
                   label='Learning curve points:  ',
                   callback=[self.setCurvePoints, lambda: self.computeCurve(self.commitOnChange)])

        OWGUI.checkBox(box, self, 'commitOnChange', 'Apply setting on any change')
        self.commitBtn = OWGUI.button(box, self, "Apply Setting", callback=self.computeCurve, disabled=1)

        # start of content (right) area
        self.layout = QVBoxLayout(self.mainArea)
        tabs = QTabWidget(self.mainArea, 'tabs')
        
        # graph widget
        tab = QVGroupBox(self)
        self.graph = QwtPlot(tab, None)
        self.graph.setAxisAutoScale(QwtPlot.xBottom)
        self.graph.setAxisAutoScale(QwtPlot.yLeft)
        tabs.insertTab(tab, "Graph")
        self.setGraphGrid()

        # table widget
        tab = QVGroupBox(self)
        self.table = QTable(tab)
        self.table.setSelectionMode(QTable.NoSelection)
        self.header = self.table.horizontalHeader()
        self.vheader = self.table.verticalHeader()
        tabs.insertTab(tab, "Table")

        self.layout.add(tabs)
        self.resize(550,200)

    ##############################################################################    
    # slots: handle input signals        

    def dataset(self, data):
        if data:
            self.infoa.setText('%d instances in input data set' % len(data))
            self.data = data
            if (len(self.learners)):
                self.computeCurve()
            self.replotGraph()
        else:
            self.infoa.setText('No data on input.')
            self.curves = []
            self.scores = []
            self.graph.removeCurves()
            self.graph.replot()
        self.commitBtn.setEnabled(self.data<>None)

    # manage learner signal
    # we use following additional attributes for learner:
    # - isSelected, learner is selected (display the learning curve)
    # - curvekey, id of the learning curve plot for the learner
    # - score, evaluation score for the learning
    def learner(self, learner, id=None):
        ids = [x[0] for x in self.learners]
        if not learner: # remove a learner and corresponding results
            if not ids.count(id):
                return # no such learner, removed before
            indx = ids.index(id)
            for i in range(self.steps):
                self.curves[i].remove(indx)
            del self.scores[indx]
            self.graph.removeCurve(self.learners[indx].curvekey)
            del self.learners[indx]
            self.setTable()
            self.updatellb()
        else:
            if ids.count(id): # update (already seen a learner from this source)
                indx = ids.index(id)
                prevLearner = self.learners[indx][1]
                learner.isSelected = prevLearner.isSelected
                self.learners[indx] = (id, learner)
                if self.data:
                    curve = self.getLearningCurve([learner])
                    score = [self.scoring[self.scoringF][1](x)[0] for x in curve]
                    self.scores[indx] = score
                    for i in range(self.steps):
                        self.curves[i].add(curve[i], 0, replace=indx)
                    learner.score = score
                    self.graph.removeCurve(prevLearner.curvekey)
                    self.drawLearningCurve(learner)
                self.updatellb()
            else: # add new learner
                learner.isSelected = 1
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
                    learner.score = score
                self.updatellb()
                self.drawLearningCurve(learner)
        if len(self.learners):
            self.infob.setText("%d learners on input." % len(self.learners))
        else:
            self.infob.setText("No learners.")
        self.commitBtn.setEnabled(len(self.learners))
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
        for (i,l) in enumerate(self.learners):
            l[1].score = self.scores[i]
        self.setTable()
        self.replotGraph()

    def getLearningCurve(self, learners):   
        pb = OWGUI.ProgressBar(self, iterations=self.steps*self.folds)
        curve = orngTest.learningCurveN(learners, self.data, folds=self.folds, proportions=self.curvePoints, callback=pb.advance)
        pb.finish()
        return curve

    def setCurvePoints(self):
        self.curvePoints = [(x+1.)/self.steps for x in range(self.steps)]

    def setTable(self):
        self.table.setNumCols(0)
        self.table.setNumCols(len(self.learners))
        self.table.setNumRows(self.steps)

        # set the headers
        for (i, l) in enumerate(self.learners):
            self.header.setLabel(i, l[1].name)
        for (i, p) in enumerate(self.curvePoints):
            self.vheader.setLabel(i, "%4.2f" % p)

        # set the table contents
        for l in range(len(self.learners)):
            for p in range(self.steps):
                self.table.setText(p, l, "%7.5f" % self.scores[l][p])

        for i in range(len(self.learners)):
            self.table.setColumnWidth(i, 80)

    # management of learner selection

    def updatellb(self):
        self.blockSelectionChanges = 1
        self.llb.clear()
        colors = ColorPaletteHSV(len(self.learners))
        for (i,lt) in enumerate(self.learners):
            l = lt[1]
            self.llb.insertItem(ColorPixmap(colors[i]), l.name)
            self.llb.setSelected(i, l.isSelected)
            l.color = colors[i]
        self.blockSelectionChanges = 0

    def learnerSelectionChanged(self):
        if self.blockSelectionChanges: return
        for (i,lt) in enumerate(self.learners):
            l = lt[1]
            if l.isSelected <> self.llb.isSelected(i):
                if l.isSelected: # learner was deselected
                    self.graph.removeCurve(l.curvekey)
                else: # learner was selected
                    self.drawLearningCurve(l)
                self.graph.replot()
            l.isSelected = self.llb.isSelected(i)

    # Graph specific methods

    def setGraphGrid(self):
        self.graph.enableGridY(self.graphShowGrid)
        self.graph.enableGridX(self.graphShowGrid)

    def setGraphStyle(self, learner):
        curvekey = learner.curvekey
        if self.graphDrawLines:
            self.graph.setCurveStyle(curvekey, QwtCurve.Lines)
        else:
            self.graph.setCurveStyle(curvekey, QwtCurve.NoCurve)
        self.graph.setCurveSymbol(curvekey, QwtSymbol(QwtSymbol.Ellipse, \
          QBrush(QColor(0,0,0)), QPen(QColor(0,0,0)),
          QSize(self.graphPointSize, self.graphPointSize)))
        self.graph.setCurvePen(curvekey, QPen(learner.color, 5))

    def drawLearningCurve(self, learner):
        if not self.data: return
        curvekey = self.graph.insertCurve(learner.name)
        self.graph.setCurveData(curvekey, self.curvePoints, learner.score)
        learner.curvekey = curvekey
        self.setGraphStyle(learner)
        self.graph.replot()

    def replotGraph(self):
        self.graph.removeCurves()   # first remove all curves
        for l in self.learners:
            self.drawLearningCurve(l[1])

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = OWLearningCurveC()
    appl.setMainWidget(ow)
    ow.show()
    
    l1 = orange.BayesLearner()
    l1.name = 'Naive Bayes'
    ow.learner(l1, 1)

    data = orange.ExampleTable('iris.tab')
    ow.dataset(data)

    l2 = orange.BayesLearner()
    l2.name = 'Naive Bayes (m=10)'
    l2.estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m=10)
    l2.conditionalEstimatorConstructor = orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m=10))

    l3 = orange.kNNLearner(name="k-NN")
    ow.learner(l3, 3)

    import orngTree
    l4 = orngTree.TreeLearner(minSubset=2)
    l4.name = "Decision Tree"
    ow.learner(l4, 4)

#    ow.learner(None, 1)
#    ow.learner(None, 2)
#    ow.learner(None, 4)

    appl.exec_loop()
