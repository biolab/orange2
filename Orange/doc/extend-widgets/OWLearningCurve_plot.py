"""
<name>Learning Curve (C)</name>
<description>Takes a data set and a set of learners and plots a learning curve in a table</description>
<icon>icons/LearningCurveC.png</icon>
<priority>1020</priority>
"""

from OWWidget import *
from OWColorPalette import ColorPixmap
import OWGUI, orngTest, orngStat
from owplot import *

import warnings

class OWLearningCurveC(OWWidget):
    settingsList = ["folds", "steps", "scoringF", "commitOnChange",
                    "graphPointSize", "graphDrawLines", "graphShowGrid"]

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'LearningCurveC')

        self.inputs = [("Data", ExampleTable, self.dataset), ("Learner", orange.Learner, self.learner, 0)]

        self.folds = 5     # cross validation folds
        self.steps = 10    # points in the learning curve
        self.scoringF = 0  # scoring function
        self.commitOnChange = 1 # compute curve on any change of parameters
        self.graphPointSize = 5 # size of points in the graphs
        self.graphDrawLines = 1 # draw lines between points in the graph
        self.graphShowGrid = 1  # show gridlines in the graph
        self.selectedLearners = [] 
        self.loadSettings()

        warnings.filterwarnings("ignore", ".*builtin attribute.*", orange.AttributeWarning)

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

        ## class selection (classQLB)
        OWGUI.separator(self.controlArea)
        self.cbox = OWGUI.widgetBox(self.controlArea, "Learners")
        self.llb = OWGUI.listBox(self.cbox, self, "selectedLearners", selectionMode=QListWidget.MultiSelection, callback=self.learnerSelectionChanged)
        
        self.llb.setMinimumHeight(50)
        self.blockSelectionChanges = 0

        OWGUI.separator(self.controlArea)
        box = OWGUI.widgetBox(self.controlArea, "Evaluation Scores")
        scoringNames = [x[0] for x in self.scoring]
        OWGUI.comboBox(box, self, "scoringF", items=scoringNames,
                       callback=self.computeScores)

        OWGUI.separator(self.controlArea)
        box = OWGUI.widgetBox(self.controlArea, "Options")
        OWGUI.spin(box, self, 'folds', 2, 100, step=1,
                   label='Cross validation folds:  ',
                   callback=lambda: self.computeCurve(self.commitOnChange))
        OWGUI.spin(box, self, 'steps', 2, 100, step=1,
                   label='Learning curve points:  ',
                   callback=[self.setCurvePoints, lambda: self.computeCurve(self.commitOnChange)])

        OWGUI.checkBox(box, self, 'commitOnChange', 'Apply setting on any change')
        self.commitBtn = OWGUI.button(box, self, "Apply Setting", callback=self.computeCurve, disabled=1)

        # start of content (right) area
        tabs = OWGUI.tabWidget(self.mainArea)

        # graph widget
        tab = OWGUI.createTabPage(tabs, "Graph")
        self.graph = OWPlot(tab)
        self.graph.set_axis_autoscale(xBottom)
        self.graph.set_axis_autoscale(yLeft)
        tab.layout().addWidget(self.graph)
        self.setGraphGrid()

        # table widget
        tab = OWGUI.createTabPage(tabs, "Table")
        self.table = OWGUI.table(tab, selectionMode=QTableWidget.NoSelection)

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
            self.graph.clear()
            self.graph.replot()
        self.commitBtn.setEnabled(self.data<>None)

    # manage learner signal
    # we use following additional attributes for learner:
    # - isSelected, learner is selected (display the learning curve)
    # - curve, learning curve for the learner
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
            self.learners[indx][1].curve.detach()
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
                    prevLearner.curve.detach()
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


    # management of learner selection

    def updatellb(self):
        self.blockSelectionChanges = 1
        self.llb.clear()
        colors = ColorPaletteHSV(len(self.learners))
        for (i,lt) in enumerate(self.learners):
            l = lt[1]
            item = QListWidgetItem(ColorPixmap(colors[i]), l.name)
            self.llb.addItem(item)
            item.setSelected(l.isSelected)
            l.color = colors[i]
        self.blockSelectionChanges = 0

    def learnerSelectionChanged(self):
        if self.blockSelectionChanges: return
        for (i,lt) in enumerate(self.learners):
            l = lt[1]
            if l.isSelected != (i in self.selectedLearners):
                if l.isSelected: # learner was deselected
                    l.curve.detach()
                else: # learner was selected
                    self.drawLearningCurve(l)
                self.graph.replot()
            l.isSelected = i in self.selectedLearners

    # Graph specific methods

    def setGraphGrid(self):
        self.graph.enableGridYL(self.graphShowGrid)
        self.graph.enableGridXB(self.graphShowGrid)

    def setGraphStyle(self, learner):
        curve = learner.curve
        if self.graphDrawLines:
            curve.set_style(OWCurve.LinesPoints)
        else:
            curve.set_style(OWCurve.Points)
        curve.set_symbol(OWPoint.Ellipse)
        curve.set_point_size(self.graphPointSize)
        curve.set_color(self.graph.color(OWPalette.Data))
        curve.set_pen(QPen(learner.color, 5))

    def drawLearningCurve(self, learner):
        if not self.data: return
        curve = self.graph.add_curve(learner.name, xData=self.curvePoints, yData=learner.score, autoScale=True)
        
        learner.curve = curve
        self.setGraphStyle(learner)
        self.graph.replot()

    def replotGraph(self):
        self.graph.clear()
        for l in self.learners:
            self.drawLearningCurve(l[1])

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = OWLearningCurveC()
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

    l3 = orange.kNNLearner(name="k-NN")
    ow.learner(l3, 3)

    import orngTree
    l4 = orngTree.TreeLearner(minSubset=2)
    l4.name = "Decision Tree"
    ow.learner(l4, 4)

#    ow.learner(None, 1)
#    ow.learner(None, 2)
#    ow.learner(None, 4)
    


    appl.exec_()
