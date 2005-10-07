"""
<name>Lift Curve</name>
<description>Displays a lift curve based on evaluation of classifiers.</description>
<icon>LiftCurve.png</icon>
<priority>1020</priority>
"""

from OWTools import *
from OWWidget import *
from OWGraph import *
from OWGUI import *
from OWROC import *

import orngEval
import statc, math

class singleClassLiftCurveGraph(singleClassROCgraph):
    def __init__(self, parent = None, name = None, title = ""):
        singleClassROCgraph.__init__(self, parent, name)

        self.enableYRaxis(1)
        self.setXaxisTitle("P Rate")
        self.setAxisAutoScale(QwtPlot.yRight)
        self.setAxisAutoScale(QwtPlot.yLeft)
        self.setYLaxisTitle("TP")
        self.setShowYRaxisTitle(1)
        self.setYRaxisTitle("Cost")
        
        self.setShowMainTitle(1)
        self.setMainTitle(title)
        self.averagingMethod = 'merge'

    def computeCurve(self, res, classIndex=-1, keepConcavities=1):
        return orngEval.computeLiftCurve(res, classIndex)

    def setNumberOfClassifiersIterationsAndClassifierColors(self, classifierNames, iterationsNum, classifierColor):
        singleClassROCgraph.setNumberOfClassifiersIterationsAndClassifierColors(self, classifierNames, iterationsNum, classifierColor)
        self.setCurveYAxis(self.performanceLineCKey, QwtPlot.yRight)
        self.setCurveSymbol(self.performanceLineCKey, QwtSymbol())

    def setTestSetData(self, splitByIterations, targetClass):
        self.splitByIterations = splitByIterations
        ## generate the "base" unmodified Lift curves
        self.targetClass = targetClass
        iteration = 0
        
        for isplit in splitByIterations:
            # unmodified Lift curve
            P, N, curves = self.computeCurve(isplit, self.targetClass)
            self.setIterationCurves(iteration, curves)
            iteration += 1

    ## the lift curve is the average curve from the selected test sets
    ## no other average curves here
    def calcAverageCurves(self):
        ##
        ## self.averagingMethod == 'merge':
        mergedIterations = orngEval.ExperimentResults(1, self.splitByIterations[0].classifierNames, self.splitByIterations[0].classValues, self.splitByIterations[0].weights, classifiers=self.splitByIterations[0].classifiers, loaded=self.splitByIterations[0].loaded)
        i = 0
        for isplit in self.splitByIterations:
            if self.showIterations[i]:
                for te in isplit.results:
                    mergedIterations.results.append( te )
            i += 1
        self.mergedConvexHullData = []
        if len(mergedIterations.results) > 0:
            self.P, self.N, curves = self.computeCurve(mergedIterations, self.targetClass, 1)
            _, _, convexCurves = self.computeCurve(mergedIterations, self.targetClass, 0)
            classifier = 0
            for c in curves:
                x = [px for (px, py, pf) in c]
                y = [py for (px, py, pf) in c]
                ckey = self.mergedCKeys[classifier]
                self.setCurveData(ckey, x, y)
                classifier += 1
            classifier = 0
            for c in convexCurves:
                self.mergedConvexHullData.append(c) ## put all points of all curves into one big array
                x = [px for (px, py, pf) in c]
                y = [py for (px, py, pf) in c]
                ckey = self.mergedConvexCKeys[classifier]
                self.setCurveData(ckey, x, y)
                classifier += 1

            self.setCurveData(self.diagonalCKey, [0.0, 1.0], [0.0, self.P])               
        else:
            for c in range(len(self.mergedCKeys)):
                self.setCurveData(self.mergedCKeys[c], [], [])
                self.setCurveData(self.mergedConvexCKeys[c], [], [])

    ## always set to 'merge' mode
    def setAveragingMethod(self, m):
        self.averagingMethod = 'merge'
        self.updateCurveDisplay()

    ## performance line
    def calcUpdatePerformanceLine(self):
        ## now draw the closest line to the curve
        b = (self.averagingMethod == 'merge') and self.showPerformanceLine
        self.removeMarkers()
        costx = []
        costy = []

        firstGlobalMinP = 1
        globalMinCost = 0
        globalMinCostPoints = []

        for (x, TP, fp) in self.hullCurveDataForPerfLine:
            first = 1
            minc = 0
            localMinCostPoints = []
            for (cNum, (threshold, FPrate)) in fp:
                cost = self.pvalue*(1.0 - TP/self.P)*self.FNcost + (1.0 - self.pvalue)*FPrate*self.FPcost
                if first or cost < minc:
                    first = 0
                    minc = cost
                    localMinCostPoints = [ (x, minc, threshold, cNum) ]
                else:
                    if cost == minc:
                        localMinCostPoints.append( (x, minc, threshold, cNum) )

            if firstGlobalMinP or minc < globalMinCost:
                firstGlobalMinP = 0
                globalMinCost = minc
                globalMinCostPoints = [l for l in localMinCostPoints]
            else:
                if minc == globalMinCost:
                    globalMinCostPoints.extend(localMinCostPoints)

            costx.append(x)
            costy.append(minc)

        self.setCurveData(self.performanceLineCKey, costx, costy)
        self.curve(self.performanceLineCKey).setEnabled(b)
        self.update()

        nOnMinc = {}
        for (x, minc, threshold, cNum) in globalMinCostPoints:
            s = "c:%.1f, th:%1.3f %s" % (minc, threshold, self.classifierNames[cNum])
            mkey = self.insertMarker(s, QwtPlot.xBottom, QwtPlot.yRight)
            onYCn = nOnMinc.get(str(x), 0)

            lminc = self.invTransform(QwtPlot.yLeft, self.transform(QwtPlot.yRight, minc)) ## ugly
            if onYCn > 0:
                lminc = lminc - onYCn*0.05
                nOnMinc[str(x)] = nOnMinc[str(x)] + 1
                self.setMarkerSymbol(mkey, QwtSymbol())
            else:
                nOnMinc[str(x)] = 1
                self.setMarkerSymbol(mkey, self.performanceLineSymbol)

            lminc = self.invTransform(QwtPlot.yRight, self.transform(QwtPlot.yLeft, lminc)) ## ugly ugly

            self.marker(mkey).setXValue(x)
            self.marker(mkey).setYValue(lminc)
            if x >= 0.90:
                self.marker(mkey).setLabelAlignment(Qt.AlignLeft)
            else:
                self.marker(mkey).setLabelAlignment(Qt.AlignRight)

            self.marker(mkey).setEnabled(b)

    def setPointWidth(self, v):
        self.performanceLineSymbol.setSize(v, v)
        for mkey in self.markerKeys():
            self.setMarkerSymbol(mkey, self.performanceLineSymbol)
        self.update()

class OWLiftCurve(OWROC):
    settingsList = ["PointWidth", "CurveWidth", "ShowDiagonal",
                    "ConvexHullCurveWidth", "HullColor", "ShowConvexHull", "ShowConvexCurves", "EnablePerformance"]
    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Lift Curve Analysis", 1)

        # inputs
        self.inputs=[("Evaluation Results", orngTest.ExperimentResults, self.results, Multiple)]

        # default settings
        self.PointWidth = 7
        self.CurveWidth = 3
        self.ConvexCurveWidth = 1
        self.ShowDiagonal = TRUE
        self.ConvexHullCurveWidth = 3
        self.HullColor = str(Qt.yellow.name())
        self.ShowConvexHull = TRUE
        self.ShowConvexCurves = FALSE
        self.EnablePerformance = TRUE

        #load settings
        self.loadSettings()

        # temp variables
        self.dres = None
        self.classifierColor = None
        self.numberOfClasses  = 0
        self.targetClass = None
        self.numberOfClassifiers = 0
        self.numberOfIterations = 0
        self.graphs = []
        self.maxp = 1000
        self.defaultPerfLinePValues = []

        # performance analysis (temporary values
        self.FPcost = 500.0
        self.FNcost = 500.0
        self.pvalue = 50.0 ##0.400

        # list of values (remember for each class)
        self.FPcostList = []
        self.FNcostList = []
        self.pvalueList = []

        # GUI
        self.grid.expand(3, 3)
        self.graphsGridLayoutQGL = QGridLayout(self.mainArea)
        # save each ROC graph in separate file
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFile)

        ## general tab
        self.tabs = QTabWidget(self.controlArea, 'tabWidget')
        self.generalTab = QVGroupBox(self)

        ## target class
        self.classCombo = OWGUI.comboBox(self.generalTab, self, 'targetClass', box='Target Class', items=[], callback=self.target)
        self.classCombo.setMaximumSize(150, 20)

        ## classifiers selection (classifiersQLB)
        self.classifiersQVGB = QVGroupBox(self.generalTab)
        self.classifiersQVGB.setTitle("Classifiers")
        self.classifiersQLB = QListBox(self.classifiersQVGB)
        self.classifiersQLB.setSelectionMode(QListBox.Multi)
        self.connect(self.classifiersQLB, SIGNAL("selectionChanged()"), self.classifiersSelectionChange)
        self.unselectAllClassifiersQLB = QPushButton("(Un)select all", self.classifiersQVGB)
        self.connect(self.unselectAllClassifiersQLB, SIGNAL("clicked()"), self.SUAclassifiersQLB)

        # show Lift Curve convex hull
        OWGUI.checkBox(self.generalTab, self, 'ShowConvexHull', 'Show Lift Convex Hull', tooltip='', callback=self.setShowConvexHull)
        self.tabs.insertTab(self.generalTab, "General")
        

        # performance analysis
        self.performanceTab = QVGroupBox(self)
        self.performanceTabCosts = QVGroupBox(self.performanceTab)
        OWGUI.checkBox(self.performanceTabCosts, self, 'EnablePerformance', 'Show Cost Function', tooltip='', callback=self.setShowPerformanceAnalysis)

        ## FP and FN cost ranges
        mincost = 1; maxcost = 1000; stepcost = 5;
        self.maxpsum = 100; self.minp = 1; self.maxp = self.maxpsum - self.minp ## need it also in self.pvaluesUpdated
        stepp = 1.0

        OWGUI.hSlider(self.performanceTabCosts, self, 'FPcost', box='FP Cost', minValue=mincost, maxValue=maxcost, step=stepcost, callback=self.costsChanged, ticks=50)
        OWGUI.hSlider(self.performanceTabCosts, self, 'FNcost', box='FN Cost', minValue=mincost, maxValue=maxcost, step=stepcost, callback=self.costsChanged, ticks=50)
        OWGUI.hSlider(self.performanceTabCosts, self, 'pvalue', box='p(cl) [%]', minValue=self.minp, maxValue=self.maxp, step=stepp, callback=self.pvaluesUpdated, ticks=5, labelFormat="%2.1f")
        OWGUI.button(self.performanceTabCosts, self, 'Default p(cl)', self.setDefaultPValues) ## reset p values to default

        ## test set selection (testSetsQLB)
        self.testSetsQVGB = QVGroupBox(self.performanceTab)
        self.testSetsQVGB.setTitle("Test sets")
        self.testSetsQLB = QListBox(self.testSetsQVGB)
        self.testSetsQLB.setSelectionMode(QListBox.Multi)
        self.connect(self.testSetsQLB, SIGNAL("selectionChanged()"), self.testSetsSelectionChange)
        self.unselectAllTestSetsQLB = QPushButton("(Un)select all", self.testSetsQVGB)
        self.connect(self.unselectAllTestSetsQLB, SIGNAL("clicked()"), self.SUAtestSetsQLB)
        self.tabs.insertTab(self.performanceTab, "Analysis")

        # settings tab
        self.settingsTab = QVGroupBox(self)
        OWGUI.hSlider(self.settingsTab, self, 'PointWidth', box='Point Width', minValue=3, maxValue=5, step=9, callback=self.setPointWidth, ticks=1)
        OWGUI.hSlider(self.settingsTab, self, 'CurveWidth', box='Lift Curve Width', minValue=1, maxValue=5, step=1, callback=self.setCurveWidth, ticks=1)
        OWGUI.hSlider(self.settingsTab, self, 'ConvexHullCurveWidth', box='Lift Curve Convex Hull', minValue=2, maxValue=9, step=1, callback=self.setConvexHullCurveWidth, ticks=1)
        OWGUI.checkBox(self.settingsTab, self, 'ShowDiagonal', 'Show Diagonal', tooltip='', callback=self.setShowDiagonal)
        self.tabs.insertTab(self.settingsTab, "Settings")

        self.resize(800, 600)

    def calcAllClassGraphs(self):
        for (cl, g) in enumerate(self.graphs):
            g.setNumberOfClassifiersIterationsAndClassifierColors(self.dres.classifierNames, self.numberOfIterations, self.classifierColor)
            g.setTestSetData(self.dresSplitByIterations, cl)
            g.setShowConvexHull(self.ShowConvexHull)
            g.setShowPerformanceLine(self.EnablePerformance)

            ## user settings
            g.setPointWidth(self.PointWidth)
            g.setCurveWidth(self.CurveWidth)
            g.setShowDiagonal(self.ShowDiagonal)
            g.setConvexHullCurveWidth(self.ConvexHullCurveWidth)
            g.setHullColor(QColor(self.HullColor))

    def results(self, dres):
        self.FPcostList = []
        self.FNcostList = []
        self.pvalueList = []

        if not dres:
            self.targetClass = None
            self.classCombo.clear()
            self.removeGraphs()
            self.testSetsQLB.clear()
            return
        self.dres = dres

        self.classifiersQLB.clear()
        self.testSetsQLB.clear()
        self.removeGraphs()
        self.classCombo.clear()

        self.defaultPerfLinePValues = []
        if self.dres <> None:
            ## classQLB
            self.numberOfClasses = len(self.dres.classValues)
            self.graphs = []

            for i in range(self.numberOfClasses):
                self.FPcostList.append( 500)
                self.FNcostList.append( 500)
                graph = singleClassLiftCurveGraph(self.mainArea, "", "Predicted Class: " + self.dres.classValues[i])
                self.graphs.append( graph )
                self.classCombo.insertItem(self.dres.classValues[i])

            ## classifiersQLB
            self.classifierColor = []
            self.numberOfClassifiers = self.dres.numberOfLearners
            if self.numberOfClassifiers > 1:
                allCforHSV = self.numberOfClassifiers - 1
            else:
                allCforHSV = self.numberOfClassifiers
            for i in range(self.numberOfClassifiers):
                newColor = QColor()
                newColor.setHsv(i*255/allCforHSV, 255, 255)
                self.classifierColor.append( newColor )

            ## testSetsQLB
            self.dresSplitByIterations = orngEval.splitByIterations(self.dres)
            self.numberOfIterations = len(self.dresSplitByIterations)

            self.calcAllClassGraphs()

            ## classifiersQLB
            for i in range(self.numberOfClassifiers):
                newColor = self.classifierColor[i]
                self.classifiersQLB.insertItem(ColorPixmap(newColor), self.dres.classifierNames[i])
            self.classifiersQLB.selectAll(1)

            ## testSetsQLB
            self.testSetsQLB.insertStrList([str(i) for i in range(self.numberOfIterations)])
            self.testSetsQLB.selectAll(1)

            ## calculate default pvalues
            reminder = self.maxp
            for f in orngEval.aprioriDistributions(self.dres):
                v = int(round(f * self.maxp))
                reminder -= v
                if reminder < 0:
                    v = v+reminder
                self.defaultPerfLinePValues.append(v)
                self.pvalueList.append( v)

            self.targetClass = 0 ## select first target
            self.target()
        else:
            self.classifierColor = None
        self.performanceTabCosts.setEnabled(1)
        self.setDefaultPValues()

if __name__ == "__main__":
    a = QApplication(sys.argv)
    owdm = OWLiftCurve()
    a.setMainWidget(owdm)
    owdm.show()
    a.exec_loop()


