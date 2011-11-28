"""
<name>Lift Curve</name>
<description>Displays a lift curve based on evaluation of classifiers.</description>
<contact>Tomaz Curk</contact>
<icon>icons/LiftCurve.png</icon>
<priority>1020</priority>
"""
from OWColorPalette import ColorPixmap
from OWWidget import *
from OWGraph import *
from OWGUI import *
from OWROC import *

import orngStat, orngEval
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
        return orngStat.computeLiftCurve(res, classIndex)

    def setNumberOfClassifiersIterationsAndClassifierColors(self, classifierNames, iterationsNum, classifierColor):
        singleClassROCgraph.setNumberOfClassifiersIterationsAndClassifierColors(self, classifierNames, iterationsNum, classifierColor)
        self.performanceLineCKey.setYAxis(QwtPlot.yRight)
        self.performanceLineCKey.setSymbol(QwtSymbol())

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
        for show, isplit in zip(self.showIterations, self.splitByIterations):
            if show:
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
                curve = self.mergedCKeys[classifier]
                curve.setData(x, y)
                classifier += 1
            classifier = 0
            for c in convexCurves:
                self.mergedConvexHullData.append(c) ## put all points of all curves into one big array
                x = [px for (px, py, pf) in c]
                y = [py for (px, py, pf) in c]
                curve = self.mergedConvexCKeys[classifier]
                curve.setData(x, y)
                classifier += 1

            self.diagonalCKey.setData([0.0, 1.0], [0.0, self.P])
        else:
            for c in range(len(self.mergedCKeys)):
                self.mergedCKeys[c].setData([], [])
                self.mergedConvexCKeys[c].setData([], [])

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
                if TP > self.P:
                    import warnings
                    warnings.warn("The sky is falling!!")
                cost = self.pvalue*(1.0 - TP/(self.P or 1))*self.FNcost + (1.0 - self.pvalue)*FPrate*self.FPcost
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

        if self.performanceLineCKey: #self.curve(self.performanceLineCKey):
            self.performanceLineCKey.setData(costx, costy)
            self.performanceLineCKey.setVisible(b)
        self.replot()
#        self.update()

        nOnMinc = {}
        for (x, minc, threshold, cNum) in globalMinCostPoints:
            s = "c:%.1f, th:%1.3f %s" % (minc, threshold, self.classifierNames[cNum])
            marker = self.addMarker(s, 0, 0)
            marker.setAxis(QwtPlot.xBottom, QwtPlot.yRight)
            onYCn = nOnMinc.get(str(x), 0)

            lminc = self.invTransform(QwtPlot.yLeft, self.transform(QwtPlot.yRight, minc)) ## ugly
            if onYCn > 0:
                lminc = lminc - onYCn*0.05
                nOnMinc[str(x)] = nOnMinc[str(x)] + 1
                marker.setSymbol(QwtSymbol())
            else:
                nOnMinc[str(x)] = 1
                marker.setSymbol(self.performanceLineSymbol)

            lminc = self.invTransform(QwtPlot.yRight, self.transform(QwtPlot.yLeft, lminc)) ## ugly ugly

            marker.setXValue(x)
            marker.setYValue(lminc)
            if x >= 0.90:
                marker.setLabelAlignment(Qt.AlignLeft)
            else:
                marker.setLabelAlignment(Qt.AlignRight)

            marker.setVisible(b)

    def setPointWidth(self, v):
        self.performanceLineSymbol.setSize(v, v)
        for marker in [item for item in self.itemList() if isinstance(item, QwtPlotMarker)]:
            marker.setSymbol(self.performanceLineSymbol)
        self.replot()
#        self.update()

class OWLiftCurve(OWROC):
    settingsList = ["PointWidth", "CurveWidth", "ShowDiagonal",
                    "ConvexHullCurveWidth", "HullColor", "ShowConvexHull", "ShowConvexCurves", "EnablePerformance"]
    contextHandlers = {"": EvaluationResultsContextHandler("", "targetClass", "selectedClassifiers")}

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Lift Curve Analysis", 1)

        # inputs
        self.inputs=[("Evaluation Results", orngTest.ExperimentResults, self.results, Default)]

        # default settings
        self.PointWidth = 7
        self.CurveWidth = 3
        self.ConvexCurveWidth = 1
        self.ShowDiagonal = TRUE
        self.ConvexHullCurveWidth = 3
        self.HullColor = str(QColor(Qt.yellow).name())
        self.ShowConvexHull = TRUE
        self.ShowConvexCurves = FALSE
        self.EnablePerformance = TRUE
        self.classifiers = []
        self.selectedClassifiers = []

        #load settings
        self.loadSettings()

### Moved here to override the saved settings since the controls do not exist any more
        self.CurveWidth = 3
        self.ConvexCurveWidth = 1
        self.ConvexHullCurveWidth = 3

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
        #self.grid.expand(3, 3)
        import sip
        sip.delete(self.mainArea.layout())
        self.graphsGridLayoutQGL = QGridLayout(self.mainArea)
        self.mainArea.setLayout(self.graphsGridLayoutQGL)

        # save each ROC graph in separate file
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFile)

        ## general tab
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.generalTab = OWGUI.createTabPage(self.tabs, "General")

        
        ## target class
        self.classCombo = OWGUI.comboBox(self.generalTab, self, 'targetClass', box='Target class', items=[], callback=self.target)
        OWGUI.separator(self.generalTab)

        ## classifiers selection (classifiersQLB)
        self.classifiersQVGB = OWGUI.widgetBox(self.generalTab, "Classifiers", addSpace=True)
        self.classifiersQLB = OWGUI.listBox(self.classifiersQVGB, self, "selectedClassifiers", selectionMode = QListWidget.MultiSelection, callback = self.classifiersSelectionChange)
        self.unselectAllClassifiersQLB = OWGUI.button(self.classifiersQVGB, self, "(Un)select all", callback = self.SUAclassifiersQLB)
##        OWGUI.checkBox(self.classifiersQVGB, self, 'ShowConvexHull', 'Show convex lift hull', tooltip='', callback=self.setShowConvexHull)
##        OWGUI.checkBox(self.classifiersQVGB, self, 'ShowDiagonal', 'Show diagonal', tooltip='', callback=self.setShowDiagonal)

        # show Lift Curve convex hull
        OWGUI.checkBox(self.generalTab, self, 'ShowConvexHull', 'Show lift convex hull', tooltip='', callback=self.setShowConvexHull)
                

        # performance analysis
        self.performanceTab = OWGUI.createTabPage(self.tabs, "Analysis")
        self.performanceTabCosts = OWGUI.widgetBox(self.performanceTab)
        OWGUI.checkBox(self.performanceTabCosts, self, 'EnablePerformance', 'Show cost function', tooltip='', callback=self.setShowPerformanceAnalysis)

        ## FP and FN cost ranges
        mincost = 1; maxcost = 1000; stepcost = 5;
        self.maxpsum = 100; self.minp = 1; self.maxp = self.maxpsum - self.minp ## need it also in self.pvaluesUpdated
        stepp = 1.0

        OWGUI.widgetLabel(self.performanceTabCosts, "False positive cost")
        OWGUI.hSlider(OWGUI.indentedBox(self.performanceTabCosts), self, 'FPcost', minValue=mincost, maxValue=maxcost, step=stepcost, callback=self.costsChanged, ticks=50)
        OWGUI.widgetLabel(self.performanceTabCosts, "False negative cost")
        OWGUI.hSlider(OWGUI.indentedBox(self.performanceTabCosts), self, 'FNcost', minValue=mincost, maxValue=maxcost, step=stepcost, callback=self.costsChanged, ticks=50)

        OWGUI.widgetLabel(self.performanceTabCosts, "Prior target class probability [%]")
        ptc = OWGUI.indentedBox(self.performanceTabCosts)
        OWGUI.hSlider(ptc, self, 'pvalue', minValue=self.minp, maxValue=self.maxp, step=stepp, callback=self.pvaluesUpdated, ticks=5, labelFormat="%2.1f")
        OWGUI.separator(ptc)
        OWGUI.button(ptc, self, 'Compute from data', self.setDefaultPValues) ## reset p values to default


        ## test set selection (testSetsQLB)
        self.testSetsQVGB = OWGUI.widgetBox(self.performanceTab, "Test sets")
        self.testSetsQLB = OWGUI.listBox(self.testSetsQVGB, self, selectionMode = QListWidget.MultiSelection, callback = self.testSetsSelectionChange)
        self.unselectAllTestSetsQLB = OWGUI.button(self.testSetsQVGB, self, "(Un)select all", callback = self.SUAtestSetsQLB)

        # settings tab
        self.settingsTab = OWGUI.createTabPage(self.tabs, "Settings")
        OWGUI.hSlider(self.settingsTab, self, 'PointWidth', box='Point width', minValue=0, maxValue=9, step=1, callback=self.setPointWidth, ticks=1)
        OWGUI.hSlider(self.settingsTab, self, 'CurveWidth', box='Lift curve width', minValue=1, maxValue=5, step=1, callback=self.setCurveWidth, ticks=1)
        OWGUI.hSlider(self.settingsTab, self, 'ConvexHullCurveWidth', box='Lift curve convex hull', minValue=2, maxValue=9, step=1, callback=self.setConvexHullCurveWidth, ticks=1)
        OWGUI.checkBox(self.settingsTab, self, 'ShowDiagonal', 'Show diagonal', tooltip='', callback=self.setShowDiagonal)
        OWGUI.rubber(self.settingsTab)
##        self.SettingsTab.addStretch(100)

#        OWGUI.rubber(self.controlArea)
        self.resize(770, 530)

    def sendReport(self):
        # need to reimport - Qt provides something stupid instead
        from __builtin__ import hex
        self.reportSettings("Settings",
                            [("Classifiers", ", ".join('<font color="#%s">%s</font>' % ("".join(("0"+hex(x)[2:])[-2:] for x in self.classifierColor[cNum].getRgb()[:3]), str(item.text()))
                                                        for cNum, item in enumerate(self.classifiersQLB.item(i) for i in range(self.classifiersQLB.count()))
                                                          if item.isSelected())),
                             ("Target class", self.classCombo.itemText(self.targetClass)
                                              if self.targetClass is not None else
                                              "N/A"),
                             ("Costs", "FP=%i, FN=%i" % (self.FPcost, self.FNcost)),
                             ("Prior target class probability", "%i%%" % self.pvalue)
                            ])
        if self.targetClass is not None:
            self.reportRaw("<br/>")
            self.reportImage(self.graphs[self.targetClass].saveToFileDirect, QSize(500, 400))


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
        self.closeContext()

        self.FPcostList = []
        self.FNcostList = []
        self.pvalueList = []

        self.classCombo.clear()
        self.removeGraphs()
        self.testSetsQLB.clear()
        self.classifiersQLB.clear()

        self.dres = dres

        if not dres:
            self.targetClass = None
            self.openContext("", dres)
            return
        
        self.warning(0)
        if len(dres.results) > 0 and dres.results[0].multilabel_flag == 1:
            text = "there is no consensus on how to apply it in multi-class problems"
            self.warning(0, text)
            return
        
        self.defaultPerfLinePValues = []
        if self.dres <> None:
            ## classQLB
            self.numberOfClasses = len(self.dres.classValues)
            self.graphs = []

            for i in range(self.numberOfClasses):
                self.FPcostList.append( 500)
                self.FNcostList.append( 500)
                graph = singleClassLiftCurveGraph(self.mainArea, "", "Predicted class: " + self.dres.classValues[i])
                self.graphs.append( graph )
                self.classCombo.addItem(self.dres.classValues[i])

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
            self.dresSplitByIterations = orngStat.splitByIterations(self.dres)
            self.numberOfIterations = len(self.dresSplitByIterations)

            self.calcAllClassGraphs()

            ## classifiersQLB
            for i in range(self.numberOfClassifiers):
                newColor = self.classifierColor[i]
                self.classifiersQLB.addItem(QListWidgetItem(ColorPixmap(newColor), self.dres.classifierNames[i]))
            self.classifiersQLB.selectAll()

            ## testSetsQLB
            self.testSetsQLB.addItems([str(i) for i in range(self.numberOfIterations)])
            self.testSetsQLB.selectAll()

            ## calculate default pvalues
            reminder = self.maxp
            for f in orngStat.classProbabilitiesFromRes(self.dres):
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
        self.openContext("", self.dres)
        self.performanceTabCosts.setEnabled(1)
        self.setDefaultPValues()

if __name__ == "__main__":
    a = QApplication(sys.argv)
    owdm = OWLiftCurve()
    owdm.show()
    a.exec_()


