"""
<name>Lift Curve</name>
<description>None.</description>
<category>Evaluation</category>
<icon>icons\LiftCurve.png</icon>
<priority>1020</priority>
"""

from OData import *
from OWTools import *
from OWWidget import *
from OWGraph import *
from OWGUI import *
from OWROC import *
from OWLiftCurveOptions import *

import orngEval
import statc, math

class singleClassLiftCurveGraph(singleClassROCgraph):
    def __init__(self, parent = None, name = None, title = ""):
        singleClassROCgraph.__init__(self, parent, name)
        self.averagingMethod = 'merge'
        self.setAxisAutoScale(QwtPlot.yRight)
        self.enableYRaxis(1)

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
            curves = self.computeCurve(isplit, self.targetClass)
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
            curves = self.computeCurve(mergedIterations, self.targetClass, 1)
            convexCurves = self.computeCurve(mergedIterations, self.targetClass, 0)
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

        for (x, TPrate, fp) in self.hullCurveDataForPerfLine:
            first = 1
            minc = 0
            localMinCostPoints = []
            for (cNum, (threshold, FPrate)) in fp:
                cost = self.pvalue*(1.0 - TPrate)*self.FNcost + (1.0 - self.pvalue)*FPrate*self.FPcost
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
                    "ConvexHullCurveWidth", "HullColor"]
    def __init__(self,parent=None):
        "Constructor"
        OWWidget.__init__(self,
        parent,
        "&Lift Curve",
        """None.
        """,
        TRUE,
        TRUE)

        #set default settings
        self.PointWidth = 7
        self.CurveWidth = 3
        self.ShowDiagonal = TRUE
        self.ConvexHullCurveWidth = 3
        self.HullColor = str(Qt.yellow.name())

        #load settings
        self.loadSettings()

        # GUI
        self.missClassificationCostQVB = QVGroupBox(self)
        self.missClassificationCostQVB.hide()
        self.grid.expand(3, 3)
        self.grid.addMultiCellWidget(self.missClassificationCostQVB,0,3,2,2)

        self.graphsGridLayoutQGL = QGridLayout(self.mainArea)
##        self.graphsGridLayoutQGL.setResizeMode(QGridLayout.Fixed)
        ## save each Lift Curve graph in separate file
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFile)
        self.graphButton.setText("&Save Graphs (each in its own file)")

        # inputs
        # data and graph temp variables
        self.addInput("results")

        # temp variables
        self.dres = None
        self.classifierColor = None
        self.numberOfClasses  = 0
        self.numberOfClassifiers = 0
        self.numberOfIterations = 0
        self.graphs = []
        self.maxp = 1000
        self.defaultPerfLinePValues = []

        self.options = OWLiftCurveOptions()
        self.activateLoadedSettings()

        #connect settingsbutton to show options
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)

        #connect GUI controls of options in options dialog to settings
        self.connect(self.options.pointWidthSlider, SIGNAL("valueChanged(int)"), self.setPointWidth)
        self.connect(self.options.lineWidthSlider, SIGNAL("valueChanged(int)"), self.setCurveWidth)
        self.connect(self.options.showDiagonalQCB, SIGNAL("toggled(bool)"), self.setShowDiagonal)
        self.connect(self.options.hullWidthSlider, SIGNAL("valueChanged(int)"), self.setConvexHullCurveWidth)
        self.connect(self.options, PYSIGNAL("hullColorChange(QColor &)"), self.setHullColor)

        # GUI connections
        self.splitQS = QSplitter(self.space)
        self.splitQS.setOrientation(Qt.Vertical)

        ## class selection (classQLB)
        self.classQVGB = QVGroupBox(self.splitQS)
        self.classQVGB.setTitle("Classes")
        self.classQLB = QListBox(self.classQVGB)
        self.classQLB.setSelectionMode(QListBox.Multi)
        self.unselectAllClassedQLB = QPushButton("(Un)select all", self.classQVGB)
        self.connect(self.unselectAllClassedQLB, SIGNAL("clicked()"), self.SUAclassQLB)
        self.connect(self.classQLB, SIGNAL("selectionChanged()"), self.classSelectionChange)

        ## classifiers selection (classifiersQLB)
        self.classifiersQVGB = QVGroupBox(self.splitQS)
        self.classifiersQVGB.setTitle("Classifiers")
        self.classifiersQLB = QListBox(self.classifiersQVGB)
        self.classifiersQLB.setSelectionMode(QListBox.Multi)
        self.connect(self.classifiersQLB, SIGNAL("selectionChanged()"), self.classifiersSelectionChange)
        self.unselectAllClassifiersQLB = QPushButton("(Un)select all", self.classifiersQVGB)
        self.connect(self.unselectAllClassifiersQLB, SIGNAL("clicked()"), self.SUAclassifiersQLB)

        # show Lift Curve convex hull
        self.convexhullQCB = QCheckBox("Lift Curve convex hull", self.classifiersQVGB)
        self.connect(self.convexhullQCB, SIGNAL("stateChanged(int)"), self.setShowConvexHull)

        ## test set selection (testSetsQLB)
        self.testSetsQVGB = QVGroupBox(self.splitQS)
        self.testSetsQVGB.setTitle("Test sets")
        self.testSetsQLB = QListBox(self.testSetsQVGB)
        self.testSetsQLB.setSelectionMode(QListBox.Multi)
        self.connect(self.testSetsQLB, SIGNAL("selectionChanged()"), self.testSetsSelectionChange)
        self.unselectAllTestSetsQLB = QPushButton("(Un)select all", self.testSetsQVGB)
        self.connect(self.unselectAllTestSetsQLB, SIGNAL("clicked()"), self.SUAtestSetsQLB)

        self.performanceQVGB = QVGroupBox(self.space)
        self.performanceQVGB.setTitle("Performance line")
        self.showPerformanceAnalysisQCB = QCheckBox("Enable", self.performanceQVGB)
        self.connect(self.showPerformanceAnalysisQCB, SIGNAL("stateChanged(int)"), self.setShowPerformanceAnalysis)
        self.showPerformanceAnalysisQCB.setChecked(0)

        self.resize(800, 768)
        szs = self.splitQS.sizes()
        sum = 0
        for v in szs: sum += v
        self.splitQS.setSizes( [round(1.0/5.0*sum), round(2.0/5.0*sum), round(2.0/5.0*sum)] )

    def activateLoadedSettings(self):
        self.options.pointWidthSlider.setValue(self.PointWidth)
        self.options.pointWidthLCD.display(self.PointWidth)
        self.setPointWidth(self.PointWidth)
        #
        self.options.lineWidthSlider.setValue(self.CurveWidth)
        self.options.lineWidthLCD.display(self.CurveWidth)
        self.setCurveWidth(self.CurveWidth)
        #
        self.options.showDiagonalQCB.setChecked(self.ShowDiagonal)
        self.setShowDiagonal(self.ShowDiagonal)
        #
        self.options.hullWidthSlider.setValue(self.ConvexHullCurveWidth)
        self.options.hullWidthLCD.display(self.ConvexHullCurveWidth)
        self.setConvexHullCurveWidth(self.ConvexHullCurveWidth)
        #
        self.options.hullColor.setNamedColor(QString(self.HullColor))
        self.setHullColor(self.options.hullColor)

    def calcAllClassGraphs(self):
        cl = 0
        for g in self.graphs:
            g.setNumberOfClassifiersIterationsAndClassifierColors(self.dres.classifierNames, self.numberOfIterations, self.classifierColor)
            g.setTestSetData(self.dresSplitByIterations, cl)
            g.setShowConvexHull(self.convexhullQCB.isChecked())
            g.setShowPerformanceLine(self.showPerformanceAnalysisQCB.isChecked())

            ## user settings
            g.setPointWidth(self.PointWidth)
            g.setCurveWidth(self.CurveWidth)
            g.setShowDiagonal(self.ShowDiagonal)
            g.setConvexHullCurveWidth(self.ConvexHullCurveWidth)
            g.setHullColor(self.options.hullColor)

##          g.replot()
##            g.repaint()
            cl += 1

    def results(self, dres):
        self.dres = dres

        self.classQLB.clear()
        self.classifiersQLB.clear()
        self.testSetsQLB.clear()
        self.removeGraphs()

        self.defaultPerfLinePValues = []
        if self.dres <> None:
            ## classQLB
            self.numberOfClasses = len(self.dres.classValues)
            self.graphs = []
            for i in range(self.numberOfClasses):
                graph = singleClassLiftCurveGraph(self.mainArea, "", self.dres.classValues[i])
                self.graphs.append( graph )
            self.classSelectionChange()

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

            ## update graphics
            ## classQLB
            self.classQLB.insertStrList(self.dres.classValues)
            self.classQLB.selectAll(1)  ##or: if numberOfClasses > 0: self.classQLB.setSelected(0, 1)

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
        else:
            self.classifierColor = None
        self.setPerformanceLineBox()
        self.setDefaultPValues()

if __name__ == "__main__":
    a = QApplication(sys.argv)
    owdm = OWLiftCurve()
    a.setMainWidget(owdm)
    owdm.show()
    a.exec_loop()
    owdm.saveSettings()

