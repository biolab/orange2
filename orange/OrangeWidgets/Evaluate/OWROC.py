"""
<name>ROC Anaylsis</name>
<description>Displays Receiver Operating Characteristics curve based on evaluation of classifiers.</description>
<contact>Tomaz Curk</contact>
<icon>ROCAnalysis.png</icon>
<priority>1010</priority>
"""

from OWTools import *
from OWWidget import *
from OWGraph import *

import OWGUI

import orngStat, orngTest
import statc, math

def TCconvexHull(curves):
    ## merge curves into one
    mergedCurve = []
    for c in curves:
        mergedCurve.extend(c)
    mergedCurve.sort() ## increasing by fp, tp

    if len(mergedCurve) == 0: return []

    hull = []
    (prevX, maxY, fscore) = (mergedCurve[0] + (0.0,))[:3]
    prevPfscore = [fscore]
    px = prevX
    for p in mergedCurve[1:]:
        (px, py, fscore) = (p + (0.0,))[:3]
        if (px == prevX):
            if py > maxY:
                prevPfscore = [fscore]
                maxY = py
            elif py == maxY:
                prevPfscore.append(fscore)
        elif (px > prevX):
            hull = orngStat.ROCaddPoint((prevX, maxY, prevPfscore), hull, keepConcavities=0)
            prevX = px
            maxY = py
            prevPfscore = [fscore]
    hull = orngStat.ROCaddPoint((prevX, maxY, prevPfscore), hull, keepConcavities=0)

    return hull

class singleClassROCgraph(OWGraph):
    def __init__(self, parent = None, name = None, title = ""):
        OWGraph.__init__(self, parent, name)

        self.setYRlabels(None)
        self.enableGridXB(0)
        self.enableGridYL(0)
        self.setAxisMaxMajor(QwtPlot.xBottom, 10)
        self.setAxisMaxMinor(QwtPlot.xBottom, 5)
        self.setAxisMaxMajor(QwtPlot.yLeft, 10)
        self.setAxisMaxMinor(QwtPlot.yLeft, 5)
        self.setAxisScale(QwtPlot.xBottom, -0.0, 1.0, 0)
        self.setAxisScale(QwtPlot.yLeft, -0.0, 1.0, 0)
        self.setShowXaxisTitle(1)
        self.setXaxisTitle("FP Rate (1-Specificity)")
        self.setShowYLaxisTitle(1)
        self.setYLaxisTitle("TP Rate (Sensitivity)")
        self.setShowMainTitle(1)
        self.setMainTitle(title)
        self.targetClass = 0
        self.averagingMethod = None
        self.splitByIterations = None
        self.VTAsamples = 10 ## vertical threshold averaging, number of samples
        self.FPcost = 500.0
        self.FNcost = 500.0
        self.pvalue = 400.0 ##0.400

        self.performanceLineSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(Qt.color0), QPen(self.black), QSize(7,7))
        self.convexHullPen = QPen(Qt.yellow, 3)

        self.removeCurves()

    def computeCurve(self, res, classIndex=-1, keepConcavities=1):
        return orngStat.TCcomputeROC(res, classIndex, keepConcavities)

    def setNumberOfClassifiersIterationsAndClassifierColors(self, classifierNames, iterationsNum, classifierColor):
        classifiersNum = len(classifierNames)
        self.removeCurves()
        self.classifierColor = classifierColor
        self.classifierNames = classifierNames

        for cNum in range(classifiersNum):
            self.classifierIterationCKeys.append([])
            self.classifierIterationConvexCKeys.append([])
            self.classifierIterationROCdata.append([])
            for iNum in range(iterationsNum):
                ckey = self.insertCurve('')
                self.setCurvePen(ckey, QPen(self.classifierColor[cNum], 3))
                self.classifierIterationCKeys[cNum].append(ckey)
                ckey = self.insertCurve('')
                self.setCurvePen(ckey, QPen(self.classifierColor[cNum], 1))
                self.classifierIterationConvexCKeys[cNum].append(ckey)
                self.classifierIterationROCdata[cNum].append(None)

            self.showClassifiers.append(0)
            self.showIterations.append(0)

            ## 'merge' average curve keys
            ckey = self.insertCurve('')
            self.setCurvePen(ckey, QPen(self.classifierColor[cNum], 2))
            self.mergedCKeys.append(ckey)

            ckey = self.insertCurve('')
            self.setCurvePen(ckey, QPen(self.classifierColor[cNum], 1))
            self.mergedConvexCKeys.append(ckey)

            newSymbol = QwtSymbol(QwtSymbol.None, QBrush(Qt.color0), QPen(self.classifierColor[cNum], 2), QSize(0,0))
            ## 'vertical' average curve keys
            curve = errorBarQwtPlotCurve(self, '', connectPoints = 1, tickXw = 1.0/self.VTAsamples/5.0)
            ckey = self.insertCurve(curve)
            self.setCurveSymbol(ckey, newSymbol)
            self.setCurveStyle(ckey, QwtCurve.UserCurve)
            self.verticalCKeys.append(ckey)

            ## 'threshold' average curve keys
            curve = errorBarQwtPlotCurve(self, '', connectPoints = 1, tickXw = 1.0/self.VTAsamples/5.0, tickYw = 1.0/self.VTAsamples/5.0, showVerticalErrorBar = 1, showHorizontalErrorBar = 1)
            ckey = self.insertCurve(curve)
            self.setCurveSymbol(ckey, newSymbol)
            self.setCurveStyle(ckey, QwtCurve.UserCurve)
            self.thresholdCKeys.append(ckey)

        ## iso-performance line on top of all curves
        self.performanceLineCKey = self.insertCurve('')
        self.setCurvePen(self.performanceLineCKey, QPen(Qt.black, 2))
        self.setCurveSymbol(self.performanceLineCKey, self.performanceLineSymbol)

    def removeCurves(self):
        OWGraph.removeCurves(self)
        self.classifierColor = []
        self.classifierNames = []
        self.classifierIterationROCdata = []
        self.showClassifiers = []
        self.showIterations = []
        self.showConvexCurves = 0
        self.showConvexHull = 0
        self.showPerformanceLine = 0
        self.showDiagonal = 0

        ## 'merge' average curve keys
        self.mergedCKeys = []
        self.mergedConvexCKeys = []
        ## 'vertical' average curve keys
        self.verticalCKeys = []
        ## 'threshold' average curve keys
        self.thresholdCKeys = []
        ## 'None' average curve keys
        self.classifierIterationCKeys = []
        self.classifierIterationConvexCKeys = []

        ## convex hull calculation
        self.mergedConvexHullData = []
        self.verticalConvexHullData = []
        self.thresholdConvexHullData = []
        self.classifierConvexHullData = []
        self.hullCurveDataForPerfLine = [] ## for performance analysis

        ## diagonal curve
        self.diagonalCKey = self.insertCurve('')
        self.setCurvePen(self.diagonalCKey, QPen(Qt.black, 1))
        self.setCurveData(self.diagonalCKey, [0.0, 1.0], [0.0, 1.0])
        
        ## convex hull curve keys
        self.mergedConvexHullCKey = self.insertCurve('')
        self.setCurvePen(self.mergedConvexHullCKey, self.convexHullPen)
        self.verticalConvexHullCKey = self.insertCurve('')
        self.setCurvePen(self.verticalConvexHullCKey, self.convexHullPen)
        self.thresholdConvexHullCKey = self.insertCurve('')
        self.setCurvePen(self.thresholdConvexHullCKey, self.convexHullPen)
        self.classifierConvexHullCKey = self.insertCurve('')
        self.setCurvePen(self.classifierConvexHullCKey, self.convexHullPen)

        ## iso-performance line
        self.performanceLineCKey = -1

    def setIterationCurves(self, iteration, curves):
        classifier = 0
        for c in curves:
            x = [px for (px, py, pf) in c]
            y = [py for (px, py, pf) in c]
            ckey = self.classifierIterationCKeys[classifier][iteration]
            self.setCurveData(ckey, x, y)
            self.classifierIterationROCdata[classifier][iteration] = c
            classifier += 1

    def setIterationConvexCurves(self, iteration, curves):
        classifier = 0
        for c in curves:
            x = [px for (px, py, pf) in c]
            y = [py for (px, py, pf) in c]
            ckey = self.classifierIterationConvexCKeys[classifier][iteration]
            self.setCurveData(ckey, x, y)
            classifier += 1

    def setTestSetData(self, splitByIterations, targetClass):
        self.splitByIterations = splitByIterations
        ## generate the "base" unmodified ROC curves
        self.targetClass = targetClass
        iteration = 0
        for isplit in splitByIterations:
            # unmodified ROC curve
            curves = self.computeCurve(isplit, self.targetClass, 1)
            self.setIterationCurves(iteration, curves)

            # convex ROC curve
            curves = self.computeCurve(isplit, self.targetClass, 0)
            self.setIterationConvexCurves(iteration, curves)
            iteration += 1

    def updateCurveDisplay(self):
        self.curve(self.diagonalCKey).setEnabled(self.showDiagonal)

        showSomething = 0
        for cNum in range(len(self.showClassifiers)):
            showCNum = (self.showClassifiers[cNum] <> 0)

            ## 'merge' averaging
            b = (self.averagingMethod == 'merge') and showCNum
            showSomething = showSomething or b
            curve =  self.curve(self.mergedCKeys[cNum])
            if curve <> None: curve.setEnabled(b)

            b = b and self.showConvexCurves
            curve =  self.curve(self.mergedConvexCKeys[cNum])
            if curve <> None: curve.setEnabled(b)

            ## 'vertical' averaging
            b = (self.averagingMethod == 'vertical') and showCNum
            showSomething = showSomething or b
            curve =  self.curve(self.verticalCKeys[cNum])
            if curve <> None: curve.setEnabled(b)

            ## 'threshold' averaging
            b = (self.averagingMethod == 'threshold') and showCNum
            showSomething = showSomething or b
            curve =  self.curve(self.thresholdCKeys[cNum])
            if curve <> None: curve.setEnabled(b)

            ## 'None' averaging
            for iNum in range(len(self.showIterations)):
                b = (self.averagingMethod == None) and showCNum and (self.showIterations[iNum] <> 0)
                showSomething = showSomething or b
                self.curve(self.classifierIterationCKeys[cNum][iNum]).setEnabled(b)
                b = b and self.showConvexCurves
                self.curve(self.classifierIterationConvexCKeys[cNum][iNum]).setEnabled(b)

        chb = (showSomething) and (self.averagingMethod == None) and self.showConvexHull
        curve =  self.curve(self.classifierConvexHullCKey)
        if curve <> None: curve.setEnabled(chb)

        chb = (showSomething) and (self.averagingMethod == 'merge') and self.showConvexHull
        curve =  self.curve(self.mergedConvexHullCKey)
        if curve <> None: curve.setEnabled(chb)

        chb = (showSomething) and (self.averagingMethod == 'vertical') and self.showConvexHull
        curve =  self.curve(self.verticalConvexHullCKey)
        if curve <> None: curve.setEnabled(chb)

        chb = (showSomething) and (self.averagingMethod == 'threshold') and self.showConvexHull
        curve =  self.curve(self.thresholdConvexHullCKey)
        if curve <> None: curve.setEnabled(chb)

        ## performance line
        b = (self.averagingMethod == 'merge') and self.showPerformanceLine
        for mkey in self.markerKeys():
            self.marker(mkey).setEnabled(b)
        curve = self.curve(self.performanceLineCKey)
        if curve <> None: curve.setEnabled(b)

        self.updateLayout()
        self.update()

    def setShowConvexCurves(self, b):
        self.showConvexCurves = b
        self.updateCurveDisplay()

    def setShowConvexHull(self, b):
        self.showConvexHull = b
        self.updateCurveDisplay()

    def setShowPerformanceLine(self, b):
        self.showPerformanceLine = b
        self.updateCurveDisplay()

    def setShowClassifiers(self, list):
        self.showClassifiers = list
        self.calcConvexHulls()
        self.calcUpdatePerformanceLine() ## new data for performance line
        self.updateCurveDisplay()

    def setShowIterations(self, list):
        self.showIterations = list
        self.calcAverageCurves()
        self.calcConvexHulls()
        self.calcUpdatePerformanceLine() ## new data for performance line
        self.updateCurveDisplay()

    ## calculate the average curve for the selected test sets (with all the averaging methods)
    def calcAverageCurves(self):
        ##
        ## self.averagingMethod == 'merge':
        mergedIterations = orngTest.ExperimentResults(1, self.splitByIterations[0].classifierNames, self.splitByIterations[0].classValues, self.splitByIterations[0].weights, classifiers=self.splitByIterations[0].classifiers, loaded=self.splitByIterations[0].loaded)
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

        ## prepare a common input structure for vertical and threshold averaging
        ROCS = []
        somethingToShow = 0
        for c in self.classifierIterationROCdata:
            ROCS.append([])
            i = 0
            for s in self.showIterations:
                if s:
                    somethingToShow = 1
                    ROCS[-1].append(c[i])
                i += 1

        ## remove curve
        self.verticalConvexHullData = []
        self.thresholdConvexHullData = []

        if somethingToShow == 0:
            for ckey in self.verticalCKeys:
                self.setCurveData(ckey, [], [])

            for ckey in self.thresholdCKeys:
                self.setCurveData(ckey, [], [])
            return

        ##
        ## self.averagingMethod == 'vertical':
        ## calculated from the self.classifierIterationROCdata data
        (averageCurves, verticalErrorBarValues) = orngStat.TCverticalAverageROC(ROCS, self.VTAsamples)
        classifier = 0
        for c in averageCurves:
            self.verticalConvexHullData.append(c)
            xs = []
            mps = []
            for pcn in range(len(c)):
                (px, py) = c[pcn]
                ## for the error bar plot
                xs.append(px)
                mps.append(py + 0.0)

                xs.append(px)
                mps.append(py + verticalErrorBarValues[classifier][pcn])

                xs.append(px)
                mps.append(py - verticalErrorBarValues[classifier][pcn])

            ckey =  self.verticalCKeys[classifier]
            self.setCurveData(ckey, xs, mps)
            classifier += 1

        ##
        ## self.averagingMethod == 'threshold':
        ## calculated from the self.classifierIterationROCdata data
        (averageCurves, verticalErrorBarValues, horizontalErrorBarValues) = orngStat.TCthresholdlAverageROC(ROCS, self.VTAsamples)
        classifier = 0
        for c in averageCurves:
            self.thresholdConvexHullData.append(c)
            xs = []
            mps = []
            for pcn in range(len(c)):
                (px, py) = c[pcn]
                ## for the error bar plot
                xs.append(px + 0.0)
                mps.append(py + 0.0)

                xs.append(px - horizontalErrorBarValues[classifier][pcn])
                mps.append(py + verticalErrorBarValues[classifier][pcn])

                xs.append(px + horizontalErrorBarValues[classifier][pcn])
                mps.append(py - verticalErrorBarValues[classifier][pcn])

            ckey = self.thresholdCKeys[classifier]
            self.setCurveData(ckey, xs, mps)
            classifier += 1

        ## self.averagingMethod == 'None'
        ## already calculated

    def calcConvexHulls(self):
        ## self.classifierConvexHullCKey = -1
        hullData = []
        for cNum in range(len(self.showClassifiers)):
            for iNum in range(len(self.showIterations)):
                if (self.showClassifiers[cNum] <> 0) and (self.showIterations[iNum] <> 0):
                    hullData.append(self.classifierIterationROCdata[cNum][iNum])

        convexHullCurve = TCconvexHull(hullData)
        x = [px for (px, py, pf) in convexHullCurve]
        y = [py for (px, py, pf) in convexHullCurve]
        self.setCurveData(self.classifierConvexHullCKey, x, y)

        ## self.mergedConvexHullCKey = -1
        hullData = []
        for cNum in range(len(self.mergedConvexHullData)):
            if (self.showClassifiers[cNum] <> 0):
                ncurve = []
                for (px, py, pfscore) in self.mergedConvexHullData[cNum]:
                    ncurve.append( (px, py, (cNum, pfscore)) )
                hullData.append(ncurve)

        self.hullCurveDataForPerfLine = TCconvexHull(hullData) # keep data about curve for performance line drawing
        x = [px for (px, py, pf) in self.hullCurveDataForPerfLine]
        y = [py for (px, py, pf) in self.hullCurveDataForPerfLine]
        self.setCurveData(self.mergedConvexHullCKey, x, y)

        ## self.verticalConvexHullCKey = -1
        hullData = []
        for cNum in range(len(self.verticalConvexHullData)):
            if (self.showClassifiers[cNum] <> 0):
                hullData.append(self.verticalConvexHullData[cNum])

        convexHullCurve = TCconvexHull(hullData)
        x = [px for (px, py, pf) in convexHullCurve]
        y = [py for (px, py, pf) in convexHullCurve]
        self.setCurveData(self.verticalConvexHullCKey, x, y)

        ## self.thresholdConvexHullCKey = -1
        hullData = []
        for cNum in range(len(self.thresholdConvexHullData)):
            if (self.showClassifiers[cNum] <> 0):
                hullData.append(self.thresholdConvexHullData[cNum])

        convexHullCurve = TCconvexHull(hullData)
        x = [px for (px, py, pf) in convexHullCurve]
        y = [py for (px, py, pf) in convexHullCurve]
        self.setCurveData(self.thresholdConvexHullCKey, x, y)

    def setAveragingMethod(self, m):
        self.averagingMethod = m
        self.updateCurveDisplay()

    ## performance line
    def calcUpdatePerformanceLine(self):
    	closestpoints = orngStat.TCbestThresholdsOnROCcurve(self.FPcost, self.FNcost, self.pvalue, self.hullCurveDataForPerfLine)
    	m = (self.FPcost*(1.0 - self.pvalue)) / (self.FNcost*self.pvalue)

        ## now draw the closest line to the curve
        b = (self.averagingMethod == 'merge') and self.showPerformanceLine
        self.removeMarkers()
        lpx = []
        lpy = []
        first = 1
        for (x, y, fscorelist) in closestpoints:
            if first:
                first = 0
                lpx.append(x - 2.0)
                lpy.append(y - 2.0*m)
            lpx.append(x)
            lpy.append(y)
            px = x
            py = y
            for (cNum, threshold) in fscorelist:
                s = "%1.3f %s" % (threshold, self.classifierNames[cNum])
                py = py - 0.05
                if py < 0.05:
                    py = 0.05
                if py > 1.0:
                    py = 1.0
                if px < 0.0:
                    px = 0.0
                if px > 0.8:
                    px = 0.8
                mkey = self.insertMarker(s)
                self.marker(mkey).setXValue(px)
                self.marker(mkey).setYValue(py)
                self.marker(mkey).setLabelAlignment(Qt.AlignRight)
                self.marker(mkey).setEnabled(b)
        if len(closestpoints) > 0:
            lpx.append(x + 2.0)
            lpy.append(y + 2.0*m)

        self.setCurveData(self.performanceLineCKey, lpx, lpy)
        self.curve(self.performanceLineCKey).setEnabled(b)
        self.update()

    def costChanged(self, FPcost, FNcost):
        self.FPcost = float(FPcost)
        self.FNcost = float(FNcost)
        self.calcUpdatePerformanceLine()

    def pChanged(self, pvalue):
        self.pvalue = float(pvalue)
        self.calcUpdatePerformanceLine()

    def setPointWidth(self, v):
        self.performanceLineSymbol.setSize(v, v)
        self.setCurveSymbol(self.performanceLineCKey, self.performanceLineSymbol)
        self.update()

    def setCurveWidth(self, v):
        for cNum in range(len(self.showClassifiers)):
            self.setCurvePen(self.mergedCKeys[cNum], QPen(self.classifierColor[cNum], v))
            self.setCurvePen(self.verticalCKeys[cNum], QPen(self.classifierColor[cNum], v))
            self.setCurvePen(self.thresholdCKeys[cNum], QPen(self.classifierColor[cNum], v))
            for iNum in range(len(self.showIterations)):
                self.setCurvePen(self.classifierIterationCKeys[cNum][iNum], QPen(self.classifierColor[cNum], v))
        self.update()

    def setConvexCurveWidth(self, v):
        for cNum in range(len(self.showClassifiers)):
            self.setCurvePen(self.mergedConvexCKeys[cNum], QPen(self.classifierColor[cNum], v))
            for iNum in range(len(self.showIterations)):
                self.setCurvePen(self.classifierIterationConvexCKeys[cNum][iNum], QPen(self.classifierColor[cNum], v))
        self.update()

    def setShowDiagonal(self, v):
        self.showDiagonal = v
        self.updateCurveDisplay()

    def setConvexHullCurveWidth(self, v):
        self.convexHullPen.setWidth(v)
        self.setCurvePen(self.mergedConvexHullCKey, self.convexHullPen)
        self.setCurvePen(self.verticalConvexHullCKey, self.convexHullPen)
        self.setCurvePen(self.thresholdConvexHullCKey, self.convexHullPen)
        self.setCurvePen(self.classifierConvexHullCKey, self.convexHullPen)
        self.update()

    def setHullColor(self, c):
        self.convexHullPen.setColor(c)
        self.setCurvePen(self.mergedConvexHullCKey, self.convexHullPen)
        self.setCurvePen(self.verticalConvexHullCKey, self.convexHullPen)
        self.setCurvePen(self.thresholdConvexHullCKey, self.convexHullPen)
        self.setCurvePen(self.classifierConvexHullCKey, self.convexHullPen)
        self.update()

    def sizeHint(self):
        return QSize(100, 100)

class OWROC(OWWidget):
    settingsList = ["PointWidth", "CurveWidth", "ConvexCurveWidth", "ShowDiagonal",
                    "ConvexHullCurveWidth", "HullColor", "AveragingMethodIndex",
                    "ShowConvexHull", "ShowConvexCurves", "EnablePerformance"]
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "ROC Analysis", 1)

        # inputs
        self.inputs=[("Evaluation Results", orngTest.ExperimentResults, self.test_results, Default)]

        # default settings
        self.PointWidth = 7
        self.CurveWidth = 3
        self.ConvexCurveWidth = 1
        self.ShowDiagonal = TRUE
        self.ConvexHullCurveWidth = 3
        self.HullColor = str(Qt.yellow.name())
        self.AveragingMethodIndex = 0 ##'merge'
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

        self.AveragingMethodNames = ['merge', 'vertical', 'threshold', None]
        self.AveragingMethod = self.AveragingMethodNames[self.AveragingMethodIndex]

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

        # show convex ROC curves and show ROC convex hull
        self.convexCurvesQCB = OWGUI.checkBox(self.generalTab, self, 'ShowConvexCurves', 'Show Convex ROC Rurves', tooltip='', callback=self.setShowConvexCurves)
        OWGUI.checkBox(self.generalTab, self, 'ShowConvexHull', 'Show ROC Convex Hull', tooltip='', callback=self.setShowConvexHull)
        self.tabs.insertTab(self.generalTab, "General")
        

        # performance analysis
        self.performanceTab = QVGroupBox(self)
        self.performanceTabCosts = QVGroupBox(self.performanceTab)
        OWGUI.checkBox(self.performanceTabCosts, self, 'EnablePerformance', 'Show Performance Line', tooltip='', callback=self.setShowPerformanceAnalysis)

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
        OWGUI.radioButtonsInBox(self.settingsTab, self, 'AveragingMethodIndex', ['Merge (expected ROC perf.)', 'Vertical', 'Threshold', 'None'], box='Averaging ROC curves', callback=self.selectAveragingMethod)
        OWGUI.hSlider(self.settingsTab, self, 'PointWidth', box='Point Width', minValue=3, maxValue=9, step=1, callback=self.setPointWidth, ticks=1)
        OWGUI.hSlider(self.settingsTab, self, 'CurveWidth', box='ROC Curve Width', minValue=1, maxValue=5, step=1, callback=self.setCurveWidth, ticks=1)
        OWGUI.hSlider(self.settingsTab, self, 'ConvexCurveWidth', box='ROC Convex Curve Width', minValue=1, maxValue=5, step=1, callback=self.setConvexCurveWidth, ticks=1)
        OWGUI.hSlider(self.settingsTab, self, 'ConvexHullCurveWidth', box='ROC Convex Hull', minValue=2, maxValue=9, step=1, callback=self.setConvexHullCurveWidth, ticks=1)
        OWGUI.checkBox(self.settingsTab, self, 'ShowDiagonal', 'Show Diagonal ROC Line', tooltip='', callback=self.setShowDiagonal)
        self.tabs.insertTab(self.settingsTab, "Settings")
      
        self.resize(800, 600)

    def saveToFile(self):
        for g in self.graphs:
            if g.isVisible():
                g.saveToFile()

    def setPointWidth(self):
        for g in self.graphs:
            g.setPointWidth(self.PointWidth)

    def setCurveWidth(self):
        for g in self.graphs:
            g.setCurveWidth(self.CurveWidth)

    def setConvexCurveWidth(self):
        for g in self.graphs:
            g.setConvexCurveWidth(self.ConvexCurveWidth)

    def setShowDiagonal(self):
        for g in self.graphs:
            g.setShowDiagonal(self.ShowDiagonal)

    def setConvexHullCurveWidth(self):
        for g in self.graphs:
            g.setConvexHullCurveWidth(self.ConvexHullCurveWidth)

    def setHullColor(self):
        self.HullColor = str(c.name())
        for g in self.graphs:
            g.setHullColor(self.HullColor)

    ##
    def selectUnselectAll(self, qlb):
        selected = 0
        for i in range(qlb.count()):
            if qlb.isSelected(i):
                selected = 1
                break
        qlb.selectAll(not(selected))

    def SUAclassifiersQLB(self):
        self.selectUnselectAll(self.classifiersQLB)

    def SUAtestSetsQLB(self):
        self.selectUnselectAll(self.testSetsQLB)
    ##

    def selectAveragingMethod(self):
        self.AveragingMethod = self.AveragingMethodNames[self.AveragingMethodIndex]
        if self.AveragingMethod == 'merge':
            self.performanceTabCosts.setEnabled(self.EnablePerformance)
        elif self.AveragingMethod == 'vertical':
            self.performanceTabCosts.setEnabled(0)
        elif self.AveragingMethod == 'threshold':
            self.performanceTabCosts.setEnabled(0)
        else:
            self.performanceTabCosts.setEnabled(0)

        self.convexCurvesQCB.setEnabled(self.AveragingMethod == 'merge' or self.AveragingMethod == None)
        self.performanceTabCosts.setEnabled(self.AveragingMethod == 'merge')

        for g in self.graphs:
            g.setAveragingMethod(self.AveragingMethod)

    ## class selection (classQLB)
    def target(self):   
        for i in range(len(self.graphs)):
            self.graphs[i].hide()

        if (self.targetClass <> None) and (len(self.graphs) > 0):
            if self.targetClass >= len(self.graphs):
                self.targetClass = len(self.graphs) - 1
            if self.targetClass < 0:
                self.targetClass = 0
            self.graphsGridLayoutQGL.addWidget(self.graphs[self.targetClass], 0, 0)
            self.graphs[self.targetClass].show()

            self.FPcost = self.FPcostList[self.targetClass]
            self.FNcost = self.FNcostList[self.targetClass]
            self.pvalue = self.pvalueList[self.targetClass]
    ##

    ## classifiers selection (classifiersQLB)
    def classifiersSelectionChange(self):
        list = []
        for i in range(self.classifiersQLB.count()):
            if self.classifiersQLB.isSelected(i):
                list.append( 1 )
            else:
                list.append( 0 )
        for g in self.graphs:
            g.setShowClassifiers(list)

    def setShowConvexCurves(self):
        for g in self.graphs:
            g.setShowConvexCurves(self.ShowConvexCurves)

    def setShowConvexHull(self):
        for g in self.graphs:
            g.setShowConvexHull(self.ShowConvexHull)
    ##

    def setShowPerformanceAnalysis(self):
        for g in self.graphs:
            g.setShowPerformanceLine(self.EnablePerformance)

    ## test set selection (testSetsQLB)
    def testSetsSelectionChange(self):
        list = []
        for i in range(self.testSetsQLB.count()):
            if self.testSetsQLB.isSelected(i):
                list.append( 1 )
            else:
                list.append( 0 )
        for g in self.graphs:
            g.setShowIterations(list)
    ##

    def calcAllClassGraphs(self):
        for (cl, g) in enumerate(self.graphs):
            g.setNumberOfClassifiersIterationsAndClassifierColors(self.dres.classifierNames, self.numberOfIterations, self.classifierColor)
            g.setTestSetData(self.dresSplitByIterations, cl)
            g.setShowConvexCurves(self.ShowConvexCurves)
            g.setShowConvexHull(self.ShowConvexHull)
            g.setAveragingMethod(self.AveragingMethod)
            g.setShowPerformanceLine(self.EnablePerformance)

            ## user settings
            g.setPointWidth(self.PointWidth)
            g.setCurveWidth(self.CurveWidth)
            g.setConvexCurveWidth(self.ConvexCurveWidth)
            g.setShowDiagonal(self.ShowDiagonal)
            g.setConvexHullCurveWidth(self.ConvexHullCurveWidth)
            g.setHullColor(QColor(self.HullColor))

    def removeGraphs(self):
        for g in self.graphs:
            g.removeCurves()
            g.hide()

    def costsChanged(self):
        if self.targetClass <> None and (len(self.graphs) > 0):
            self.FPcostList[self.targetClass] = self.FPcost
            self.FNcostList[self.targetClass] = self.FNcost
            self.graphs[self.targetClass].costChanged(self.FPcost, self.FNcost)

    def pvaluesUpdated(self):
        if (self.targetClass == None) or (len(self.graphs) == 0): return

        ## update p values
        if self.pvalue > self.maxpsum - (len(self.pvalueList) - 1):
            self.pvalue = self.maxpsum - (len(self.pvalueList) - 1)

        self.pvalueList[self.targetClass] = self.pvalue ## set new value
        sum = int(statc.sum(self.pvalueList))
        ## adjust for big changes
        distrib = []
        for vi in range(len(self.pvalueList)):
            if vi == self.targetClass:
                distrib.append(0.0)
            else:
                distrib.append(self.pvalueList[vi] / float(sum - self.pvalue))

        dif = self.maxpsum - sum
        for vi in range(len(distrib)):
            self.pvalueList[vi] += int(float(dif) * distrib[vi])
            if self.pvalueList[vi] < self.minp:
                self.pvalueList[vi] = self.minp
            if self.pvalueList[vi] > self.maxp:
                self.pvalueList[vi] = self.maxp

        ## small changes
        dif = self.maxpsum - int(statc.sum(self.pvalueList))
        while abs(dif) > 0:
            if dif > 0: vi = self.pvalueList.index(min(self.pvalueList[:self.targetClass] + [self.maxp + 1] + self.pvalueList[self.targetClass+1:]))
            else: vi = self.pvalueList.index(max(self.pvalueList[:self.targetClass] + [self.minp - 1] + self.pvalueList[self.targetClass+1:]))

            if dif > 0: self.pvalueList[vi] += 1
            elif dif < 0: self.pvalueList[vi] -= 1

            if self.pvalueList[vi] < self.minp: self.pvalueList[vi] = self.minp
            if self.pvalueList[vi] > self.maxp: self.pvalueList[vi] = self.maxp
            dif = self.maxpsum - int(statc.sum(self.pvalueList))

        ## apply new pvalues
        for (index, graph) in enumerate(self.graphs):
            graph.pChanged(float(self.pvalueList[index]) / float(self.maxp))

    def setDefaultPValues(self):
        self.pvaluesList = [v for v in self.defaultPerfLinePValues]
        self.pvalue = self.pvaluesList[self.targetClass]
        self.pvaluesUpdated()

    def test_results(self, dres):
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
                graph = singleClassROCgraph(self.mainArea, "", "Predicted Class: " + self.dres.classValues[i])
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
            self.dresSplitByIterations = orngStat.splitByIterations(self.dres)
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
        self.performanceTabCosts.setEnabled(self.AveragingMethod == 'merge')
        self.setDefaultPValues()

if __name__ == "__main__":
    a = QApplication(sys.argv)
    owdm = OWROC()
    a.setMainWidget(owdm)
    owdm.show()
    a.exec_loop()
