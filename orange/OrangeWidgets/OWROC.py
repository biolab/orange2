"""
<name>ROC anaylsis</name>
<description>None.</description>
<category>Evaluation</category>
<icon>icons\ROCAnalysis.png</icon>
<priority>1010</priority>
"""

from OData import *
from OWTools import *
from OWWidget import *
from OWGraph import *
from OWGUI import *
from OWROCOptions import *

import orngEval
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
            hull = orngEval.ROCaddPoint((prevX, maxY, prevPfscore), hull, keepConcavities=0)
            prevX = px
            maxY = py
            prevPfscore = [fscore]
    hull = orngEval.ROCaddPoint((prevX, maxY, prevPfscore), hull, keepConcavities=0)

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
        return orngEval.TCcomputeROC(res, classIndex, keepConcavities)

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
        (averageCurves, verticalErrorBarValues) = orngEval.TCverticalAverageROC(ROCS, self.VTAsamples)
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
        (averageCurves, verticalErrorBarValues, horizontalErrorBarValues) = orngEval.TCthresholdlAverageROC(ROCS, self.VTAsamples)
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
        m = (self.FPcost*(1.0 - self.pvalue)) / (self.FNcost*self.pvalue)

        ## put the iso-performance line in point (0.0, 1.0)
        x0, y0 = (0.0, 1.0)
        x1, y1 = (1.0, 1.0 + m)
        d01 = math.sqrt((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0))

        ## calculate and find the closest point to the line
        firstp = 1
        mind = 0.0
        a = (x0*y1 - x1*y0)
        closestpoints = []
        for (x, y, fscorelist) in self.hullCurveDataForPerfLine:
            d = ((y0 - y1)*x + (x1 - x0)*y + a) / d01
            d = abs(d)
            if firstp or d < mind:
                mind, firstp = d, 0
                closestpoints = [(x, y, fscorelist)]
            else:
                if abs(d - mind) <= 0.0001: ## close enough
                    closestpoints.append( (x, y, fscorelist) )

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
        return QSize(170, 170)

class BalancedSpinBoxCallback:
    def __init__(self, widget, index, minp, maxp, maxsum):
        self.parentwidget = widget
        self.index = index
        self.minp = minp
        self.maxp = maxp
        self.maxsum = maxsum
        widget.callbackDeposit.append(self)

    def balanceNewValue(self, values, value):
        if value > self.maxsum - (len(values) - 1):
            value = self.maxsum - (len(values) - 1)

        values[self.index] = value ## set new value
        sum = int(statc.sum(values))

        ## adjust for big changes
        distrib = []
        for vi in range(len(values)):
            if vi == self.index:
                distrib.append(0.0)
            else:
                distrib.append(values[vi] / float(sum - value))

        dif = self.maxsum - sum
        for vi in range(len(distrib)):
            values[vi] += int(float(dif) * distrib[vi])
            if values[vi] < self.minp:
                values[vi] = self.minp
            if values[vi] > self.maxp:
                values[vi] = self.maxp

        ## small changes
        dif = self.maxsum - int(statc.sum(values))
        while abs(dif) > 0:
            if dif > 0: vi = values.index(min(values[:self.index] + [self.maxp + 1] + values[self.index+1:]))
            else: vi = values.index(max(values[:self.index] + [self.minp - 1] + values[self.index+1:]))

            if dif > 0: values[vi] += 1
            elif dif < 0: values[vi] -= 1

            if values[vi] < self.minp: values[vi] = self.minp
            if values[vi] > self.maxp: values[vi] = self.maxp
            dif = self.maxsum - int(statc.sum(values))

        return values

    def __call__(self, value):
        if not(self.parentwidget.updatingpValues):
            setattr(self.parentwidget, "pvalues", self.balanceNewValue(self.parentwidget.pvalues, value))
            self.parentwidget.pvaluesUpdated()

class CostChange:
    def __init__(self, widget, index):
        self.parentwidget = widget
        self.index = index
        widget.callbackDeposit.append(self)

    def __call__(self, value):
        self.parentwidget.costsChanged(self.index)

class OWROC(OWWidget):
    settingsList = ["PointWidth", "CurveWidth", "ConvexCurveWidth", "ShowDiagonal",
                    "ConvexHullCurveWidth", "HullColor"]
    def __init__(self,parent=None):
        "Constructor"
        OWWidget.__init__(self,
        parent,
        "&ROC",
        """None.
        """,
        TRUE,
        TRUE)

        #set default settings
        self.PointWidth = 7
        self.CurveWidth = 3
        self.ConvexCurveWidth = 1
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
        ## save each ROC graph in separate file
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
        self.averagingMethod = 'merge'
        self.graphs = []
        self.maxp = 1000
        self.defaultPerfLinePValues = []

        self.options = OWROCOptions()
        self.setOptions()

        #connect settingsbutton to show options
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)

        #connect GUI controls of options in options dialog to settings
        self.connect(self.options.pointWidthSlider, SIGNAL("valueChanged(int)"), self.setPointWidth)
        self.connect(self.options.lineWidthSlider, SIGNAL("valueChanged(int)"), self.setCurveWidth)
        self.connect(self.options.convexWidthSlider, SIGNAL("valueChanged(int)"), self.setConvexCurveWidth)
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

        # show convex ROC curves
        self.convexCurvesQCB = QCheckBox("convex ROC curves", self.classifiersQVGB) ## !!! only in None and Merge average mode
        self.connect(self.convexCurvesQCB, SIGNAL("stateChanged(int)"), self.setShowConvexCurves)

        # show ROC convex hull
        self.convexhullQCB = QCheckBox("ROC convex hull", self.classifiersQVGB)
        self.connect(self.convexhullQCB, SIGNAL("stateChanged(int)"), self.setShowConvexHull)

        ## test set selection (testSetsQLB)
        self.testSetsQVGB = QVGroupBox(self.splitQS)
        self.testSetsQVGB.setTitle("Test sets")
        self.testSetsQLB = QListBox(self.testSetsQVGB)
        self.testSetsQLB.setSelectionMode(QListBox.Multi)
        self.connect(self.testSetsQLB, SIGNAL("selectionChanged()"), self.testSetsSelectionChange)
        self.unselectAllTestSetsQLB = QPushButton("(Un)select all", self.testSetsQVGB)
        self.connect(self.unselectAllTestSetsQLB, SIGNAL("clicked()"), self.SUAtestSetsQLB)

        self.averagingQBG = QButtonGroup(4, Qt.Vertical, "Averaging ROC curves", self.testSetsQVGB)
        # merge averaging of selected ROC curves
        self.mergeAverageQRB = QRadioButton("Merge (average expected ROC performance)", self.averagingQBG)
        self.mergeAverageQRB.setChecked(1)
        # vertical averaging of selected ROC curves
        self.verticalAverageQRB = QRadioButton("Vertical", self.averagingQBG)
        # threshold averaging of selected ROC curves
        self.thresholdAverageQRB = QRadioButton("Threshold", self.averagingQBG)
        # none
        self.noAverageQRB = QRadioButton("None", self.averagingQBG)
        self.connect(self.averagingQBG, SIGNAL("clicked(int)"), self.selectAveragingMethod)

        self.performanceQVGB = QVGroupBox(self.space)
        self.performanceQVGB.setTitle("Performance line (only in Merge averaging)")
        self.showPerformanceAnalysisQCB = QCheckBox("Enable", self.performanceQVGB)
        self.connect(self.showPerformanceAnalysisQCB, SIGNAL("stateChanged(int)"), self.setShowPerformanceAnalysis)
        self.showPerformanceAnalysisQCB.setChecked(0)

        self.resize(800, 768)
        szs = self.splitQS.sizes()
        sum = 0
        for v in szs: sum += v
        self.splitQS.setSizes( [round(1.0/5.0*sum), round(2.0/5.0*sum), round(2.0/5.0*sum)] )

    def saveToFile(self):
        qfileName = QFileDialog.getSaveFileName("graph.png","Portable Network Graphics (.PNG)\nWindows Bitmap (.BMP)\nGraphics Interchange Format (.GIF)", None, "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        ext = ext.upper()
        cl = 0
        for g in self.graphs:
            if g.isVisible():
                clfname = fil + "_" + str(cl) + "." + ext
                g.saveToFileDirect(clfname, ext)
            cl += 1

    def setPointWidth(self, v):
        self.PointWidth = v
        for g in self.graphs:
            g.setPointWidth(v)

    def setCurveWidth(self, v):
        self.CurveWidth = v
        for g in self.graphs:
            g.setCurveWidth(v)

    def setConvexCurveWidth(self, v):
        self.ConvexCurveWidth = v
        for g in self.graphs:
            g.setConvexCurveWidth(v)

    def setShowDiagonal(self, v):
        self.ShowDiagonal = v
        for g in self.graphs:
            g.setShowDiagonal(v)

    def setConvexHullCurveWidth(self, v):
        self.ConvexHullCurveWidth = v
        for g in self.graphs:
            g.setConvexHullCurveWidth(v)

    def setHullColor(self, c):
        self.HullColor = str(c.name())
        for g in self.graphs:
            g.setHullColor(c)

    def setOptions(self):
        self.options.pointWidthSlider.setValue(self.PointWidth)
        self.options.pointWidthLCD.display(self.PointWidth)
        self.setPointWidth(self.PointWidth)
        #
        self.options.lineWidthSlider.setValue(self.CurveWidth)
        self.options.lineWidthLCD.display(self.CurveWidth)
        self.setCurveWidth(self.CurveWidth)
        #
        self.options.convexWidthSlider.setValue(self.ConvexCurveWidth)
        self.options.convexWidthLCD.display(self.ConvexCurveWidth)
        self.setConvexCurveWidth(self.ConvexCurveWidth)
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

    ##
    def selectUnselectAll(self, qlb):
        selected = 0
        for i in range(qlb.count()):
            if qlb.isSelected(i):
                selected = 1
                break
        qlb.selectAll(not(selected))

    def SUAclassQLB(self):
        self.selectUnselectAll(self.classQLB)

    def SUAclassifiersQLB(self):
        self.selectUnselectAll(self.classifiersQLB)

    def SUAtestSetsQLB(self):
        self.selectUnselectAll(self.testSetsQLB)
    ##

    def selectAveragingMethod(self, id):
        if id == 0: ##self.mergeAverageQRB.is = QRadioButton("Merge (average expected ROC performance)", self.averagingQBG)
            self.averagingMethod = 'merge'
            self.missClassificationCostQVB.show()
        elif id == 1: ##self.verticalAverageQRB = QRadioButton("Vertical", self.averagingQBG)
            self.averagingMethod = 'vertical'
            self.missClassificationCostQVB.hide()
        elif id == 2: ##self.thresholdAverageQRB = QRadioButton("Threshold", self.averagingQBG)
            self.averagingMethod = 'threshold'
            self.missClassificationCostQVB.hide()
        else: ##self.noAverageQRB = QRadioButton("None", self.averagingQBG)
            self.averagingMethod = None
            self.missClassificationCostQVB.hide()

        self.convexCurvesQCB.setEnabled(self.averagingMethod == 'merge' or self.averagingMethod == None)
        self.missClassificationCostQVB.setEnabled(self.averagingMethod == 'merge')
        self.showPerformanceAnalysisQCB.setEnabled(self.averagingMethod == 'merge')

        for g in self.graphs:
            g.setAveragingMethod(self.averagingMethod)

    ## class selection (classQLB)
    def classSelectionChange(self):
        numOfClasseVisible = 0
        for i in range(self.classQLB.numRows()):
            if self.classQLB.isSelected(i):
                numOfClasseVisible += 1

        maxCol = int(round(math.sqrt(numOfClasseVisible)))
        if maxCol == 0: maxCol = 1
        maxRow = int(round(float(numOfClasseVisible)/ maxCol)) + 1
        self.graphsGridLayoutQGL.expand(maxRow, maxCol)
        curRow = 0
        curCol = 0
        for i in range(self.classQLB.numRows()):
            if self.classQLB.isSelected(i):
                self.graphsGridLayoutQGL.addWidget(self.graphs[i], curRow, curCol)
                self.graphs[i].show()
                curCol += 1
                if curCol == maxCol:
                    curCol = 0
                    curRow += 1
            else:
                self.graphs[i].hide()
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

    def setShowConvexCurves(self, v):
        for g in self.graphs:
            g.setShowConvexCurves(v)

    def setShowConvexHull(self, v):
        for g in self.graphs:
            g.setShowConvexHull(v)
    ##

    def setShowPerformanceAnalysis(self, b):
        if b:
            self.missClassificationCostQVB.show()
        else:
            self.missClassificationCostQVB.hide()
        for g in self.graphs:
            g.setShowPerformanceLine(b)

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
        cl = 0
        for g in self.graphs:
            g.setNumberOfClassifiersIterationsAndClassifierColors(self.dres.classifierNames, self.numberOfIterations, self.classifierColor)
            g.setTestSetData(self.dresSplitByIterations, cl)
            g.setShowConvexCurves(self.convexCurvesQCB.isChecked())
            g.setShowConvexHull(self.convexhullQCB.isChecked())
            g.setAveragingMethod(self.averagingMethod)
            g.setShowPerformanceLine(self.showPerformanceAnalysisQCB.isChecked())

            ## user settings
            g.setPointWidth(self.PointWidth)
            g.setCurveWidth(self.CurveWidth)
            g.setConvexCurveWidth(self.ConvexCurveWidth)
            g.setShowDiagonal(self.ShowDiagonal)
            g.setConvexHullCurveWidth(self.ConvexHullCurveWidth)
            g.setHullColor(self.options.hullColor)

##          g.replot()
##            g.repaint()
            cl += 1

    def removeGraphs(self):
        for g in self.graphs:
            g.removeCurves()
            g.hide()

    def costsChanged(self, index):
        self.graphs[index].costChanged(self.FPcostQSpinBoxes[index].value(), self.FNcostQSpinBoxes[index].value())

    def pvaluesUpdated(self):
        self.updatingpValues = 1
        index = 0
        for qsb in self.pvaluesQSpinBoxes:
            qsb.setValue(self.pvalues[index])
            self.graphs[index].pChanged(float(self.pvalues[index]) / float(self.maxp))
            index += 1
        self.updatingpValues = 0

    def setDefaultPValues(self):
        self.pvalues = [v for v in self.defaultPerfLinePValues]
        self.pvaluesUpdated()

    def setPerformanceLineBox(self):
        list = self.missClassificationCostQVB.children()
        for c in list[1:]: ## first is layout
            self.missClassificationCostQVB.removeChild(c)
            c.close(1)

        show = 0
        if self.missClassificationCostQVB.isVisible():
            show = 1
            self.missClassificationCostQVB.hide()

        ## FP and FN cost ranges
        mincost = 1
        maxcost = 1000
        stepcost = 5
        ## prior prob. range
        self.callbackDeposit = []
        self.pvalues = []
        self.pvaluesQSpinBoxes = []
        self.FPcostQSpinBoxes = []
        self.FNcostQSpinBoxes = []
        self.updatingpValues = 0
        maxpsum = 1000
        minp = 1
        self.maxp = maxpsum - minp ## need it also in self.pvaluesUpdated
        stepp = 3
        index = 0

        self.pvalues = []
        for c in self.dres.classValues:
            self.pvalues.append(1)

        wa = QPushButton("Default p(cl)", self.missClassificationCostQVB)
        self.connect(wa, SIGNAL("clicked()"), self.setDefaultPValues)

        for c in self.dres.classValues:
            tmpclQVGB = QVGroupBox(self.missClassificationCostQVB)
            tmpclQVGB.setTitle("cl: " + str(c))

            hb = QHBox(tmpclQVGB)
            QLabel('FP cost:', hb)
            wa = QSpinBox(mincost, maxcost, stepcost, hb)
            self.FPcostQSpinBoxes.append(wa)
            self.connect(wa, SIGNAL("valueChanged(int)"), CostChange(self, index))

            hb = QHBox(tmpclQVGB)
            QLabel('FN cost:', hb)
            wa = QSpinBox(mincost, maxcost, stepcost, hb)
            self.FNcostQSpinBoxes.append(wa)
            self.connect(wa, SIGNAL("valueChanged(int)"), CostChange(self, index))

            hb = QHBox(tmpclQVGB)
            QLabel('p(cl) [1/1000]:', hb)
            wa = QSpinBox(minp, self.maxp, stepp, hb)
            self.pvaluesQSpinBoxes.append(wa)

            self.connect(wa, SIGNAL("valueChanged(int)"), BalancedSpinBoxCallback(self, index, minp, self.maxp, maxpsum))

            index += 1

        if show: self.missClassificationCostQVB.show()

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
                graph = singleClassROCgraph(self.mainArea, "", self.dres.classValues[i])
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
    owdm = OWROC()
    a.setMainWidget(owdm)
    owdm.show()
    a.exec_loop()
    owdm.saveSettings()

