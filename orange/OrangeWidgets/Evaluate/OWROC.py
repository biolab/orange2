"""
<name>ROC Analysis</name>
<description>Displays Receiver Operating Characteristics curve based on evaluation of classifiers.</description>
<contact>Tomaz Curk</contact>
<icon>icons/ROCAnalysis.png</icon>
<priority>1010</priority>
"""
from OWColorPalette import ColorPixmap
from OWWidget import *
from OWGraph import *
import OWGUI
import orngStat, orngTest
import statc, math
import time
import warnings

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
        self.setMainTitle(title)
        self.setShowMainTitle(1)
        self.targetClass = 0
        self.averagingMethod = None
        self.splitByIterations = None
        self.VTAsamples = 10 ## vertical threshold averaging, number of samples
        self.FPcost = 500.0
        self.FNcost = 500.0
        self.pvalue = 400.0 ##0.400

        self.performanceLineSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(Qt.color0), QPen(Qt.black), QSize(7,7))
        self.defaultLineSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(Qt.black), QPen(Qt.black), QSize(8,8))
        self.convexHullPen = QPen(Qt.yellow, 3)

        self.removeMarkers()
        self.performanceMarkerKeys = []

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
                curve = self.addCurve('', pen = QPen(self.classifierColor[cNum], 3), style=QwtPlotCurve.Lines)
                self.classifierIterationCKeys[cNum].append(curve)
                curve = self.addCurve('', pen = QPen(self.classifierColor[cNum], 1), style=QwtPlotCurve.Lines)
                self.classifierIterationConvexCKeys[cNum].append(curve)
                self.classifierIterationROCdata[cNum].append(None)

            self.showClassifiers.append(0)
            self.showIterations.append(0)

            ## 'merge' average curve keys
            curve = self.addCurve('', pen = QPen(self.classifierColor[cNum], 2), style=QwtPlotCurve.Lines)
            self.mergedCKeys.append(curve)

            curve = self.addCurve('', pen = QPen(Qt.black, 5), style=QwtPlotCurve.Lines)
            curve.setSymbol(self.defaultLineSymbol)
            
            self.mergedCThresholdKeys.append(curve)
            self.mergedCThresholdMarkers.append([])

            curve = self.addCurve('', pen = QPen(self.classifierColor[cNum], 1), style=QwtPlotCurve.Lines)
##            self.mergedConvexCKeys.append(QPen(Qt.black, 5))
            self.mergedConvexCKeys.append(curve)

            newSymbol = QwtSymbol(QwtSymbol.NoSymbol, QBrush(Qt.color0), QPen(self.classifierColor[cNum], 2), QSize(0,0))
            ## 'vertical' average curve keys
            curve = errorBarQwtPlotCurve('', connectPoints = 1, tickXw = 1.0/self.VTAsamples/5.0)
            curve.attach(self)
            curve.setSymbol(newSymbol)
            curve.setStyle(QwtPlotCurve.UserCurve)
            self.verticalCKeys.append(curve)

            ## 'threshold' average curve keys
            curve = errorBarQwtPlotCurve('', connectPoints = 1, tickXw = 1.0/self.VTAsamples/5.0, tickYw = 1.0/self.VTAsamples/5.0, showVerticalErrorBar = 1, showHorizontalErrorBar = 1)
            curve.attach(self)
            curve.setSymbol(newSymbol)
            curve.setStyle(QwtPlotCurve.UserCurve)
            self.thresholdCKeys.append(curve)

        ## iso-performance line on top of all curves
        self.performanceLineCKey = self.addCurve('', pen = QPen(Qt.black, 2), style=QwtPlotCurve.Lines)
        self.performanceLineCKey.setSymbol(self.performanceLineSymbol)

    def removeCurves(self):
        self.clear()
        self.classifierColor = []
        self.classifierNames = []
        self.classifierIterationROCdata = []
        self.showClassifiers = []
        self.showIterations = []
        self.showConvexCurves = 0
        self.showConvexHull = 0
        self.showPerformanceLine = 0
        self.showDefaultThresholdPoint = 0
        self.showDiagonal = 0

        ## 'merge' average curve keys
        self.mergedCKeys = []
        self.mergedCThresholdKeys = []
        self.mergedCThresholdMarkers = []
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
        self.diagonalCKey = self.addCurve('', pen = QPen(Qt.black, 1), symbol = QwtSymbol.NoSymbol, xData = [0.0, 1.0], yData = [0.0, 1.0], style=QwtPlotCurve.Lines)

        ## convex hull curve keys
        self.mergedConvexHullCKey = self.addCurve('', pen = self.convexHullPen, style=QwtPlotCurve.Lines)
        self.verticalConvexHullCKey = self.addCurve('', pen = self.convexHullPen, style=QwtPlotCurve.Lines)
        self.thresholdConvexHullCKey = self.addCurve('', pen = self.convexHullPen, style=QwtPlotCurve.Lines)
        self.classifierConvexHullCKey = self.addCurve('', pen = self.convexHullPen, style=QwtPlotCurve.Lines)

        ## iso-performance line
        self.performanceLineCKey = None

    def setIterationCurves(self, iteration, curves):
        classifier = 0
        for c in curves:
            x = [px for (px, py, pf) in c]
            y = [py for (px, py, pf) in c]
            curve = self.classifierIterationCKeys[classifier][iteration]
            curve.setData(x, y)
            self.classifierIterationROCdata[classifier][iteration] = c
            classifier += 1

    def setIterationConvexCurves(self, iteration, curves):
        classifier = 0
        for c in curves:
            x = [px for (px, py, pf) in c]
            y = [py for (px, py, pf) in c]
            curve = self.classifierIterationConvexCKeys[classifier][iteration]
            curve.setData(x, y)
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
        self.diagonalCKey.setVisible(self.showDiagonal)

        showSomething = 0
        for cNum in range(len(zip(self.showClassifiers, self.mergedCKeys))):
            showCNum = (self.showClassifiers[cNum] <> 0)

            ## 'merge' averaging
            b = (self.averagingMethod == 'merge') and showCNum
            showSomething = showSomething or b
            if self.mergedCKeys[cNum]:
                self.mergedCKeys[cNum].setVisible(b)

            b2 = b and self.showDefaultThresholdPoint
            if self.mergedCThresholdKeys[cNum] <> None:
                self.mergedCThresholdKeys[cNum].setVisible(b2)

            for marker in self.mergedCThresholdMarkers[cNum]:
                marker.setVisible(b2)

            b = b and self.showConvexCurves
            if self.mergedConvexCKeys[cNum] <> None:
                self.mergedConvexCKeys[cNum].setVisible(b)

            ## 'vertical' averaging
            b = (self.averagingMethod == 'vertical') and showCNum
            showSomething = showSomething or b
            self.verticalCKeys[cNum].setVisible(b)

            ## 'threshold' averaging
            b = (self.averagingMethod == 'threshold') and showCNum
            showSomething = showSomething or b
            self.thresholdCKeys[cNum].setVisible(b)

            ## 'None' averaging
            for iNum in range(len(zip(self.showIterations, self.classifierIterationCKeys[cNum], self.classifierIterationConvexCKeys[cNum]))):
                b = (self.averagingMethod == None) and showCNum and (self.showIterations[iNum] <> 0)
                showSomething = showSomething or b
                self.classifierIterationCKeys[cNum][iNum].setVisible(b)
                b = b and self.showConvexCurves
                self.classifierIterationConvexCKeys[cNum][iNum].setVisible(b)

        chb = (showSomething) and (self.averagingMethod == None) and self.showConvexHull
        if self.classifierConvexHullCKey:
            self.classifierConvexHullCKey.setVisible(chb)

        chb = (showSomething) and (self.averagingMethod == 'merge') and self.showConvexHull
        if self.mergedConvexHullCKey:
            self.mergedConvexHullCKey.setVisible(chb)

        chb = (showSomething) and (self.averagingMethod == 'vertical') and self.showConvexHull
        if self.verticalConvexHullCKey:
            self.verticalConvexHullCKey.setVisible(chb)

        chb = (showSomething) and (self.averagingMethod == 'threshold') and self.showConvexHull
        if self.thresholdConvexHullCKey:
            self.thresholdConvexHullCKey.setVisible(chb)

        ## performance line
        b = (self.averagingMethod == 'merge') and self.showPerformanceLine
        for marker in self.performanceMarkerKeys:
            marker.setVisible(b)

        if self.performanceLineCKey:
##            curve.setVisible(b)
            self.performanceLineCKey.setVisible(b)

##        self.updateLayout()
##        self.update()
        self.replot()

    def setShowConvexCurves(self, b):
        self.showConvexCurves = b
        self.updateCurveDisplay()

    def setShowConvexHull(self, b):
        self.showConvexHull = b
        self.updateCurveDisplay()

    def setShowPerformanceLine(self, b):
        self.showPerformanceLine = b
        self.updateCurveDisplay()

    def setShowDefaultThresholdPoint(self, b):
        self.showDefaultThresholdPoint = b
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
        for i, (isplit, show) in enumerate(zip(self.splitByIterations, self.showIterations)):
            if show:
                for te in isplit.results:
                    mergedIterations.results.append( te )
                    
        self.mergedConvexHullData = []
        if len(mergedIterations.results) > 0:
            curves = self.computeCurve(mergedIterations, self.targetClass, 1)
            convexCurves = self.computeCurve(mergedIterations, self.targetClass, 0)
            classifier = 0
            for c in curves:
                x = [px for (px, py, pf) in c]
                y = [py for (px, py, pf) in c]
                self.mergedCKeys[classifier].setData(x, y)

                # points of defualt threshold classifiers
                defPoint = [(abs(pf-0.5), pf, px, py) for (px, py, pf) in c]
                defPoints = []
                if len(defPoint) > 0:
                    defPoint.sort()
                    defPoints = [(px, py, pf) for (d, pf, px, py) in defPoint if d == defPoint[0][0]]
                else:
                    defPoints = []
                defX = [px for (px, py, pf) in defPoints]
                defY = [py for (px, py, pf) in defPoints]
                self.mergedCThresholdKeys[classifier].setData(defX, defY)

                for marker in self.mergedCThresholdMarkers[classifier]:
                    marker.detach()
                self.mergedCThresholdMarkers[classifier] = []
                for (dx, dy, pf) in defPoints:
                    dx = max(min(0.95, dx + 0.01), 0.01)
                    dy = min(max(0.01, dy - 0.02), 0.95)
                    marker = self.addMarker('%3.2g' % (pf), dx, dy, alignment = Qt.AlignRight)
                    self.mergedCThresholdMarkers[classifier].append(marker)
                classifier += 1
            classifier = 0
            for c in convexCurves:
                self.mergedConvexHullData.append(c) ## put all points of all curves into one big array
                x = [px for (px, py, pf) in c]
                y = [py for (px, py, pf) in c]
                curve = self.mergedConvexCKeys[classifier]
                curve.setData(x, y)
                classifier += 1
        else:
            for c in range(len(self.mergedCKeys)):
                self.mergedCKeys[c].setData([], [])
                self.mergedCThresholdKeys[c].setData([], [])
                for marker in self.mergedCThresholdMarkers[c]:
                    marker.detach()
                self.mergedCThresholdMarkers[c] = []
                self.mergedConvexCKeys[c].setData([], [])

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
            for curve in self.verticalCKeys:
                curve.setData([], [])

            for curve in self.thresholdCKeys:
                curve.setData([], [])
            return

        ##
        ## self.averagingMethod == 'vertical':
        ## calculated from the self.classifierIterationROCdata data
        try:
            (averageCurves, verticalErrorBarValues) = orngStat.TCverticalAverageROC(ROCS, self.VTAsamples)
        except (ValueError, SystemError), er:
            print >> sys.stderr, "Failed to compute vertical average ROC curve. " + er.message
            averageCurves, verticalErrorBarValues = [], []
            pass
         
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
            ckey.setData(xs, mps)
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
            ckey.setData(xs, mps)
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
        self.classifierConvexHullCKey.setData(x, y)

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
        self.mergedConvexHullCKey.setData(x, y)

        ## self.verticalConvexHullCKey = -1
        hullData = []
        for cNum in range(len(self.verticalConvexHullData)):
            if (self.showClassifiers[cNum] <> 0):
                hullData.append(self.verticalConvexHullData[cNum])

        convexHullCurve = TCconvexHull(hullData)
        x = [px for (px, py, pf) in convexHullCurve]
        y = [py for (px, py, pf) in convexHullCurve]
        self.verticalConvexHullCKey.setData(x, y)

        ## self.thresholdConvexHullCKey = -1
        hullData = []
        for cNum in range(len(self.thresholdConvexHullData)):
            if (self.showClassifiers[cNum] <> 0):
                hullData.append(self.thresholdConvexHullData[cNum])

        convexHullCurve = TCconvexHull(hullData)
        x = [px for (px, py, pf) in convexHullCurve]
        y = [py for (px, py, pf) in convexHullCurve]
        self.thresholdConvexHullCKey.setData(x, y)

    def setAveragingMethod(self, m):
        self.averagingMethod = m
        self.updateCurveDisplay()

    ## performance line
    def calcUpdatePerformanceLine(self):
        closestpoints = orngStat.TCbestThresholdsOnROCcurve(self.FPcost, self.FNcost, self.pvalue, self.hullCurveDataForPerfLine)
        m = (self.FPcost*(1.0 - self.pvalue)) / (self.FNcost*self.pvalue)

        ## now draw the closest line to the curve
        b = (self.averagingMethod == 'merge') and self.showPerformanceLine
        lpx = []
        lpy = []
        first = 1
        ## remove old markers
        for marker in self.performanceMarkerKeys:
            try:
                marker.detach()
            except RuntimeError: ## RuntimeError: underlying C/C++ object has been deleted
                pass
        self.performanceMarkerKeys = []
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
                px = max(min(0.95, px + 0.01), 0.01)
                py = min(max(0.01, py - 0.02), 0.95)
                marker = self.addMarker(s, px, py, alignment = Qt.AlignRight)
                marker.setVisible(b)
                self.performanceMarkerKeys.append(marker)
        if len(closestpoints) > 0:
            lpx.append(x + 2.0)
            lpy.append(y + 2.0*m)

        if self.performanceLineCKey:
            self.performanceLineCKey.setData(lpx, lpy)
            self.performanceLineCKey.setVisible(b)
        self.replot()
#        self.update()

    def costChanged(self, FPcost, FNcost):
        self.FPcost = float(FPcost)
        self.FNcost = float(FNcost)
        self.calcUpdatePerformanceLine()

    def pChanged(self, pvalue):
        self.pvalue = float(pvalue)
        self.calcUpdatePerformanceLine()

    def setPointWidth(self, v):
        self.performanceLineSymbol.setSize(v, v)
        if self.performanceLineCKey:
            self.performanceLineCKey.setSymbol(self.performanceLineSymbol)
            
        def setW(curve):
            sym = curve.symbol() #.setPen(QPen(self.classifierColor[cNum], v))
            sym.setSize(v + 1, v + 1)
            if QWT_VERSION_STR >= "5.2": # in Qwt 5.1.* curve.setSymbol results in a crash 
                curve.setSymbol(sym)
            
        for item in self.itemList():
            setW(item)
            
#        for cNum in range(len(zip(self.showClassifiers, self.mergedCKeys))):
#            setW(self.mergedCKeys[cNum]) #.setPen(QPen(self.classifierColor[cNum], v))
#            setW(self.verticalCKeys[cNum]) #.setPen(QPen(self.classifierColor[cNum], v))
#            setW(self.thresholdCKeys[cNum]) #.setPen(QPen(self.classifierColor[cNum], v))
#            setW(self.mergedCThresholdKeys[cNum])
#            for iNum in range(len(zip(self.showIterations, self.classifierIterationCKeys[cNum]))):
#                setW(self.classifierIterationCKeys[cNum][iNum]) #.setPen(QPen(self.classifierColor[cNum], v))
        self.replot()
#        self.update()

    def setCurveWidth(self, v):
        for cNum in range(len(zip(self.showClassifiers, self.mergedCKeys))):
            self.mergedCKeys[cNum].setPen(QPen(self.classifierColor[cNum], v))
            self.verticalCKeys[cNum].setPen(QPen(self.classifierColor[cNum], v))
            self.thresholdCKeys[cNum].setPen(QPen(self.classifierColor[cNum], v))
            for iNum in range(len(zip(self.showIterations, self.classifierIterationCKeys[cNum]))):
                self.classifierIterationCKeys[cNum][iNum].setPen(QPen(self.classifierColor[cNum], v))
        self.replot()
#        self.update()

    def setConvexCurveWidth(self, v):
        for cNum in range(len(zip(self.showClassifiers, self.mergedConvexCKeys))):
            self.mergedConvexCKeys[cNum].setPen(QPen(self.classifierColor[cNum], v))
            for iNum in range(len(zip(self.showIterations, self.classifierIterationConvexCKeys[cNum]))):
                self.classifierIterationConvexCKeys[cNum][iNum].setPen(QPen(self.classifierColor[cNum], v))
        self.replot()
#        self.update()

    def setShowDiagonal(self, v):
        self.showDiagonal = v
        self.updateCurveDisplay()

    def setConvexHullCurveWidth(self, v):
        self.convexHullPen.setWidth(v)
        self.mergedConvexHullCKey.setPen(self.convexHullPen)
        self.verticalConvexHullCKey.setPen(self.convexHullPen)
        self.thresholdConvexHullCKey.setPen(self.convexHullPen)
        self.classifierConvexHullCKey.setPen(self.convexHullPen)
        self.replot()
#        self.update()

    def setHullColor(self, c):
        self.convexHullPen.setColor(c)
        self.mergedConvexHullCKey.setPen(self.convexHullPen)
        self.verticalConvexHullCKey.setPen(self.convexHullPen)
        self.thresholdConvexHullCKey.setPen(self.convexHullPen)
        self.classifierConvexHullCKey.setPen(self.convexHullPen)
        self.replot()
#        self.update()

    def sizeHint(self):
        return QSize(100, 100)

class OWROC(OWWidget):
    settingsList = ["PointWidth", "CurveWidth", "ConvexCurveWidth", "ShowDiagonal",
                    "ConvexHullCurveWidth", "HullColor", "AveragingMethodIndex",
                    "ShowConvexHull", "ShowConvexCurves", "EnablePerformance", "DefaultThresholdPoint"]
    contextHandlers = {"": EvaluationResultsContextHandler("", "targetClass", "selectedClassifiers")}

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
        self.HullColor = str(QColor(Qt.yellow).name())
        self.AveragingMethodIndex = 0 ##'merge'
        self.ShowConvexHull = TRUE
        self.ShowConvexCurves = FALSE
        self.EnablePerformance = TRUE
        self.DefaultThresholdPoint = TRUE

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
        self.classifiers = []
        self.selectedClassifiers = []

        # performance analysis (temporary values
        self.FPcost = 500.0
        self.FNcost = 500.0
        self.pvalue = 50.0 ##0.400

        # list of values (remember for each class)
        self.FPcostList = []
        self.FNcostList = []
        self.pvalueList = []

        self.AveragingMethodNames = ['merge', 'vertical', 'threshold', None]
        self.AveragingMethod = self.AveragingMethodNames[min(3, self.AveragingMethodIndex)]

        # GUI
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
        #self.classCombo.setMaximumSize(150, 20)

        ## classifiers selection (classifiersQLB)
        self.classifiersQVGB = OWGUI.widgetBox(self.generalTab, "Classifiers")
        self.classifiersQLB = OWGUI.listBox(self.classifiersQVGB, self, "selectedClassifiers", selectionMode = QListWidget.MultiSelection, callback = self.classifiersSelectionChange)
        self.unselectAllClassifiersQLB = OWGUI.button(self.classifiersQVGB, self, "(Un)select All", callback = self.SUAclassifiersQLB)

        # show convex ROC curves and show ROC convex hull
        self.convexCurvesQCB = OWGUI.checkBox(self.generalTab, self, 'ShowConvexCurves', 'Show convex ROC curves', tooltip='', callback=self.setShowConvexCurves)
        OWGUI.checkBox(self.generalTab, self, 'ShowConvexHull', 'Show ROC convex hull', tooltip='', callback=self.setShowConvexHull)
        

        # performance analysis
        self.performanceTab = OWGUI.createTabPage(self.tabs, "Analysis")
        self.performanceTabCosts = OWGUI.widgetBox(self.performanceTab, box = 1)
        OWGUI.checkBox(self.performanceTabCosts, self, 'EnablePerformance', 'Show performance line', tooltip='', callback=self.setShowPerformanceAnalysis)
        OWGUI.checkBox(self.performanceTabCosts, self, 'DefaultThresholdPoint', 'Default threshold (0.5) point', tooltip='', callback=self.setShowDefaultThresholdPoint)

        ## FP and FN cost ranges
        mincost = 1; maxcost = 1000; stepcost = 5;
        self.maxpsum = 100; self.minp = 1; self.maxp = self.maxpsum - self.minp ## need it also in self.pvaluesUpdated
        stepp = 1.0

        OWGUI.hSlider(self.performanceTabCosts, self, 'FPcost', box='FP Cost', minValue=mincost, maxValue=maxcost, step=stepcost, callback=self.costsChanged, ticks=50)
        OWGUI.hSlider(self.performanceTabCosts, self, 'FNcost', box='FN Cost', minValue=mincost, maxValue=maxcost, step=stepcost, callback=self.costsChanged, ticks=50)

        ptc = OWGUI.widgetBox(self.performanceTabCosts, "Prior target class probability [%]")
        OWGUI.hSlider(ptc, self, 'pvalue', minValue=self.minp, maxValue=self.maxp, step=stepp, callback=self.pvaluesUpdated, ticks=5, labelFormat="%2.1f")
        OWGUI.button(ptc, self, 'Compute from data', self.setDefaultPValues) ## reset p values to default

        ## test set selection (testSetsQLB)
        self.testSetsQVGB = OWGUI.widgetBox(self.performanceTab, "Test sets")
        self.testSetsQLB = OWGUI.listBox(self.testSetsQVGB, self, selectionMode = QListWidget.MultiSelection, callback = self.testSetsSelectionChange)
        self.unselectAllTestSetsQLB = OWGUI.button(self.testSetsQVGB, self, "(Un)select All", callback = self.SUAtestSetsQLB)

        # settings tab
        self.settingsTab = OWGUI.createTabPage(self.tabs, "Settings")
        OWGUI.radioButtonsInBox(self.settingsTab, self, 'AveragingMethodIndex', ['Merge (expected ROC perf.)', 'Vertical', 'Threshold', 'None'], box='Averaging ROC curves', callback=self.selectAveragingMethod)
        OWGUI.hSlider(self.settingsTab, self, 'PointWidth', box='Point width', minValue=0, maxValue=9, step=1, callback=self.setPointWidth, ticks=1)
        OWGUI.hSlider(self.settingsTab, self, 'CurveWidth', box='ROC curve width', minValue=1, maxValue=5, step=1, callback=self.setCurveWidth, ticks=1)
        OWGUI.hSlider(self.settingsTab, self, 'ConvexCurveWidth', box='ROC convex curve width', minValue=1, maxValue=5, step=1, callback=self.setConvexCurveWidth, ticks=1)
        OWGUI.hSlider(self.settingsTab, self, 'ConvexHullCurveWidth', box='ROC convex hull', minValue=2, maxValue=9, step=1, callback=self.setConvexHullCurveWidth, ticks=1)
        OWGUI.checkBox(self.settingsTab, self, 'ShowDiagonal', 'Show diagonal ROC line', tooltip='', callback=self.setShowDiagonal)
        self.settingsTab.layout().addStretch(100)
      
        self.resize(800, 600)

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
            self.reportImage(self.graphs[self.targetClass].saveToFileDirect, QSize(400, 400))
        
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
            if qlb.item(i).isSelected():
                selected = 1
                break
        if selected: qlb.clearSelection()
        else: qlb.selectAll()

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
            if self.classifiersQLB.item(i).isSelected():
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

    def setShowDefaultThresholdPoint(self):
        for g in self.graphs:
            g.setShowDefaultThresholdPoint(self.DefaultThresholdPoint)

    ## test set selection (testSetsQLB)
    def testSetsSelectionChange(self):
        list = []
        for i in range(self.testSetsQLB.count()):
            if self.testSetsQLB.item(i).isSelected():
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
            g.setShowDefaultThresholdPoint(self.DefaultThresholdPoint)

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
                distrib.append(0.0 if sum - self.pvalue == 0  else self.pvalueList[vi] / float(sum - self.pvalue))

        dif = self.maxpsum - sum
        for vi in range(len(distrib)):
            self.pvalueList[vi] += int(float(dif) * distrib[vi])
            if self.pvalueList[vi] < self.minp:
                self.pvalueList[vi] = self.minp
            if self.pvalueList[vi] > self.maxp:
                self.pvalueList[vi] = self.maxp

        ## small changes
        dif = self.maxpsum - int(statc.sum(self.pvalueList))
        while abs(dif) > 0 and len(self.pvalueList) > 1:
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
        if self.defaultPerfLinePValues and self.targetClass != None:
            self.pvaluesList = [v for v in self.defaultPerfLinePValues]
            self.pvalue = self.pvaluesList[self.targetClass]
            self.pvaluesUpdated()

    def test_results(self, dres):
        self.FPcostList = []
        self.FNcostList = []
        self.pvalueList = []

        self.closeContext()

        if not dres:
            self.targetClass = None
            self.classCombo.clear()
            self.testSetsQLB.clear()
            self.classifiersQLB.clear()
            self.removeGraphs()
            self.openContext("", dres)
            return
        
        self.warning(0)
        if len(dres.results) > 0 and dres.results[0].multilabel_flag == 1:
            text = "there is no consensus on how to apply it in multi-class problems"
            self.warning(0, text)
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
                graph = singleClassROCgraph(self.mainArea, "", "Predicted class: " + self.dres.classValues[i])
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
        self.performanceTabCosts.setEnabled(self.AveragingMethod == 'merge')
        self.setDefaultPValues()

if __name__ == "__main__":
    a = QApplication(sys.argv)
    owdm = OWROC()
    owdm.show()
    a.exec_()
