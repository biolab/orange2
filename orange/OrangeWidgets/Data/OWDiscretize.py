"""
<name>Discretize</name>
<description>Discretization of continuous attributes.</description>
<icon>icons/Discretize.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>2100</priority>
"""
import orange
from OWWidget import *
from OWGraph import *
import OWGUI

def frange(low, up, steps):
    inc=(up-low)/steps
    return [low+i*inc for i in range(steps)]

class DiscGraph(OWGraph):
    def __init__(self, master, *args):
        OWGraph.__init__(self, *args)
        self.master=master

        self.rugKeys = []
        self.cutLineKeys = []
        self.cutMarkerKeys = []
        self.probCurveKey = None
        self.baseCurveKey = None
        self.lookaheadCurveKey = None

        self.setAxisScale(QwtPlot.yRight, 0.0, 1.0, 0.0)
        self.setYLaxisTitle("Split gain")
        self.setXaxisTitle("Attribute value")
        self.setYRaxisTitle("Class probability")
        self.setShowYRaxisTitle(1)
        self.setShowYLaxisTitle(1)
        self.setShowXaxisTitle(1)
        self.enableYLaxis(1)
        self.enableYRaxis(1)

        self.resolution=50
        #self.setCursor(Qt.ArrowCursor)
        #self.canvas().setCursor(Qt.ArrowCursor)

        self.data = self.attr = self.contingency = None
        self.minVal = self.maxVal = 0
        self.curCutPoints=[]


    def computeAddedScore(self, spoints):
        candidateSplits = [x for x in frange(self.minVal, self.maxVal, self.resolution) if x not in spoints]
        idisc = orange.IntervalDiscretizer(points = [-99999] + spoints)
        var = idisc.constructVariable(self.data.domain[self.master.continuousIndices[self.master.selectedAttr]])
        measure = self.master.measures[self.master.measure][1]
        score=[]
        chisq = self.master.measure == 2
        for cut in candidateSplits:
            idisc.points = spoints + [cut]
            idisc.points.sort()
            score.append(measure(var, self.data))

        return candidateSplits, score


    def invalidateBaseScore(self):
        self.baseCurveX = self.baseCurveY = None


    def computeLookaheadScore(self, split):
        if self.data and self.data.domain.classVar:
            self.lookaheadCurveX, self.lookaheadCurveY = self.computeAddedScore(list(self.curCutPoints) + [split])
        else:
            self.lookaheadCurveX = self.lookaheadCurveY = None


    def clearAll(self):
        for rug in self.rugKeys:
            rug.detach()
        self.rugKeys = []

        if self.baseCurveKey:
            self.baseCurveKey.detach()
            self.baseCurveKey = None

        if self.lookaheadCurveKey:
            self.lookaheadCurveKey.detach()
            self.lookaheadCurveKey = None

        if self.probCurveKey:
            self.probCurveKey.detach()
            self.probCurveKey = None

        for c in self.cutLineKeys:
            c.detach()
        self.cutLineKeys = []

        for m in self.cutMarkerKeys:
            m.detach()
        self.cutMarkerKeys = []

        self.replot()


    def setData(self, attr, data):
        self.clearAll()
        self.attr, self.data = attr, data
        self.curCutPoints = []

        if not data or not attr:
            self.snapDecimals = 1
            self.probDist = None
            return

        if data.domain.classVar:
            self.contingency = orange.ContingencyAttrClass(attr, data)
            try:
                self.condProb = orange.ConditionalProbabilityEstimatorConstructor_loess(
                   self.contingency,
                   nPoints=50)
            except:
                self.condProb = None
            self.probDist = None
            attrValues = self.contingency.keys()
        else:
            self.condProb = self.contingency = None
            self.probDist = orange.Distribution(attr, data)
            attrValues = self.probDist.keys()

        if attrValues:
            self.minVal, self.maxVal = min(attrValues), max(attrValues)
        else:
            self.minVal, self.maxVal = 0, 1
        mdist = self.maxVal - self.minVal
        if mdist > 1e-30:
            self.snapDecimals = -int(math.ceil(math.log(mdist, 10)) -2)
        else:
            self.snapDecimals = 1

        self.baseCurveX = None

        self.plotRug(True)
        self.plotProbCurve(True)
        self.plotCutLines(True)

        self.updateLayout()
        self.replot()


    def plotRug(self, noUpdate = False):
        for rug in self.rugKeys:
            rug.detach()
        self.rugKeys = []

        if self.master.showRug:
            targetClass = self.master.targetClass

            if self.contingency:
                freqhigh = [(val, freq[targetClass]) for val, freq in self.contingency.items() if freq[targetClass] > 1e-6]
                freqlow = [(val, freq.abs - freq[targetClass]) for val, freq in self.contingency.items()]
                freqlow = [f for f in freqlow if f[1] > 1e-6]
            elif self.probDist:
                freqhigh = []
                freqlow = self.probDist.items()
            else:
                return

            if freqhigh:
                maxf = max([f[1] for f in freqhigh])
                if freqlow:
                    maxf = max(maxf, max([f[1] for f in freqlow]))
            elif freqlow:
                maxf = max([f[1] for f in freqlow])
            else:
                return

            freqfac = maxf > 1e-6 and .1 / maxf or 1

            for val, freq in freqhigh:
                c = self.addCurve("", Qt.gray, Qt.gray, 1, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = [val, val], yData = [1.0, 1.0 - max(.02, freqfac * freq)], autoScale = 1)
                c.setYAxis(QwtPlot.yRight)
                self.rugKeys.append(c)

            for val, freq in freqlow:
                c = self.addCurve("", Qt.gray, Qt.gray, 1, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = [val, val], yData = [0.04, 0.04 + max(.02, freqfac * freq)], autoScale = 1)
                c.setYAxis(QwtPlot.yRight)
                self.rugKeys.append(c)

        if not noUpdate:
            self.replot()


    def plotBaseCurve(self, noUpdate = False):
        if self.baseCurveKey:
            self.baseCurveKey.detach()
            self.baseCurveKey = None

        if self.master.showBaseLine and self.master.resetIndividuals and self.data and self.data.domain.classVar and self.attr:
            if not self.baseCurveX:
                self.baseCurveX, self.baseCurveY = self.computeAddedScore(list(self.curCutPoints))

            #self.setAxisOptions(QwtPlot.yLeft, self.master.measure == 3 and QwtAutoScale.Inverted or QwtAutoScale.None)
            self.axisScaleEngine(QwtPlot.yLeft).setAttributes(self.master.measure == 3 and QwtScaleEngine.Inverted or QwtScaleEngine.NoAttribute)
            #self.axisScaleEngine(QwtPlot.yLeft).setAttribute(QwtScaleEngine.Inverted, self.master.measure == 3)
            self.baseCurveKey = self.addCurve("", Qt.black, Qt.black, 1, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = self.baseCurveX, yData = self.baseCurveY, lineWidth = 2, autoScale = 1)
            self.baseCurveKey.setYAxis(QwtPlot.yLeft)

        if not noUpdate:
            self.replot()


    def plotLookaheadCurve(self, noUpdate = False):
        if self.lookaheadCurveKey:
            self.lookaheadCurveKey.detach()
            self.lookaheadCurveKey = None

        if self.lookaheadCurveX and self.master.showLookaheadLine:
            #self.setAxisOptions(QwtPlot.yLeft, self.master.measure == 3 and QwtAutoScale.Inverted or QwtAutoScale.None)
            self.axisScaleEngine(QwtPlot.yLeft).setAttributes(self.master.measure == 3 and QwtScaleEngine.Inverted or QwtScaleEngine.NoAttribute)
            self.lookaheadCurveKey = self.addCurve("", Qt.black, Qt.black, 1, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = self.lookaheadCurveX, yData = self.lookaheadCurveY, lineWidth = 1, autoScale = 1)
            self.lookaheadCurveKey.setYAxis(QwtPlot.yLeft)
            #self.lookaheadCurveKey.setVisible(1)

        if not noUpdate:
            self.replot()


    def plotProbCurve(self, noUpdate = False):
        if self.probCurveKey:
            self.probCurveKey.detach()
            self.probCurveKey = None

        if self.contingency and self.condProb and self.master.showTargetClassProb:
            xData = self.contingency.keys()[1:-1]
            self.probCurveKey = self.addCurve("", Qt.gray, Qt.gray, 1, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = xData, yData = [self.condProb(x)[self.master.targetClass] for x in xData], lineWidth = 2, autoScale = 1)
            self.probCurveKey.setYAxis(QwtPlot.yRight)

        if not noUpdate:
            self.replot()


    def plotCutLines(self, noUpdate = False):
        attr = self.data.domain[self.master.continuousIndices[self.master.selectedAttr]]
        for c in self.cutLineKeys:
            c.detach()
        self.cutLineKeys = []

        for m in self.cutMarkerKeys:
            m.detach()
        self.cutMarkerKeys = []

        for cut in self.curCutPoints:
            c = self.addCurve("", Qt.blue, Qt.blue, 1, style = QwtPlotCurve.Steps, symbol = QwtSymbol.NoSymbol, xData = [cut, cut], yData = [.9, 0.1], autoScale = 1)
            c.setYAxis(QwtPlot.yRight)
            self.cutLineKeys.append(c)

            m = self.addMarker(str(attr(cut)), cut, .9, Qt.AlignCenter | Qt.AlignTop, bold=1)
            m.setYAxis(QwtPlot.yRight)
            self.cutMarkerKeys.append(m)
        if not noUpdate:
            self.replot()

    def getCutCurve(self, cut):
        ccc = self.transform(QwtPlot.xBottom, cut)
        for i, c in enumerate(self.curCutPoints):
            cc = self.transform(QwtPlot.xBottom, c)
            if abs(cc-ccc)<3:
                self.cutLineKeys[i].curveInd = i
                return self.cutLineKeys[i]
        return None


    def setSplits(self, splits):
        if self.data:
            self.curCutPoints = splits

            self.baseCurveX = None
            self.plotBaseCurve()
            self.plotCutLines()


    def addCutPoint(self, cut):
        self.curCutPoints.append(cut)
        c = self.addCurve("", Qt.blue, Qt.blue, 1, style = QwtPlotCurve.Steps, symbol = QwtSymbol.NoSymbol, xData = [cut, cut], yData = [1.0, 0.015], autoScale = 1)
        c.setYAxis(QwtPlot.yRight)
        self.cutLineKeys.append(c)
        c.curveInd = len(self.cutLineKeys) - 1
        return c


    def mousePressEvent(self, e):
        if not self.data:
            return

        self.mouseCurrentlyPressed = 1

        canvasPos = self.canvas().mapFrom(self, e.pos())
        cut = self.invTransform(QwtPlot.xBottom, canvasPos.x())
        curve = self.getCutCurve(cut)
        if not curve and self.master.snap:
            curve = self.getCutCurve(round(cut, self.snapDecimals))

        if curve:
            if e.button() == Qt.RightButton:
                self.curCutPoints.pop(curve.curveInd)
                self.plotCutLines(True)
            else:
                cut = self.curCutPoints.pop(curve.curveInd)
                self.plotCutLines(True)
                self.selectedCutPoint=self.addCutPoint(cut)
        else:
            self.selectedCutPoint=self.addCutPoint(cut)
            self.plotCutLines(True)

        self.baseCurveX = None
        self.plotBaseCurve()
        self.master.synchronizeIf()


    def mouseMoveEvent(self, e):
        if not self.data:
            return

        canvasPos = self.canvas().mapFrom(self, e.pos())

        if self.mouseCurrentlyPressed:
            if self.selectedCutPoint:
                canvasPos = self.canvas().mapFrom(self, e.pos())
                pos = self.invTransform(QwtPlot.xBottom, canvasPos.x())
                if self.master.snap:
                    pos = round(pos, self.snapDecimals)

                if self.curCutPoints[self.selectedCutPoint.curveInd]==pos:
                    return
                if pos > self.maxVal or pos < self.minVal:
                    self.curCutPoints.pop(self.selectedCutPoint.curveInd)
                    self.baseCurveX = None
                    self.plotCutLines(True)
                    self.mouseCurrentlyPressed = 0
                    return

                self.curCutPoints[self.selectedCutPoint.curveInd] = pos
                self.selectedCutPoint.setData([pos, pos], [.9, 0.1])

                self.computeLookaheadScore(pos)
                self.plotLookaheadCurve()
                self.replot()

                self.master.synchronizeIf()


        elif self.getCutCurve(self.invTransform(QwtPlot.xBottom, canvasPos.x())):
            self.canvas().setCursor(Qt.SizeHorCursor)
        else:
            self.canvas().setCursor(Qt.ArrowCursor)


    def mouseReleaseEvent(self, e):
        if not self.data:
            return

        self.mouseCurrentlyPressed = 0
        self.selectedCutPoint = None
        self.baseCurveX = None
        self.plotBaseCurve()
        self.plotCutLines(True)
        self.master.synchronizeIf()
        if self.lookaheadCurveKey and self.lookaheadCurveKey:
            self.lookaheadCurveKey.setVisible(0)
        self.replot()


    def targetClassChanged(self):
        self.plotRug()
        self.plotProbCurve()


class CustomListItemDelegate(QItemDelegate):
    def paint(self, painter, option, index):
        item = self.parent().itemFromIndex(index)
        item.setText(item.name + item.master.indiLabels[item.labelIdx])
        QItemDelegate.paint(self, painter, option, index)


class ListItemWithLabel(QListWidgetItem):
    def __init__(self, icon, name, labelIdx, master):
        QListWidgetItem.__init__(self, icon, name)
        self.name = name
        self.master = master
        self.labelIdx = labelIdx

#    def paint(self, painter):
#        btext = str(self.text())
#        self.setText(btext + self.master.indiLabels[self.labelIdx])
#        QListWidgetItem.paint(self, painter)
#        self.setText(btext)


class OWDiscretize(OWWidget):
    settingsList=["autoApply", "measure", "showBaseLine", "showLookaheadLine", "showTargetClassProb", "showRug", "snap", "autoSynchronize"]
    contextHandlers = {"": PerfectDomainContextHandler("", ["targetClass", "discretization", "classDiscretization",
                                                     "indiDiscretization", "intervals", "classIntervals", "indiIntervals",
                                                     "outputOriginalClass", "indiData", "indiLabels", "resetIndividuals",
                                                     "selectedAttr", "customSplits", "customClassSplits"])}

    callbackDeposit=[]

    D_N_METHODS = 5
    D_LEAVE, D_ENTROPY, D_FREQUENCY, D_WIDTH, D_REMOVE = range(5)
    D_NEED_N_INTERVALS = [2, 3]

    def __init__(self, parent=None, signalManager=None, name="Interactive Discretization"):
        OWWidget.__init__(self, parent, signalManager, name)
        self.showBaseLine=1
        self.showLookaheadLine=1
        self.showTargetClassProb=1
        self.showRug=0
        self.snap=1
        self.measure=0
        self.targetClass=0
        self.discretization = self.classDiscretization = self.indiDiscretization = 1
        self.intervals = self.classIntervals = self.indiIntervals = 3
        self.outputOriginalClass = True
        self.indiData = []
        self.indiLabels = []
        self.resetIndividuals = 0
        self.customClassSplits = ""

        self.selectedAttr = 0
        self.customSplits = ["", "", ""]
        self.autoApply = True
        self.dataChanged = False
        self.autoSynchronize = True
        self.pointsChanged = False

        self.customLineEdits = []
        self.needsDiscrete = []

        self.data = self.originalData = None

        self.loadSettings()

        self.inputs=[("Examples", ExampleTable, self.setData)]
        self.outputs=[("Examples", ExampleTable)]
        self.measures=[("Information gain", orange.MeasureAttribute_info()),
                       #("Gain ratio", orange.MeasureAttribute_gainRatio),
                       ("Gini", orange.MeasureAttribute_gini()),
                       ("chi-square", orange.MeasureAttribute_chiSquare()),
                       ("chi-square prob.", orange.MeasureAttribute_chiSquare(computeProbabilities=1)),
                       ("Relevance", orange.MeasureAttribute_relevance()),
                       ("ReliefF", orange.MeasureAttribute_relief())]
        self.discretizationMethods=["Leave continuous", "Entropy-MDL discretization", "Equal-frequency discretization", "Equal-width discretization", "Remove continuous attributes"]
        self.classDiscretizationMethods=["Equal-frequency discretization", "Equal-width discretization"]
        self.indiDiscretizationMethods=["Default", "Leave continuous", "Entropy-MDL discretization", "Equal-frequency discretization", "Equal-width discretization", "Remove attribute"]

        self.mainHBox =  OWGUI.widgetBox(self.mainArea, orientation=0)

        vbox = self.controlArea
        box = OWGUI.radioButtonsInBox(vbox, self, "discretization", self.discretizationMethods[:-1], "Default discretization", callback=[self.clearLineEditFocus, self.defaultMethodChanged])
        self.needsDiscrete.append(box.buttons[1])
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.interBox = OWGUI.widgetBox(OWGUI.indentedBox(box))
        OWGUI.widgetLabel(self.interBox, "Number of intervals (for equal width/frequency)")
        OWGUI.separator(self.interBox, height=4)
        self.intervalSlider=OWGUI.hSlider(OWGUI.indentedBox(self.interBox), self, "intervals", None, 2, 10, callback=[self.clearLineEditFocus, self.defaultMethodChanged])
        OWGUI.appendRadioButton(box, self, "discretization", self.discretizationMethods[-1])
        OWGUI.separator(vbox)

        ribg = OWGUI.radioButtonsInBox(vbox, self, "resetIndividuals", ["Use default discretization for all attributes", "Explore and set individual discretizations"], "Individual attribute treatment", callback = self.setAllIndividuals)
        ll = QWidget(ribg)
        ll.setFixedHeight(1)
        OWGUI.widgetLabel(ribg, "Set discretization of all attributes to")
        hcustbox = OWGUI.widgetBox(OWGUI.indentedBox(ribg), 0, 0)
        for c in range(1, 4):
            OWGUI.appendRadioButton(ribg, self, "resetIndividuals", "Custom %i" % c, insertInto = hcustbox)

        OWGUI.separator(vbox)

        box = self.classDiscBox = OWGUI.radioButtonsInBox(vbox, self, "classDiscretization", self.classDiscretizationMethods, "Class discretization", callback=[self.clearLineEditFocus, self.classMethodChanged])
        cinterBox = OWGUI.widgetBox(box)
        OWGUI.widgetLabel(cinterBox, "Number of intervals")
        OWGUI.separator(cinterBox, height=4)
        self.intervalSlider=OWGUI.hSlider(OWGUI.indentedBox(cinterBox), self, "classIntervals", None, 2, 10, callback=[self.clearLineEditFocus, self.classMethodChanged])
        hbox = OWGUI.widgetBox(box, orientation = 0)
        OWGUI.appendRadioButton(box, self, "discretization", "Custom" + "  ", insertInto = hbox)
        self.classCustomLineEdit = OWGUI.lineEdit(hbox, self, "customClassSplits", callback = self.classCustomChanged, focusInCallback = self.classCustomSelected)
#        Can't validate - need to allow spaces
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.separator(box)
        self.classIntervalsLabel = OWGUI.widgetLabel(box, "Current splits: ")
        OWGUI.separator(box)
        OWGUI.checkBox(box, self, "outputOriginalClass", "Output original class", callback = self.commitIf)
        OWGUI.widgetLabel(box, "("+"Widget always uses discretized class internally."+")")

        OWGUI.separator(vbox)
        #OWGUI.rubber(vbox)

        box = OWGUI.widgetBox(vbox, "Commit")
        applyButton = OWGUI.button(box, self, "Commit", callback = self.commit)
        autoApplyCB = OWGUI.checkBox(box, self, "autoApply", "Commit automatically", callback=[self.clearLineEditFocus])
        OWGUI.setStopper(self, applyButton, autoApplyCB, "dataChanged", self.commit)
        OWGUI.rubber(vbox)

        #self.mainSeparator = OWGUI.separator(self.mainHBox, width=25)        # space between control and main area
        self.mainIABox =  OWGUI.widgetBox(self.mainHBox, "Individual attribute settings")
        self.mainBox = OWGUI.widgetBox(self.mainIABox, orientation=0)
        OWGUI.separator(self.mainIABox)#, height=30)
        graphBox = OWGUI.widgetBox(self.mainIABox, "", orientation=0)
        
        
#        self.needsDiscrete.append(graphBox)
        graphOptBox = OWGUI.widgetBox(graphBox)
        OWGUI.separator(graphBox, width=10)
        self.graph = DiscGraph(self, graphBox)
        graphBox.layout().addWidget(self.graph)

        #graphOptBox.layout().setSpacing(4)
        box = OWGUI.widgetBox(graphOptBox, "Split gain measure", addSpace=True)
        self.measureCombo=OWGUI.comboBox(box, self, "measure", orientation=0, items=[e[0] for e in self.measures], callback=[self.clearLineEditFocus, self.graph.invalidateBaseScore, self.graph.plotBaseCurve])
        OWGUI.checkBox(box, self, "showBaseLine", "Show discretization gain", callback=[self.clearLineEditFocus, self.graph.plotBaseCurve])
        OWGUI.checkBox(box, self, "showLookaheadLine", "Show lookahead gain", callback=self.clearLineEditFocus)
        self.needsDiscrete.append(box)

        box = OWGUI.widgetBox(graphOptBox, "Target class", addSpace=True)
        self.targetCombo=OWGUI.comboBox(box, self, "targetClass", orientation=0, callback=[self.clearLineEditFocus, self.graph.targetClassChanged])
        stc = OWGUI.checkBox(box, self, "showTargetClassProb", "Show target class probability", callback=[self.clearLineEditFocus, self.graph.plotProbCurve])
        OWGUI.checkBox(box, self, "showRug", "Show rug (may be slow)", callback=[self.clearLineEditFocus, self.graph.plotRug])
        self.needsDiscrete.extend([self.targetCombo, stc])

        box = OWGUI.widgetBox(graphOptBox, "Editing", addSpace=True)
        OWGUI.checkBox(box, self, "snap", "Snap to grid", callback=[self.clearLineEditFocus])
        syncCB = OWGUI.checkBox(box, self, "autoSynchronize", "Apply on the fly", callback=self.clearLineEditFocus)
        syncButton = OWGUI.button(box, self, "Apply", callback = self.synchronizePressed)
        OWGUI.setStopper(self, syncButton, syncCB, "pointsChanged", self.synchronize)
        OWGUI.rubber(graphOptBox)

        self.attrList = OWGUI.listBox(self.mainBox, self, callback = self.individualSelected)
        self.attrList.setItemDelegate(CustomListItemDelegate(self.attrList))
        self.attrList.setFixedWidth(300)

        self.defaultMethodChanged()

        OWGUI.separator(self.mainBox, width=10)
        box = OWGUI.radioButtonsInBox(OWGUI.widgetBox(self.mainBox), self, "indiDiscretization", [], callback=[self.clearLineEditFocus, self.indiMethodChanged])
        #hbbox = OWGUI.widgetBox(box)
        #hbbox.layout().setSpacing(4)
        for meth in self.indiDiscretizationMethods[:-1]:
            OWGUI.appendRadioButton(box, self, "indiDiscretization", meth)
        self.needsDiscrete.append(box.buttons[2])
        self.indiInterBox = OWGUI.indentedBox(box, orientation = "horizontal")
        OWGUI.widgetLabel(self.indiInterBox, "Num. of intervals: ")
        self.indiIntervalSlider = OWGUI.hSlider(self.indiInterBox, self, "indiIntervals", None, 2, 10, callback=[self.clearLineEditFocus, self.indiMethodChanged], width = 100)
        OWGUI.rubber(self.indiInterBox) 
        OWGUI.appendRadioButton(box, self, "indiDiscretization", self.indiDiscretizationMethods[-1])
        #OWGUI.rubber(hbbox)
        #OWGUI.separator(box)
        #hbbox = OWGUI.widgetBox(box)
        for i in range(3):
            hbox = OWGUI.widgetBox(box, orientation = "horizontal")
            OWGUI.appendRadioButton(box, self, "indiDiscretization", "Custom %i" % (i+1) + " ", insertInto = hbox)
            le = OWGUI.lineEdit(hbox, self, "", callback = lambda w=i: self.customChanged(w), focusInCallback = lambda w=i: self.customSelected(w))
            le.setFixedWidth(110)
            self.customLineEdits.append(le)
            OWGUI.button(hbox, self, "CC", width=30, callback = lambda w=i: self.copyToCustom(w))
            OWGUI.rubber(hbox)
        OWGUI.rubber(box)

        #self.controlArea.setFixedWidth(0)

        self.contAttrIcon =  self.createAttributeIconDict()[orange.VarTypes.Continuous]



    def setData(self, data=None):
        self.closeContext()

        self.indiData = []
        self.attrList.clear()
        for le in self.customLineEdits:
            le.clear()
        self.indiDiscretization = 0

        self.originalData = data
        haveClass = bool(data and data.domain.classVar)
        continuousClass = haveClass and data.domain.classVar.varType == orange.VarTypes.Continuous

        self.data = self.originalData
        if continuousClass:
            if not self.discretizeClass():
                self.data = self.discClassData = None
                self.warning(0)
                self.error(0, "Cannot discretize the class")
        else:
            self.data = self.originalData
            self.discClassData = None

        for c in self.needsDiscrete:
            c.setVisible(haveClass)

        if self.data:
            domain = self.data.domain
            self.continuousIndices = [i for i, attr in enumerate(domain.attributes) if attr.varType == orange.VarTypes.Continuous]
            if not self.continuousIndices:
                self.data = None

        self.classDiscBox.setEnabled(not data or continuousClass)
        if self.data:
            for i, attr in enumerate(domain.attributes):
                if attr.varType == orange.VarTypes.Continuous:
                    self.attrList.addItem(ListItemWithLabel(self.contAttrIcon, attr.name, self.attrList.count(), self))
                    self.indiData.append([0, 4, "", "", ""])
                else:
                    self.indiData.append(None)

            self.fillClassCombo()
            self.indiLabels = [""] * self.attrList.count()

            self.graph.setData(None, self.data)
            self.selectedAttr = 0
            self.openContext("", data)
#            if self.classDiscretization == 2:
#                self.discretizeClass()

            # Prevent entropy discretization with non-discrete class
            if not haveClass:
                if self.discretization == self.D_ENTROPY:
                    self.discretization = self.D_FREQUENCY
                # Say I'm overcautious if you will, but you haven't seen as much as I did :)
                if not haveClass:
                    if self.indiDiscretization-1 == self.D_ENTROPY:
                        self.indiDiscretization = 0
                    for indiData in self.indiData:
                        if indiData and indiData[0] == self.D_ENTROPY:
                            indiData[0] = 0

            self.computeDiscretizers()
            self.attrList.setCurrentItem(self.attrList.item(self.selectedAttr))
        else:
            self.targetCombo.clear()
            self.graph.setData(None, None)

#        self.graph.setData(self.data)

        self.makeConsistent()

        # this should be here because 'resetIndividuals' is a context setting
        self.showHideIndividual()

        self.commit()


    def fillClassCombo(self):
        self.targetCombo.clear()

        if not self.data or not self.data.domain.classVar:
            return

        domain = self.data.domain
        for v in domain.classVar.values:
            self.targetCombo.addItem(str(v))
        if self.targetClass<len(domain.classVar.values):
            self.targetCombo.setCurrentIndex(self.targetClass)
        else:
            self.targetCombo.setCurrentIndex(0)
            self.targetClass=0

    def classChanged(self):
        self.fillClassCombo()
        self.computeDiscretizers()


    def clearLineEditFocus(self):
        if self.data:
            df = self.indiDiscretization
            for le in self.customLineEdits:
                if le.hasFocus():
                    le.clearFocus()
            self.indiDiscretization = self.indiData[self.continuousIndices[self.selectedAttr]][0] = df
            if self.classCustomLineEdit.hasFocus():
                self.classCustomLineEdit.clearFocus()



    def individualSelected(self):
        if not self.data:
            return

        if self.attrList.selectedItems() == []: return
        self.selectedAttr = self.attrList.row(self.attrList.selectedItems()[0])
        attrIndex = self.continuousIndices[self.selectedAttr]
        attr = self.data.domain[attrIndex]
        indiData = self.indiData[attrIndex]

        self.customSplits = indiData[2:]
        for le, cs in zip(self.customLineEdits, self.customSplits):
            le.setText(" ".join(cs))

        self.indiDiscretization, self.indiIntervals = indiData[:2]
        self.indiInterBox.setEnabled(self.indiDiscretization-1 in self.D_NEED_N_INTERVALS)

        self.graph.setData(attr, self.data)
        if hasattr(self, "discretizers"):
            self.graph.setSplits(self.discretizers[attrIndex] and self.discretizers[attrIndex].getValueFrom.transformer.points or [])
        else:
            self.graph.plotBaseCurve(False)


    def computeDiscretizers(self):
        self.discretizers = []

        if not self.data:
            return

        self.discretizers = [None] * len(self.data.domain)
        for i, idx in enumerate(self.continuousIndices):
            self.computeDiscretizer(i, idx)

        self.commitIf()


    def makeConsistent(self):
        self.interBox.setEnabled(self.discretization in self.D_NEED_N_INTERVALS)
        self.indiInterBox.setEnabled(self.indiDiscretization-1 in self.D_NEED_N_INTERVALS)


    def defaultMethodChanged(self):
        self.interBox.setEnabled(self.discretization in self.D_NEED_N_INTERVALS)

        if not self.data:
            return

        for i, idx in enumerate(self.continuousIndices):
            self.computeDiscretizer(i, idx, True)

        self.commitIf()

    def classMethodChanged(self):
        if not self.data:
            return

        self.discretizeClass()
        self.classChanged()
        attrIndex = self.continuousIndices[self.selectedAttr]
        self.graph.setData(self.data.domain[attrIndex], self.data)
        self.graph.setSplits(self.discretizers[attrIndex] and self.discretizers[attrIndex].getValueFrom.transformer.points or [])
        if self.targetClass > len(self.data.domain.classVar.values):
            self.targetClass = len(self.data.domain.classVar.values)-1


    def indiMethodChanged(self, dontSetACustom=False):
        if self.data:
            i, idx = self.selectedAttr, self.continuousIndices[self.selectedAttr]
            self.indiData[idx][0] = self.indiDiscretization
            self.indiData[idx][1] = self.indiIntervals

            self.indiInterBox.setEnabled(self.indiDiscretization-1 in self.D_NEED_N_INTERVALS)
            if self.indiDiscretization and self.indiDiscretization - self.D_N_METHODS != self.resetIndividuals - 1:
                self.resetIndividuals = 1

            if not self.data:
                return

            which = self.indiDiscretization - self.D_N_METHODS - 1
            if not dontSetACustom and which >= 0 and not self.customSplits[which]:
                attr = self.data.domain[idx]
                splitsTxt = self.indiData[idx][2+which] = [str(attr(x)) for x in self.graph.curCutPoints]
                self.customSplits[which] = splitsTxt # " ".join(splitsTxt)
                self.customLineEdits[which].setText(" ".join(splitsTxt))
                self.computeDiscretizer(i, idx)
            else:
                self.computeDiscretizer(i, idx)

            self.commitIf()


    def customSelected(self, which):
        if self.data and self.indiDiscretization != self.D_N_METHODS + which + 1: # added 1 - we need it, right?
            self.indiDiscretization = self.D_N_METHODS + which + 1
            idx = self.continuousIndices[self.selectedAttr]
            attr = self.data.domain[idx]
            self.indiMethodChanged()


    def showHideIndividual(self):
        if not self.resetIndividuals:
                self.mainArea.hide()
        elif self.mainArea.isHidden():
            self.graph.plotBaseCurve()
            self.mainArea.show()
        qApp.processEvents()
        QTimer.singleShot(0, self.adjustSize)

    def setAllIndividuals(self):
        self.showHideIndividual()

        if not self.data:
            return

        self.clearLineEditFocus()
        method = self.resetIndividuals
        if method == 1:
            return
        if method:
            method += self.D_N_METHODS - 1
        for i, idx in enumerate(self.continuousIndices):
            if self.indiData[idx][0] != method:
                self.indiData[idx][0] = method
                if i == self.selectedAttr:
                    self.indiDiscretization = method
                    self.indiMethodChanged(True) # don't set a custom
                    if method:
                        self.computeDiscretizer(i, idx)
                else:
                    self.computeDiscretizer(i, idx)

        self.attrList.reset()
        self.commitIf()


    def customChanged(self, which):
        if not self.data:
            return

        idx = self.continuousIndices[self.selectedAttr]
        le = self.customLineEdits[which]

        content = str(le.text()).replace(":", " ").replace(",", " ").split()
        content = dict.fromkeys(content).keys()  # remove duplicates (except 8.0, 8.000 ...)
        try:
            content.sort(lambda x, y:cmp(float(x), float(y)))
        except:
            content = str(le.text())

        le.setText(" ".join(content))
        self.customSplits[which] = content
        self.indiData[idx][which+2] = content

        self.indiData[idx][0] = self.indiDiscretization = which + self.D_N_METHODS + 1

        self.computeDiscretizer(self.selectedAttr, self.continuousIndices[self.selectedAttr])
        self.commitIf()


    def copyToCustom(self, which):
        self.clearLineEditFocus()
        if not self.data:
            return

        idx = self.continuousIndices[self.selectedAttr]

        if self.indiDiscretization >= self.D_N_METHODS + 1:
            splits = self.customSplits[self.indiDiscretization - self.D_N_METHODS - 1]
            try:
                valid = bool([float(i) for i in self.customSplits[which]].split())
            except:
                valid = False
        else:
            valid = False

        if not valid:
            attr = self.data.domain[idx]
            splits = list(self.discretizers[idx] and self.discretizers[idx].getValueFrom.transformer.points or [])
            splits = [str(attr(i)) for i in splits]

        self.indiData[idx][2+which] = self.customSplits[which] = splits
        self.customLineEdits[which].setText(" ".join(splits))
#        self.customSelected(which)


    # This weird construction of the list is needed for easier translation into other languages
    shortDiscNames = [""] + [" (%s)" % x for x in ("leave continuous", "entropy", "equal frequency", "equal width", "removed")] + [(" ("+"custom %i"+")") % x for x in range(1, 4)]

    def computeDiscretizer(self, i, idx, onlyDefaults=False):
        attr = self.data.domain[idx]
        indiData = self.indiData[idx]

        discType, intervals = indiData[:2]
        discName = self.shortDiscNames[discType]

        defaultUsed = not discType

        if defaultUsed:
            discType = self.discretization+1
            intervals = self.intervals

        if discType >= self.D_N_METHODS + 1:

            try:
                customs = [float(r) for r in indiData[discType-self.D_N_METHODS+1]]
            except:
                customs = []

            if not customs:
                discType = self.discretization+1
                intervals = self.intervals
                discName = "%s ->%s)" % (self.shortDiscNames[indiData[0]][:-1], self.shortDiscNames[discType][2:-1])
                defaultUsed = True

        if onlyDefaults and not defaultUsed:
            return

        discType -= 1
        try:
            if discType == self.D_LEAVE: # leave continuous
                discretizer = None
            elif discType == self.D_ENTROPY:
                discretizer = orange.EntropyDiscretization(attr, self.data)
            elif discType == self.D_FREQUENCY:
                discretizer = orange.EquiNDiscretization(attr, self.data, numberOfIntervals = intervals)
            elif discType == self.D_WIDTH:
                discretizer = orange.EquiDistDiscretization(attr, self.data, numberOfIntervals = intervals)
            elif discType == self.D_REMOVE:
                discretizer = False
            else:
                discretizer = orange.IntervalDiscretizer(points = customs).constructVariable(attr)
        except:
            discretizer = False


        self.discretizers[idx] = discretizer

        if discType == self.D_LEAVE:
            discInts = ""
        elif discType == self.D_REMOVE:
            discInts = ""
        elif not discretizer:
            discInts = ": "+"<can't discretize>"
        else:
            points = discretizer.getValueFrom.transformer.points
            discInts = points and (": " + ", ".join([str(attr(x)) for x in points])) or ": "+"<removed>"
        self.indiLabels[i] = discInts + discName
        self.attrList.reset()

        if i == self.selectedAttr:
            self.graph.setSplits(discretizer and discretizer.getValueFrom.transformer.points or [])



    def discretizeClass(self):
        if self.originalData:
            discType = self.classDiscretization
            classVar = self.originalData.domain.classVar

            if discType == 2:
                try:
                    content = self.customClassSplits.replace(":", " ").replace(",", " ").replace("-", " ").split()
                    customs = dict.fromkeys([float(x) for x in content]).keys()  # remove duplicates (except 8.0, 8.000 ...)
                    customs.sort()
                except:
                    customs = []

                if not customs:
                    discType = 0

            try:
                if discType == 0:
                    discretizer = orange.EquiNDiscretization(classVar, self.originalData, numberOfIntervals = self.classIntervals)
                elif discType == 1:
                    discretizer = orange.EquiDistDiscretization(classVar, self.originalData, numberOfIntervals = self.classIntervals)
                else:
                    discretizer = orange.IntervalDiscretizer(points = customs).constructVariable(classVar)

                self.discClassData = orange.ExampleTable(orange.Domain(self.originalData.domain.attributes, discretizer), self.originalData)
                if self.data:
                    self.data = self.discClassData
                # else, the data has no continuous attributes other then the class

                self.classIntervalsLabel.setText("Current splits: " + ", ".join([str(classVar(x)) for x in discretizer.getValueFrom.transformer.points]))
                self.error(0)
                self.warning(0)
                return True
            except:
                if self.data:
                    self.warning(0, "Cannot discretize the class; using previous class")
                else:
                    self.error(0, "Cannot discretize the class")
                self.classIntervalsLabel.setText("")
                return False


    def classCustomChanged(self):
        self.classMethodChanged()

    def classCustomSelected(self):
        if self.classDiscretization != 2: # prevent a cycle (this function called by setFocus at its end)
            self.classDiscretization = 2
            self.classMethodChanged()
            self.classCustomLineEdit.setFocus()

    def discretize(self):
        if not self.data:
            return


    def synchronizeIf(self):
        if self.autoSynchronize:
            self.synchronize()
        else:
            self.pointsChanged = True

    def synchronizePressed(self):
        self.clearLineEditFocus()
        self.synchronize()

    def synchronize(self):
        if not self.data:
            return

        slot = self.indiDiscretization - self.D_N_METHODS - 1
        if slot < 0:
            for slot in range(3):
                if not self.customLineEdits[slot].text():
                    break
            else:
                slot = 0
            self.indiDiscretization = slot + self.D_N_METHODS + 1

        idx = self.continuousIndices[self.selectedAttr]
        attr = self.data.domain[idx]
        cp = list(self.graph.curCutPoints)
        cp.sort()
        splits = [str(attr(i)) for i in cp]
        splitsTxt = " ".join(splits)
        self.indiData[idx][0] = self.indiDiscretization
        self.indiData[idx][2+slot] = self.customSplits[slot] = splits
        self.customLineEdits[slot].setText(splitsTxt)

        discretizer = orange.IntervalDiscretizer(points = cp).constructVariable(attr)
        self.discretizers[idx] = discretizer

        self.indiLabels[self.selectedAttr] = ": " + splitsTxt + self.shortDiscNames[self.indiDiscretization]
        self.attrList.reset()

        self.pointsChanged = False
        self.commitIf()


    def commitIf(self):
        if self.autoApply:
            self.commit()
        else:
            self.dataChanged = True

    def commit(self):
        self.clearLineEditFocus()

        if self.data:
            newattrs=[]
            for attr, disc in zip(self.data.domain.attributes, self.discretizers):
                if disc:
                    if disc.getValueFrom.transformer.points:
                        newattrs.append(disc)
                elif disc == None:  # can also be False -> remove
                    newattrs.append(attr)

            if self.data.domain.classVar:
                if self.outputOriginalClass:
                    newdomain = orange.Domain(newattrs, self.originalData.domain.classVar)
                else:
                    newdomain = orange.Domain(newattrs, self.data.domain.classVar)
            else:
                newdomain = orange.Domain(newattrs, None)

            newdata = orange.ExampleTable(newdomain, self.originalData)
            self.send("Examples", newdata)

        elif self.discClassData and self.outputOriginalClass:
            self.send("Examples", self.discClassData)

        elif self.originalData and not (self.originalData.domain.classVar and self.originalData.domain.classVar.varType == orange.VarTypes.Continuous and not self.discClassData):  # no continuous attributes...
            self.send("Examples", self.originalData)
        else:
            self.send("Examples", None)

        dataChanged = False


import sys
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWDiscretize()
    w.show()
#    d=orange.ExampleTable("../../doc/datasets/bridges.tab")
#    d=orange.ExampleTable("../../doc/datasets/auto-mpg.tab")
    #d = orange.ExampleTable("../../doc/datasets/iris.tab")
    d = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\iris.tab")
    w.setData(d)
    #w.setData(None)
    #w.setData(d)
    app.exec_()
    w.saveSettings()
