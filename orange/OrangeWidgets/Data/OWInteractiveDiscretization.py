"""
<name>Interactive Discretization</name>
<description>Interactive discretization of continuous attributes</description>
<icon>icons/InteractiveDiscretization.png</icon>
<priority>2105</priority>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
"""

import orange
from OWWidget import *
from OWGraph import *
from qt import *
from qtcanvas import *
import OWGUI, OWGraphTools
import Numeric

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

        self.customLink = -1

        self.setAxisScale(QwtPlot.yRight, 0.0, 1.0, 0.0)
        self.setYLaxisTitle("Split gain")
        self.setXaxisTitle("Attribute value")
        self.setYRaxisTitle("Class probability")
        self.setShowYRaxisTitle(1)
        self.setShowYLaxisTitle(1)
        self.setShowXaxisTitle(1)
        self.enableYRightAxis(1)

        self.resolution=50
        self.setCursor(Qt.arrowCursor)
        self.canvas().setCursor(Qt.arrowCursor)
        
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

            
    def computeBaseScore(self):
        self.baseCurveX, self.baseCurveY = self.computeAddedScore(list(self.curCutPoints))


    def computeLookaheadScore(self, split):
        self.lookaheadCurveX, self.lookaheadCurveY = self.computeAddedScore(list(self.curCutPoints) + [split])


    def setData(self, attr, data):
        self.clear()
        self.attr, self.data = attr, data
        self.curCutPoints = []

        if not data:
            return

        self.classColors = OWGraphTools.ColorPaletteHSV(len(data.domain.classVar.values))

        if not attr:
            return
        
        self.contingency = orange.ContingencyAttrClass(attr, data)
        self.condProb = orange.ConditionalProbabilityEstimatorConstructor_loess(self.contingency)

        attrValues = self.contingency.keys()        
        self.minVal, self.maxVal = min(attrValues), max(attrValues)
        self.snapDecimals = -int(math.ceil(math.log(self.maxVal-self.minVal, 10)) -2)
        
        self.replotAll()


    def plotRug(self, noUpdate = False):
        for rug in self.rugKeys:
            self.removeCurve(rug)
        self.rugKeys = []

        if self.master.showRug:
            targetClass = self.master.targetClass

            freqhigh = [(val, freq[targetClass]) for val, freq in self.contingency.items() if freq[targetClass] > 1e-6]
            freqlow = [(val, freq.abs - freq[targetClass]) for val, freq in self.contingency.items()]
            freqlow = [f for f in freqlow if f[1] > 1e-6]
            if not freqhigh or not freqlow:
                return
            freqfac = .1 / max(max([f[1] for f in freqhigh]), max([f[1] for f in freqlow]))

            for val, freq in freqhigh:        
                c = self.addCurve("", Qt.gray, Qt.gray, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [val, val], yData = [1.0, 1.0 - max(.02, freqfac * freq)])
                self.setCurveYAxis(c, QwtPlot.yRight)
                self.rugKeys.append(c)

            for val, freq in freqlow:        
                c = self.addCurve("", Qt.gray, Qt.gray, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [val, val], yData = [0.04, 0.04 + max(.02, freqfac * freq)])
                self.setCurveYAxis(c, QwtPlot.yRight)
                self.rugKeys.append(c)

        if not noUpdate:
            self.update()            


    def plotBaseCurve(self, noUpdate = False):
        if self.baseCurveKey:
            self.removeCurve(self.baseCurveKey)
            
        if self.master.showBaseLine:
            self.setAxisOptions(QwtPlot.yLeft, self.master.measure == 3 and QwtAutoScale.Inverted or QwtAutoScale.None)
            self.baseCurveKey = self.addCurve("", Qt.black, Qt.black, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = self.baseCurveX, yData = self.baseCurveY, lineWidth = 2)
            self.setCurveYAxis(self.baseCurveKey, QwtPlot.yLeft)
        else:
            self.baseCurveKey = None

        if not noUpdate:                
            self.update()

        
    def plotLookaheadCurve(self, noUpdate = False):
        if self.lookaheadCurveKey:
            self.removeCurve(self.lookaheadCurveKey)
            
        if self.master.showLookaheadLine:
            self.setAxisOptions(QwtPlot.yLeft, self.master.measure == 3 and QwtAutoScale.Inverted or QwtAutoScale.None)
            self.lookaheadCurveKey = self.addCurve("", Qt.black, Qt.black, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = self.lookaheadCurveX, yData = self.lookaheadCurveY, lineWidth = 1)
            self.setCurveYAxis(self.lookaheadCurveKey, QwtPlot.yLeft)
            self.curve(self.lookaheadCurveKey).setEnabled(1)
        else:
            self.lookaheadCurveKey = None

        if not noUpdate:            
            self.update()


    def plotProbCurve(self, noUpdate = False):
        if self.probCurveKey:
            self.removeCurve(self.probCurveKey)
            
        if self.master.showTargetClassProb:            
            xData = self.contingency.keys()[1:-1]
            self.probCurveKey = self.addCurve("", Qt.gray, Qt.gray, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = xData, yData = [self.condProb(x)[self.master.targetClass] for x in xData], lineWidth = 2)
            self.setCurveYAxis(self.probCurveKey, QwtPlot.yRight)
        else:
            self.probCurveKey = None

        if not noUpdate:
            self.update()

        
    def plotCutLines(self):
        attr = self.data.domain[self.master.continuousIndices[self.master.selectedAttr]]
        for c in self.cutLineKeys:
            self.removeCurve(c)
        
        for m in self.cutMarkerKeys:
            self.removeMarker(m)
        
        self.cutLineKeys = []
        self.cutMarkerKeys = []
        for cut in self.curCutPoints:
            c = self.addCurve("", Qt.blue, Qt.blue, 1, style = QwtCurve.Steps, symbol = QwtSymbol.None, xData = [cut, cut], yData = [.9, 0.1])
            self.setCurveYAxis(c, QwtPlot.yRight)
            self.cutLineKeys.append(c)

            m = self.addMarker(str(attr(cut)), cut, .9, Qt.AlignCenter + Qt.AlignTop, bold=1)
            self.setMarkerYAxis(m, QwtPlot.yRight)
            self.cutMarkerKeys.append(m)


    def getCutCurve(self, cut):
        ccc = self.transform(QwtPlot.xBottom, cut)
        for i,c in enumerate(self.curCutPoints):
            cc = self.transform(QwtPlot.xBottom, c)
            if abs(cc-ccc)<3:
                curve = self.curve(self.cutLineKeys[i])
                curve.curveInd = i
                return curve
        return None


    def setSplits(self, splits):
        if self.data:
            self.curCutPoints = splits

            self.computeBaseScore()
            self.plotBaseCurve()
            self.plotCutLines()
            self.customLink = -1
            

    def addCutPoint(self, cut):
        self.curCutPoints.append(cut)
        c = self.addCurve("", Qt.blue, Qt.blue, 1, style = QwtCurve.Steps, symbol = QwtSymbol.None, xData = [cut, cut], yData = [1.0, 0.015])
        self.setCurveYAxis(c, QwtPlot.yRight)
        self.cutLineKeys.append(c)
        curve = self.curve(c)
        curve.curveInd = len(self.cutLineKeys) - 1        
        return curve

    
    def onMousePressed(self, e):
        if not self.data:
            return
        
        self.mouseCurrentlyPressed = 1
        
        cut = self.invTransform(QwtPlot.xBottom, e.x())
        curve = self.getCutCurve(cut)
        if not curve and self.master.snap:
            curve = self.getCutCurve(round(cut, self.snapDecimals))
            
        if curve:
            if e.button() == Qt.RightButton:
                self.curCutPoints.pop(curve.curveInd)
                self.plotCutLines()
            else:
                cut = self.curCutPoints.pop(curve.curveInd)
                self.plotCutLines()
                self.selectedCutPoint=self.addCutPoint(cut)
        else:
            self.selectedCutPoint=self.addCutPoint(cut)
            self.plotCutLines()
            self.update()

        self.computeBaseScore()
        self.plotBaseCurve()
        self.master.synchronizeIf()


    def onMouseMoved(self, e):
        if not self.data:
            return
        
        if self.mouseCurrentlyPressed:
            if self.selectedCutPoint:
                pos = self.invTransform(QwtPlot.xBottom, e.x())
                if self.master.snap:
                    pos = round(pos, self.snapDecimals)

                if self.curCutPoints[self.selectedCutPoint.curveInd]==pos:
                    return
                if pos > self.maxVal or pos < self.minVal:
                    self.curCutPoints.pop(self.selectedCutPoint.curveInd)
                    self.computeBaseScore()
                    self.plotCutLines()
                    self.mouseCurrentlyPressed = 0
                    return
                
                self.curCutPoints[self.selectedCutPoint.curveInd] = pos
                self.selectedCutPoint.setData([pos, pos], [.9, 0.1])

                self.computeLookaheadScore(pos)
                self.plotLookaheadCurve()
                self.update()

                self.master.synchronizeIf()
                
                
        elif self.getCutCurve(self.invTransform(QwtPlot.xBottom, e.x())):
            self.canvas().setCursor(Qt.sizeHorCursor)
        else:
            self.canvas().setCursor(Qt.arrowCursor)

                                  
    def onMouseReleased(self, e):
        if not self.data:
            return
        
        self.mouseCurrentlyPressed = 0
        self.selectedCutPoint = None
        self.computeBaseScore()
        self.plotBaseCurve()
        self.plotCutLines()
        self.master.synchronizeIf()
        if self.lookaheadCurveKey:
            self.curve(self.lookaheadCurveKey).setEnabled(0)
        self.update()


    def targetClassChanged(self):
        self.plotRug()
        self.plotProbCurve()


    def replotAll(self):
        self.clear()
        if not self.contingency:
            return

        self.computeBaseScore()

        self.plotRug(True)
        self.plotProbCurve(True)
        self.plotBaseCurve(True)
        self.plotCutLines()

        self.updateLayout()
        self.update()
        


class ListItemWithLabel(QListBoxPixmap):
    def __init__(self, icon, name, labelIdx, master):
        QListBoxPixmap.__init__(self, icon, name)
        self.master = master
        self.labelIdx = labelIdx
        

    def paint(self, painter):
        btext = str(self.text())
        self.setText(btext + self.master.indiLabels[self.labelIdx])
        QListBoxPixmap.paint(self, painter)
        self.setText(btext)


class OWInteractiveDiscretization(OWWidget):
    settingsList=["autoApply", "measure", "showBaseLine", "showLookaheadLine", "showTargetClassProb", "showRug", "snap", "autoSynchronize"]
    contextHandlers = {"": DomainContextHandler("", ["targetClass", "discretization", "classDiscretization",
                                                     "indiDiscretization", "intervals", "classIntervals", "indiIntervals",
                                                     "outputOriginalClass", "indiData", "indiLabels", "resetIndividuals",
                                                     "selectedAttr", "customSplits"], False, False, False, False)}

    callbackDeposit=[]
    def __init__(self, parent=None, signalManager=None, name="Interactive Discretization"):
        OWWidget.__init__(self, parent, signalManager, name)
        self.showBaseLine=1
        self.showLookaheadLine=1
        self.showTargetClassProb=1
        self.showRug=1
        self.snap=1
        self.measure=0
        self.targetClass=0
        self.discretization = self.classDiscretization = self.indiDiscretization = 1
        self.intervals = self.classIntervals = self.indiIntervals = 3
        self.outputOriginalClass = True
        self.indiData = []
        self.indiLabels = []
        self.resetIndividuals = 0
        
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
        self.inputs=[("Examples", ExampleTableWithClass, self.cdata)]
        self.outputs=[("Examples", ExampleTableWithClass)]
        self.measures=[("Information gain", orange.MeasureAttribute_info()),
                       #("Gain ratio", orange.MeasureAttribute_gainRatio),
                       ("Gini", orange.MeasureAttribute_gini()),
                       ("chi-square", orange.MeasureAttribute_chiSquare()),
                       ("chi-square prob.", orange.MeasureAttribute_chiSquare(computeProbabilities=1)),
                       ("Relevance", orange.MeasureAttribute_relevance()),
                       ("ReliefF", orange.MeasureAttribute_relief())]
        self.discretizationMethods=["Leave continuous", "Entropy-MDL discretization", "Equal-frequency discretization", "Equal-width discretization"]
        self.classDiscretizationMethods=["Equal-frequency discretization", "Equal-width discretization"]
        self.indiDiscretizationMethods=["Default", "Leave continuous", "Entropy-MDL discretization", "Equal-frequency discretization", "Equal-width discretization"]

        self.layout = QVBoxLayout(self.mainArea, QVBoxLayout.TopToBottom,0)
        self.mainVBox =  OWGUI.widgetBox(self.mainArea)
        self.mainHBox =  OWGUI.widgetBox(self.mainVBox, orientation=0)

        vbox = OWGUI.widgetBox(self.mainHBox)
        box = OWGUI.radioButtonsInBox(vbox, self, "discretization", self.discretizationMethods, "Default discretization", callback=[self.clearLineEditFocus, self.defaultMethodChanged])
        self.needsDiscrete.append(box.buttons[1])
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.interBox = OWGUI.widgetBox(box)
        OWGUI.widgetLabel(self.interBox, "Number of intervals")
        OWGUI.separator(self.interBox, height=4)
        self.intervalSlider=OWGUI.hSlider(OWGUI.indentedBox(self.interBox), self, "intervals", None, 2, 10, callback=[self.clearLineEditFocus, self.defaultMethodChanged])
        OWGUI.separator(vbox)

        OWGUI.radioButtonsInBox(vbox, self, "resetIndividuals", ["Default discretization", "Custom 1", "Custom 2", "Custom 3", "Individual settings"], "Reset individual attribute settings", callback = self.setAllIndividuals)
        OWGUI.separator(vbox)
        
        box = self.classDiscBox = OWGUI.radioButtonsInBox(vbox, self, "classDiscretization", self.classDiscretizationMethods, "Class discretization", callback=[self.clearLineEditFocus, self.classMethodChanged])
        cinterBox = OWGUI.widgetBox(box)
        OWGUI.widgetLabel(cinterBox, "Number of intervals")
        OWGUI.separator(cinterBox, height=4)
        self.intervalSlider=OWGUI.hSlider(OWGUI.indentedBox(cinterBox), self, "classIntervals", None, 2, 10, callback=[self.clearLineEditFocus, self.classMethodChanged])
        hbox = OWGUI.widgetBox(box, orientation = 0)
        OWGUI.appendRadioButton(box, self, "discretization", "Custom" + "  ", insertInto = hbox)
        self.classCustomLineEdit = OWGUI.LineEditWFocusOut(hbox, self.classCustomChanged, focusInCallback = self.classCustomSelected)
        self.connect(self.classCustomLineEdit, SIGNAL("returnPressed ()"), self.classCustomChanged)
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.separator(box)
        self.classIntervalsLabel = OWGUI.widgetLabel(box, "Current splits: ")
        OWGUI.separator(box)
        OWGUI.checkBox(box, self, "outputOriginalClass", "Output original class")
        OWGUI.widgetLabel(box, "(Widget always uses discretized class internally.)")

        OWGUI.separator(vbox)
        OWGUI.rubber(vbox)

        box = OWGUI.widgetBox(vbox, "Commit")
        applyButton = OWGUI.button(box, self, "Commit", callback = self.commit)
        autoApplyCB = OWGUI.checkBox(box, self, "autoApply", "Commit automatically", callback=[self.clearLineEditFocus])
        OWGUI.setStopper(self, applyButton, autoApplyCB, "dataChanged", self.commit)

        OWGUI.separator(self.mainHBox, width=25)
        self.mainIABox =  OWGUI.widgetBox(self.mainHBox, "Individual attribute settings")
        self.layout.addWidget(self.mainVBox)
        self.mainBox = OWGUI.widgetBox(self.mainIABox, orientation=0)
        OWGUI.separator(self.mainIABox)#, height=30)
        graphBox = OWGUI.widgetBox(self.mainIABox, "", orientation=0)
        self.needsDiscrete.append(graphBox)
        graphOptBox = OWGUI.widgetBox(graphBox)
        OWGUI.separator(graphBox, width=10)
        self.graph = DiscGraph(self, graphBox)

        graphOptBox.setSpacing(4)
        box = OWGUI.widgetBox(graphOptBox, "Split gain measure", addSpace=True)
        self.measureCombo=OWGUI.comboBox(box, self, "measure", orientation=0, items=[e[0] for e in self.measures], callback=[self.clearLineEditFocus, self.graph.computeBaseScore, self.graph.plotBaseCurve])
        OWGUI.checkBox(box, self, "showBaseLine", "Show discretization gain", callback=[self.clearLineEditFocus, self.graph.plotBaseCurve])
        OWGUI.checkBox(box, self, "showLookaheadLine", "Show lookahead gain", callback=self.clearLineEditFocus)

        box = OWGUI.widgetBox(graphOptBox, "Target class", addSpace=True)
        self.targetCombo=OWGUI.comboBox(box, self, "targetClass", orientation=0, callback=[self.clearLineEditFocus, self.graph.targetClassChanged])
        OWGUI.checkBox(box, self, "showTargetClassProb", "Show target class probability", callback=[self.clearLineEditFocus, self.graph.plotProbCurve])
        OWGUI.checkBox(box, self, "showRug", "Show rug", callback=[self.clearLineEditFocus, self.graph.plotRug])

        box = OWGUI.widgetBox(graphOptBox, "Editing", addSpace=True)
        OWGUI.checkBox(box, self, "snap", "Snap to grid", callback=[self.clearLineEditFocus])
        syncCB = OWGUI.checkBox(box, self, "autoSynchronize", "Apply on the fly", callback=self.clearLineEditFocus)
        syncButton = OWGUI.button(box, self, "Apply", callback = self.synchronizePressed)
        OWGUI.setStopper(self, syncButton, syncCB, "pointsChanged", self.synchronize)
        OWGUI.rubber(graphOptBox)

        attrListBox = QVBox(self.mainBox)
        self.attrList = QListBox(attrListBox)
        self.attrList.setFixedWidth(300)

        self.defaultMethodChanged()
        self.connect(self.attrList, SIGNAL("highlighted ( int )"), self.individualSelected)

        OWGUI.separator(self.mainBox, width=10)
        box = OWGUI.radioButtonsInBox(QHButtonGroup(self.mainBox), self, "indiDiscretization", [], callback=[self.clearLineEditFocus, self.indiMethodChanged])
        hbbox = OWGUI.widgetBox(box)
        hbbox.setSpacing(4)
        for meth in self.indiDiscretizationMethods:
            OWGUI.appendRadioButton(box, self, "discretization", meth, insertInto = hbbox)
        self.needsDiscrete.append(box.buttons[2])
        self.indiInterBox = OWGUI.widgetBox(hbbox)
        OWGUI.widgetLabel(self.indiInterBox, "Number of intervals")
        OWGUI.separator(self.indiInterBox, height=4)
        self.indiIntervalSlider=OWGUI.hSlider(OWGUI.indentedBox(self.indiInterBox), self, "indiIntervals", None, 2, 10, callback=[self.clearLineEditFocus, self.indiMethodChanged])
        OWGUI.rubber(self.indiInterBox)
        OWGUI.separator(box)
        hbbox = OWGUI.widgetBox(box)
        for i in range(3):
            hbox = OWGUI.widgetBox(hbbox, orientation = 0)
            OWGUI.appendRadioButton(box, self, "discretization", "Custom %i" % (i+1) + " ", insertInto = hbox)
            le = OWGUI.LineEditWFocusOut(hbox, lambda w=i: self.customChanged(w), focusInCallback = lambda w=i: self.customSelected(w))
            self.connect(le, SIGNAL("returnPressed ()"), lambda w=i: self.customChanged(w))
            le.setFixedWidth(110)
            self.customLineEdits.append(le)
            OWGUI.button(hbox, self, "CC", width=30, callback = lambda w=i: self.copyToCustom(w))
        OWGUI.rubber(hbbox)

        self.controlArea.setFixedWidth(1)

        self.contAttrIcon =  self.createAttributeIconDict()[orange.VarTypes.Continuous]


    def cdata(self, data=None):
        self.closeContext()
        
        self.indiData = []
        self.attrList.clear()
        for le in self.customLineEdits:
            le.clear()
        self.indiDiscretization = 0

        self.originalData = data
        haveClass = bool(data and data.domain.classVar)
        continuousClass = haveClass and data.domain.classVar.varType == orange.VarTypes.Continuous

        if continuousClass:
            self.discretizeClass()
        else:
            self.data = self.originalData

        for c in self.needsDiscrete:
            c.setEnabled(haveClass)
        
        self.classDiscBox.setEnabled(not data or continuousClass)
        if self.data:
            domain = self.data.domain
            self.continuousIndices = [i for i, attr in enumerate(domain.attributes) if attr.varType == orange.VarTypes.Continuous]

            for i, attr in enumerate(domain.attributes):
                if attr.varType == orange.VarTypes.Continuous:
                    self.attrList.insertItem(ListItemWithLabel(self.contAttrIcon, attr.name, self.attrList.count(), self))
                    self.indiData.append([0, 4, "", "", ""])
                else:
                    self.indiData.append(None)

            self.fillClassCombo()
            self.indiLabels = [""] * self.attrList.count()

            self.graph.setData(None, data)           
            self.openContext("", data)

            self.computeDiscretizers()
            self.attrList.setCurrentItem(self.selectedAttr)
        else:
            self.graph.setData(None, None)

        # Prevent entropy discretization with non-discrete class
        if not haveClass:
            if self.discretization == 1:
                self.discretization = 2
            if self.indiDiscretization == 2:
                self.indiDiscretization = 0
            for indiData in self.indiData:
                if indiData and indiData[0] == 2:
                    indiData[0] = 0
                    
#        self.graph.setData(self.data)
        self.send("Examples", self.data)

        self.makeConsistent()        


    def fillClassCombo(self):    
        domain = self.data.domain
        self.targetCombo.clear()
        for v in domain.classVar.values:
            self.targetCombo.insertItem(str(v))
        if self.targetClass<len(domain.classVar.values):
            self.targetCombo.setCurrentItem(self.targetClass)
        else:
            self.targetCombo.setCurrentItem(0)
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
            

                
    def individualSelected(self, i):
        if not self.data:
            return

        self.selectedAttr = i
        attrIndex = self.continuousIndices[i]        
        attr = self.data.domain[attrIndex]
        indiData = self.indiData[attrIndex]

        self.customSplits = indiData[2:]
        for le, cs in zip(self.customLineEdits, self.customSplits):
            le.setText(" ".join(cs))

        self.indiDiscretization, self.indiIntervals = indiData[:2]

        if self.data.domain.classVar:
            self.graph.setData(attr, self.data)
            if hasattr(self, "discretizers"):
                self.graph.setSplits(self.discretizers[attrIndex] and self.discretizers[attrIndex].getValueFrom.transformer.points or [])

    
    def computeDiscretizers(self):
        self.discretizers = []
        
        if not self.data:
            return

        self.discretizers = [None] * len(self.data.domain)
        for i, idx in enumerate(self.continuousIndices):
            self.computeDiscretizer(i, idx)

        self.commitIf()            


    def makeConsistent(self):
        self.interBox.setEnabled(self.discretization>=2)
        self.indiInterBox.setEnabled(self.indiDiscretization in [3, 4])

    
    def defaultMethodChanged(self):
        self.interBox.setEnabled(self.discretization>=2)

        if not self.data:
            return

        for i, idx in enumerate(self.continuousIndices):
            self.computeDiscretizer(i, idx, True)

        self.commitIf()            

    def classMethodChanged(self):
        self.discretizeClass()
        self.classChanged()
        attrIndex = self.continuousIndices[self.selectedAttr]
        self.graph.setData(self.data.domain[attrIndex], self.data)
        self.graph.setSplits(self.discretizers[attrIndex] and self.discretizers[attrIndex].getValueFrom.transformer.points or [])
        if self.targetClass > len(self.data.domain.classVar.values):
            self.targetClass = len(self.data.domain.classVar.values)-1


    def indiMethodChanged(self, dontSetACustom=False):
        i, idx = self.selectedAttr, self.continuousIndices[self.selectedAttr]
        self.indiData[idx][0] = self.indiDiscretization
        self.indiData[idx][1] = self.indiIntervals

        self.indiInterBox.setEnabled(self.indiDiscretization in [3, 4])
        if self.indiDiscretization and self.indiDiscretization - 4 != self.resetIndividuals:
            self.resetIndividuals = 4

        if not self.data:
            return

        which = self.indiDiscretization - 5
        if not dontSetACustom and which >= 0 and not self.customSplits[which]:
            attr = self.data.domain[idx]
            splitsTxt = self.indiData[idx][2+which] = [str(attr(x)) for x in self.graph.curCutPoints]
            self.customSplits[which] = " ".join(splitsTxt)
            self.customLineEdits[which].setText(" ".join(splitsTxt))
            self.computeDiscretizer(i, idx)
        else:
            self.computeDiscretizer(i, idx)

        self.commitIf()            


    def customSelected(self, which):
        if self.data and self.indiDiscretization != 5+which:
            self.indiDiscretization = 5 + which
            idx = self.continuousIndices[self.selectedAttr]
            attr = self.data.domain[idx]
            self.indiMethodChanged()

        
    def setAllIndividuals(self):
        self.clearLineEditFocus()
        method = self.resetIndividuals
        if method == 4:
            return
        if method:
            method += 4
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

        self.attrList.triggerUpdate(0)
        self.commitIf()


    def customChanged(self, which):
        if not self.data:
            return

        idx = self.continuousIndices[self.selectedAttr]
        le = self.customLineEdits[which]

        content = str(le.text()).replace(":", " ").replace(",", " ").replace("-", " ").split()
        content = dict.fromkeys(content).keys()  # remove duplicates (except 8.0, 8.000 ...)
        try:
            content.sort(lambda x,y:cmp(float(x), float(y)))
        except:
            content = str(le.text())

        le.setText(" ".join(content))
        self.customSplits[which] = content
        self.indiData[idx][which+2] = content

        self.indiData[idx][0] = self.indiDiscretization = 5 + which

        self.computeDiscretizer(self.selectedAttr, self.continuousIndices[self.selectedAttr])
        self.commitIf()
                

    def copyToCustom(self, which):
        self.clearLineEditFocus()
        if not self.data:
            return

        idx = self.continuousIndices[self.selectedAttr]

        if self.indiDiscretization >= 5:
            splits = str(self.customSplits[self.indiDiscretization-5])
            try:
                valid = bool([float(i) for i in self.customSplits[which]])
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

    
    shortDiscNames = ("", " (leave continuous)", " (entropy)", " (equal frequency)", " (equal width)", " (custom 1)", " (custom 2)", " (custom 3)")

    def computeDiscretizer(self, i, idx, onlyDefaults=False):
        attr = self.data.domain[idx]
        indiData = self.indiData[idx]

        discType, intervals = indiData[:2]
        discName = self.shortDiscNames[discType]

        defaultUsed = not discType

        if defaultUsed:
            discType = self.discretization+1
            intervals = self.intervals

        if discType >= 5:

            try:
                customs = [float(r) for r in indiData[discType-5+2]]
            except:
                customs = []
                
            if not customs:
                discType = self.discretization+1
                intervals = self.intervals
                discName = "%s ->%s)" % (self.shortDiscNames[indiData[0]][:-1], self.shortDiscNames[discType][2:-1])
                defaultUsed = True

        if onlyDefaults and not defaultUsed:
            return
        
        if discType == 1: # leave continuous
            discretizer = None
        elif discType == 2:
            discretizer = orange.EntropyDiscretization(attr, self.data)
        elif discType == 3:
            discretizer = orange.EquiNDiscretization(attr, self.data, numberOfIntervals = intervals)
        elif discType == 4:
            discretizer = orange.EquiDistDiscretization(attr, self.data, numberOfIntervals = intervals)
        else:
            discretizer = orange.IntervalDiscretizer(points = customs).constructVariable(attr)


        self.discretizers[idx] = discretizer
        
        discInts = discType!=1 and (": " + ", ".join([str(attr(x)) for x in discretizer.getValueFrom.transformer.points])) or ""
        self.indiLabels[i] = discInts + discName
                        
        self.attrList.triggerUpdate(0)

        if i == self.selectedAttr:
            self.graph.setSplits(discretizer and discretizer.getValueFrom.transformer.points or [])



    def discretizeClass(self):
        if self.originalData:
            discType = self.classDiscretization
            classVar = self.originalData.domain.classVar
            
            if discType == 2:
                try:
                    content = str(self.classCustomLineEdit.text()).replace(":", " ").replace(",", " ").replace("-", " ").split()
                    customs = dict.fromkeys([float(x) for x in content]).keys()  # remove duplicates (except 8.0, 8.000 ...)
                    customs.sort()
                except:
                    customs = []

                if not customs:
                    discType = 0

                print customs                
            if discType == 0:
                discretizer = orange.EquiNDiscretization(classVar, self.originalData, numberOfIntervals = self.classIntervals)
            elif discType == 1:
                discretizer = orange.EquiDistDiscretization(classVar, self.originalData, numberOfIntervals = self.classIntervals)
            else:
                discretizer = orange.IntervalDiscretizer(points = customs).constructVariable(classVar)

            self.data = orange.ExampleTable(orange.Domain(self.originalData.domain.attributes, discretizer), self.originalData)
            
            self.classIntervalsLabel.setText("Current splits: " + ", ".join([str(classVar(x)) for x in discretizer.getValueFrom.transformer.points]))
        

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
        slot = self.indiDiscretization - 5
        if slot < 0:
            for slot in range(3):
                if not self.customLineEdits[slot]:
                    break
            else:
                slot = 0
            self.indiDiscretization = slot + 5

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

        self.indiLabels[self.selectedAttr] = ": " + splitsTxt + self.shortDiscNames[-1]
        self.attrList.triggerUpdate(0)

        self.pointsChanged = False


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
                else:
                    newattrs.append(attr)

            if self.data.domain.classVar:
                if self.outputOriginalClass:
                    newattrs.append(self.originalData.domain.classVar)
                else:
                    newattrs.append(self.data.domain.classVar)

            self.send("Examples", self.data.select(newattrs))

        else:
            self.send("Example", None)

        dataChanged = False            


import sys
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWInteractiveDiscretization()
    app.setMainWidget(w)
    w.show()
#    d=orange.ExampleTable("../../doc/datasets/bridges.tab")
#    d=orange.ExampleTable("../../doc/datasets/auto-mpg.tab")
    d = orange.ExampleTable("../../doc/datasets/iris.tab")
    w.cdata(d)
    w.cdata(None)
    w.cdata(d)
    app.exec_loop()
    w.saveSettings()
