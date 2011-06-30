"""
<name>Distributions (Qt)</name>
<description>Displays attribute value distributions.</description>
<contact>Tomaz Curk</contact>
<icon>icons/Distribution.png</icon>
<priority>100</priority>
"""

#
# OWDistributions.py
# Shows data distributions, distribution of attribute values and distribution of classes for each attribute
#

from OWColorPalette import ColorPixmap, ColorPaletteGenerator
from OWWidget import *
from OWGraphQt import *
import OWGUI
import math
import statc

from orangegraph import Curve as CppCurve
from Graph.item import *

class distribErrorBarCurve(CppCurve, PlotItem):
    def __init__(self, text = None):
        self.items = []
        CppCurve.__init__(self,[], [])
        PlotItem.__init__(self)
        
    def set_graph_transform(self, transform):
        CppCurve.setGraphTransform(self, transform)
        
    def updateProperties(self):
        if self.style() != CppCurve.UserCurve:
            for i in self.items:
                self.scene().removeItem(i)
            self.items = []
            CppCurve.updateProperties(self)
            return
            
        t = self.graphTransform()
        d = self.data()
        n = len(d)/3
        m = len(self.items)
        if m < n:
            self.items.extend([QGraphicsPathItem(self) for i in range(n-m)])
        elif m > n:
            for i in range(n,m):
                self.scene().removeItem(self.items[i])
            self.items = self.items[:n]
        for i in range(n):
            p = QPainterPath()
            px, py1 = d[3*i]
            _, py2 = d[3*i+1]
            _, py3 = d[3*i+2]
            p.moveTo(px, py1)
            p.lineTo(px, py3)
            p.moveTo(pxl, py1)
            p.lineTo(pxr, py1)
            p.moveTo(pxl, py3)
            p.lineTo(pxr, py3)
            self.items[i].setPath(t.map(p))
            self.items[i].setPen(self.pen())

class OWDistributionGraphQt(OWGraph):
    def __init__(self, settingsWidget = None, parent = None, name = None):
        OWGraph.__init__(self, parent, name, axes = [xBottom, yLeft, yRight])
        self.parent = parent

        # initialize settings
        self.attributeName = ""
        self.variableContinuous = FALSE
        self.YLaxisTitle = "Frequency"

        self.numberOfBars = 5
        self.barSize = 50
        self.showContinuousClassGraph=1
        self.showProbabilities = 0
        self.showConfidenceIntervals = 0
        self.smoothLines = 0
        self.hdata = {}
        self.probGraphValues = []

        self.targetValue = None
        self.data = None
        self.visibleOutcomes = None

        self.settingsWidget = settingsWidget

        self.probCurveKey = self.addCurve(xBottom, yRight, 0)
        self.probCurveUpperCIKey = self.addCurve(xBottom, yRight, 0)
        self.probCurveLowerCIKey = self.addCurve(xBottom, yRight, 0)

        self.tooltipManager = TooltipManager(self)

    def addCurve(self, xAxis = xBottom, yAxis = yLeft, visible = 1):
        curve = distribErrorBarCurve('')
        curve.setVisible(visible)
        curve.setXAxis(xAxis)
        curve.setYAxis(yAxis)
        OWGraph.add_custom_curve(self, curve, enableLegend=0)
        return curve


    def sizeHint(self):
        return QSize(500, 500)

    def setVisibleOutcomes(self, outcomes):
        self.visibleOutcomes = outcomes

    def setTargetValue(self, target):
        self.targetValue = target
        self.refreshProbGraph()

    def setData(self, data, variable):
        self.data = data
        self.pureHistogram = not data or not data.domain.classVar or data.domain.classVar.varType!=orange.VarTypes.Discrete
        self.dataHasClass = data and data.domain.classVar
        self.dataHasDiscreteClass = self.dataHasClass and data.domain.classVar.varType == orange.VarTypes.Discrete
        if self.dataHasDiscreteClass:
            self.dc = orange.DomainContingency(self.data)
        self.setVariable(variable)

    def setVariable(self, variable):
        self.attributeName = variable
        if variable: self.setXaxisTitle(variable)
        else:        self.setXaxisTitle("")

        if not self.data: return
        
        if variable and self.data.domain[variable].varType == orange.VarTypes.Discrete and len(self.data.domain[variable].values) > 100:
            if QMessageBox.question(self, "Confirmation", "The attribute %s has %d values. Are you sure you want to visualize this attribute?" % (variable, len(self.data.domain[variable].values)), QMessageBox.Yes , QMessageBox.No | QMessageBox.Escape | QMessageBox.Default) == QMessageBox.No:
                self.clear()
                self.tips.removeAll()
                self.hdata = {}
                return

        if self.data.domain[self.attributeName].varType == orange.VarTypes.Continuous:
            self.variableContinuous = TRUE
        else:
            self.variableContinuous = FALSE

        if self.variableContinuous:
            self.setXlabels(None)
        else:
            labels = self.data.domain[self.attributeName].values.native()
            self.setXlabels(labels)
            self.setAxisScale(xBottom, -0.5, len(labels) - 0.5, 1)

        self.calcHistogramAndProbGraph()
        self.refreshVisibleOutcomes()


    def setNumberOfBars(self, n):
        self.numberOfBars = n

        if self.variableContinuous:
            self.calcHistogramAndProbGraph()
            self.refreshVisibleOutcomes()
            #self.replot()

    def setBarSize(self, n):
        self.barSize = n
        if not(self.variableContinuous):
            self.refreshVisibleOutcomes()
            self.replot()

    def calcPureHistogram(self):
        if self.data==None:
            return
        if self.variableContinuous:
            "Continuous variable, break data into self.NumberOfBars subintervals"
            "use orange.EquiDistDiscretization(numberOfIntervals)"
            equiDist = orange.EquiDistDiscretization(numberOfIntervals = self.numberOfBars)
            d_variable = equiDist(self.attributeName, self.data)
            d_data = self.data.select([d_variable])
            tmphdata = orange.Distribution(0, d_data)

            curPos = d_variable.getValueFrom.transformer.firstVal - d_variable.getValueFrom.transformer.step
            self.subIntervalStep = d_variable.getValueFrom.transformer.step
            self.hdata = {}
            for key in tmphdata.keys():
                self.hdata[curPos] = tmphdata[key]
                curPos += self.subIntervalStep
        else:
            "Discrete variable"
            self.hdata = orange.Distribution(self.attributeName, self.data) #self.dc[self.attributeName]

    def calcHistogramAndProbGraph(self):
        "Calculates the histogram."
        if self.data == None:
            return
        if self.pureHistogram:
            self.calcPureHistogram()
            return
        if self.variableContinuous:
            "Continuous variable, break data into self.NumberOfBars subintervals"
            "use orange.EquiDistDiscretization(numberOfIntervals)"
            equiDist = orange.EquiDistDiscretization(numberOfIntervals = self.numberOfBars)
            d_variable = equiDist(self.attributeName, self.data)
#            d_data = self.data.select([d_variable, self.data.domain.classVar])
#            tmphdata = orange.DomainContingency(d_data)[0]
#            dc = orange.DomainContingency(self.data) #!!!
            tmphdata = orange.ContingencyAttrClass(d_variable, self.data)
            try:
                g = orange.ConditionalProbabilityEstimatorConstructor_loess(self.dc[self.attributeName], nPoints=200) #!!!
                self.probGraphValues = [(x, ps, [(v>=0 and math.sqrt(v)*1.96 or 0.0) for v in ps.variances]) for (x, ps) in g.probabilities.items()]
            except:
                self.probGraphValues = []
            # print [ps.variances for (x, ps) in g.probabilities.items()]
            # calculate the weighted CI=math.sqrt(prob*(1-prob)/(0.0+self.sums[curcol])),
            # where self.sums[curcol] = g.probabilities.items()[example][1].cases

            # change the attribute value (which is discretized) into the subinterval start value
            # keep the same DomainContingency data
            curPos = d_variable.getValueFrom.transformer.firstVal - d_variable.getValueFrom.transformer.step
            self.subIntervalStep = d_variable.getValueFrom.transformer.step
            self.hdata = {}
            for key in tmphdata.keys():
                self.hdata[curPos] = tmphdata[key]
                curPos += self.subIntervalStep
        else:
            "Discrete variable"
            self.hdata = self.dc[self.attributeName]
            self.probGraphValues = []
            for (x, ds) in self.hdata.items():
                ps = []
                cis = []
                cases = ds.cases
                for d in ds:
                    if cases > 0:
                        p = d / cases
                        ci = math.sqrt(p * (1-p) / (0.0 + cases))
                    else:
                        p = 0
                        ci = 0
                    ps.append(p)
                    cis.append(ci)
                self.probGraphValues.append((x, ps, cis))

    def refreshPureVisibleOutcomes(self):
        if self.dataHasDiscreteClass:
            return
        keys=self.hdata.keys()
        if self.variableContinuous:
            keys.sort()
        self.clear()
        self.tips.removeAll()
        cn=0
        for key in keys:
            ckey = PolygonCurve(pen=QPen(Qt.black), brush=QBrush(Qt.gray))
            ckey.attach(self)
            if self.variableContinuous:
                ckey.setData([key, key + self.subIntervalStep, key + self.subIntervalStep, key],[0, 0, self.hdata[key], self.hdata[key]])
                ff="%."+str(self.data.domain[self.attributeName].numberOfDecimals+1)+"f"
                text = "N(%s in ("+ff+","+ff+"])=<b>%i</b>"
                text = text%(str(self.attributeName), key, key+self.subIntervalStep, self.hdata[key])
                self.tips.addToolTip(key+self.subIntervalStep/2.0, self.hdata[key]/2.0, text, self.subIntervalStep/2.0, self.hdata[key]/2.0)
            else:
                tmpx = cn - (self.barSize/2.0)/100.0
                tmpx2 = cn + (self.barSize/2.0)/100.0
                ckey.setData([tmpx, tmpx2, tmpx2, tmpx], [0, 0, self.hdata[key], self.hdata[key]])
                text = "N(%s=%s)=<b>%i</b>"%(str(self.attributeName), str(key), self.hdata[key])
                self.tips.addToolTip(cn, self.hdata[key]/2.0, text, (self.barSize/2.0)/100.0, self.hdata[key]/2.0)
                cn+=1

        if self.dataHasClass and not self.dataHasDiscreteClass and self.showContinuousClassGraph:
            self.enableYRaxis(1)
            self.setAxisAutoScale(yRight)
            self.setYRaxisTitle(str(self.data.domain.classVar.name))
            if self.variableContinuous:
                equiDist = orange.EquiDistDiscretization(numberOfIntervals = self.numberOfBars)
                d_variable = equiDist(self.attributeName, self.data)
                d_data=self.data.select([d_variable, self.data.domain.classVar])
                c=orange.ContingencyAttrClass(d_variable, d_data)
                XY=[(key+self.subIntervalStep/2.0, val.average()) for key, val in zip(keys, c.values()) if val.cases]
                XY=statc.loess(XY, 10, 4.0, 1)
            else:
                d_data=orange.ContingencyAttrClass(self.attributeName, self.data)
                XY=[(i, dist.average()) for i, dist in zip(range(len(d_data.values())), d_data.values()) if dist.cases]
            key = self.addCurve(xBottom, yRight)
            key.setData([a[0] for a in XY], [a[1] for a in XY])
            if self.variableContinuous:
                key.setPen(QPen(Qt.black))
            else:
                key.setColor(Qt.black)
                key.setSymbol(Diamond)
                key.setPointSize(7)
        else:
            self.enableYRaxis(0)
            self.setAxisScale(yRight, 0.0, 1.0, 0.1)

        self.probCurveKey = self.addCurve(xBottom, yRight)
        self.probCurveUpperCIKey = self.addCurve(xBottom, yRight)
        self.probCurveLowerCIKey = self.addCurve(xBottom, yRight)

        self.replot()

    def refreshVisibleOutcomes(self):
        if not self.data or (not self.visibleOutcomes and not self.pureHistogram): return
        self.tips.removeAll()
        if self.pureHistogram:
            self.refreshPureVisibleOutcomes()
            return
        self.enableYRaxis(0)
        self.setAxisScale(yRight, 0.0, 1.0, 0.1)
        self.setYRaxisTitle("")
        keys = self.hdata.keys()
        if self.variableContinuous:
            keys.sort()

        self.clear()

        currentBarsHeight = [0] * len(keys)
        for oi in range(len(self.visibleOutcomes)):
            if self.visibleOutcomes[oi] == 1:
                #for all bars insert curve and
                for cn, key in enumerate(keys):
                    subBarHeight = self.hdata[key][oi]
                    if not subBarHeight:
                        continue
                    ckey = PolygonCurve(pen = QPen(self.discPalette[oi]), brush = QBrush(self.discPalette[oi]))
                    ckey.attach(self)
                    if self.variableContinuous:
                        ckey.setData([key, key + self.subIntervalStep, key + self.subIntervalStep, key], [currentBarsHeight[cn], currentBarsHeight[cn], currentBarsHeight[cn] + subBarHeight, currentBarsHeight[cn] + subBarHeight])
                        ff = "%."+str(self.data.domain[self.attributeName].numberOfDecimals+1)+"f"
                        text = "N(%s=%s|%s in ("+ff+","+ff+"])=<b>%i</b><br>P(%s=%s|%s in ("+ff+","+ff+"])=<b>%.3f</b><br>"
                        text = text%(str(self.data.domain.classVar.name), str(self.data.domain.classVar[oi]), str(self.attributeName), key, key+self.subIntervalStep, subBarHeight,
                                     str(self.data.domain.classVar.name), str(self.data.domain.classVar[oi]), str(self.attributeName), key, key+self.subIntervalStep, float(subBarHeight/sum(self.hdata[key]))) #self.probGraphValues[cn][1][oi])
                        self.tips.addToolTip(key+self.subIntervalStep/2.0, currentBarsHeight[cn] + subBarHeight/2.0, text, self.subIntervalStep/2.0, subBarHeight/2.0)
                    else:
                        tmpx = cn - (self.barSize/2.0)/100.0
                        tmpx2 = cn + (self.barSize/2.0)/100.0
                        ckey.setData([tmpx, tmpx2, tmpx2, tmpx], [currentBarsHeight[cn], currentBarsHeight[cn], currentBarsHeight[cn] + subBarHeight, currentBarsHeight[cn] + subBarHeight])
                        text = "N(%s=%s|%s=%s)=<b>%i</b><br>P(%s=%s|%s=%s)=<b>%.3f</b>"
                        text = text%(str(self.data.domain.classVar.name), str(self.data.domain.classVar[oi]), str(self.attributeName), str(key), subBarHeight,
                                     str(self.data.domain.classVar.name), str(self.data.domain.classVar[oi]), str(self.attributeName), str(key), float(subBarHeight/sum(self.hdata[key])))
                        self.tips.addToolTip(cn, currentBarsHeight[cn]+subBarHeight/2.0, text, (self.barSize/2.0)/100.0, subBarHeight/2.0)
                    currentBarsHeight[cn] += subBarHeight

        self.probCurveKey = self.addCurve(xBottom, yRight)
        self.probCurveUpperCIKey = self.addCurve(xBottom, yRight)
        self.probCurveLowerCIKey = self.addCurve(xBottom, yRight)
        self.refreshProbGraph()

    def refreshProbGraph(self):
        if not self.data or self.targetValue == None: return
        if self.showProbabilities:
            self.enableYRaxis(1)
          #  self.setShowYRaxisTitle(self.showYRaxisTitle)
          #  self.setYRaxisTitle(self.YRaxisTitle)
            xs = []
            ups = []
            mps = []
            lps = []
            cn = 0.0
            for (x, ps, cis) in self.probGraphValues:
                if self.variableContinuous:
                    xs.append(x)
                    ups.append(ps[self.targetValue] + cis[self.targetValue])
                    mps.append(ps[self.targetValue] + 0.0)
                    lps.append(ps[self.targetValue] - cis[self.targetValue])
                else:
                    if self.showConfidenceIntervals:
                        xs.append(cn)
                        mps.append(ps[self.targetValue] + cis[self.targetValue])

                    xs.append(cn)
                    mps.append(ps[self.targetValue] + 0.0)

                    if self.showConfidenceIntervals:
                        xs.append(cn)
                        mps.append(ps[self.targetValue] - cis[self.targetValue])
                cn += 1.0

            ## (re)set the curves
            if self.variableContinuous:
                newSymbol = NoSymbol
            else:
                newSymbol = Diamond
                
            self.probCurveKey.setData(xs, mps)
            self.probCurveKey.setSymbol(newSymbol)

            if self.variableContinuous:
                self.probCurveKey.setStyle(CppCurve.Lines)
                if self.showConfidenceIntervals:
                    self.probCurveUpperCIKey.setData(xs, ups)
                    self.probCurveLowerCIKey.setData(xs, lps)
            else:
                if self.showConfidenceIntervals:
                    self.probCurveKey.setStyle(CppCurve.UserCurve)
                else:
                    self.probCurveKey.setStyle(CppCurve.Dots)
        else:
            self.enableYRaxis(0)
            self.setShowYRaxisTitle(0)

        def enableIfExists(curve, en):
            if curve:
                curve.setVisible(en)

        enableIfExists(self.probCurveKey, self.showProbabilities)
        enableIfExists(self.probCurveUpperCIKey, self.showConfidenceIntervals and self.showProbabilities)
        enableIfExists(self.probCurveLowerCIKey, self.showConfidenceIntervals and self.showProbabilities)
        self.replot()

class OWDistributionsQt(OWWidget):
    settingsList = ["numberOfBars", "barSize", "graph.showContinuousClassGraph", "showProbabilities", "showConfidenceIntervals", "smoothLines", "lineWidth", "showMainTitle", "showXaxisTitle", "showYaxisTitle", "showYPaxisTitle"]
    contextHandlers = {"": DomainContextHandler("", ["attribute", "targetValue", "visibleOutcomes", "mainTitle", "xaxisTitle", "yaxisTitle", "yPaxisTitle"], matchValues=DomainContextHandler.MatchValuesClass)}

    def __init__(self, parent=None, signalManager = None):
        "Constructor"
        OWWidget.__init__(self, parent, signalManager, "&Distributions (Qt)", TRUE)
        # settings
        self.numberOfBars = 5
        self.barSize = 50
        self.showContinuousClassGraph=1
        self.showProbabilities = 1
        self.showConfidenceIntervals = 0
        self.smoothLines = 0
        self.lineWidth = 1
        self.showMainTitle = 0
        self.showXaxisTitle = 1
        self.showYaxisTitle = 1
        self.showYPaxisTitle = 1

        self.attribute = ""
        self.targetValue = 0
        self.visibleOutcomes = []
        self.outcomes = []

        # tmp values
        self.mainTitle = ""
        self.xaxisTitle = ""
        self.yaxisTitle = "frequency"
        self.yPaxisTitle = ""

        # GUI
#        self.tabs = OWGUI.tabWidget(self.controlArea)
#        self.GeneralTab = OWGUI.createTabPage(self.tabs, "Main")
#        self.SettingsTab = OWGUI.createTabPage(self.tabs, "Settings")
        self.GeneralTab = self.SettingsTab = self.controlArea

        self.graph = OWDistributionGraphQt(self, self.mainArea)
        self.mainArea.layout().addWidget(self.graph)
        self.graph.setYRlabels(None)
        self.graph.setAxisScale(yRight, 0.0, 1.0, 0.1)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        
        self.loadSettings()

        self.barSize = 50

        # inputs
        # data and graph temp variables
        self.inputs = [("Examples", ExampleTable, self.setData, Default)]

        self.data = None
        self.outcomenames = []
        self.probGraphValues = []

        b = OWGUI.widgetBox(self.controlArea, "Variable", addSpace=True)
        self.variablesQCB = OWGUI.comboBox(b, self, "attribute", valueType = str, sendSelectedValue = True, callback=self.setVariable)
        OWGUI.widgetLabel(b, "Displayed outcomes")
        self.outcomesQLB = OWGUI.listBox(b, self, "visibleOutcomes", "outcomes", selectionMode = QListWidget.MultiSelection, callback = self.outcomeSelectionChange)

        # GUI connections
        # options dialog connections
#        b = OWGUI.widgetBox(self.SettingsTab, "Bars")
#        OWGUI.spin(b, self, "numberOfBars", label="Number of bars", min=5, max=60, step=5, callback=self.setNumberOfBars, callbackOnReturn=True)
#        self.numberOfBarsSlider = OWGUI.hSlider(self.SettingsTab, self, 'numberOfBars', box='Number of bars', minValue=5, maxValue=60, step=5, callback=self.setNumberOfBars, ticks=5)
#        self.numberOfBarsSlider.setTracking(0) # no change until the user stop dragging the slider

#        self.barSizeSlider = OWGUI.hSlider(self.SettingsTab, self, 'barSize', box="Bar size", minValue=30, maxValue=100, step=5, callback=self.setBarSize, ticks=10)
#        OWGUI.spin(b, self, "barSize", label="Bar size", min=30, max=100, step=5, callback=self.setBarSize, callbackOnReturn=True)

        box = OWGUI.widgetBox(self.SettingsTab, "General graph settings", addSpace=True)
        box.setMinimumWidth(180)
        box2 = OWGUI.widgetBox(box, orientation = "horizontal")
        OWGUI.checkBox(box2, self, 'showMainTitle', 'Main title', callback = self.setShowMainTitle)
        OWGUI.lineEdit(box2, self, 'mainTitle', callback = self.setMainTitle, enterPlaceholder=True)

        box3 = OWGUI.widgetBox(box, orientation = "horizontal")
        OWGUI.checkBox(box3, self, 'showXaxisTitle', 'X axis title', callback = self.setShowXaxisTitle)
        OWGUI.lineEdit(box3, self, 'xaxisTitle', callback = self.setXaxisTitle, enterPlaceholder=True)

        box4 = OWGUI.widgetBox(box, orientation = "horizontal")
        OWGUI.checkBox(box4, self, 'showYaxisTitle', 'Y axis title', callback = self.setShowYaxisTitle)
        OWGUI.lineEdit(box4, self, 'yaxisTitle', callback = self.setYaxisTitle, enterPlaceholder=True)

        OWGUI.checkBox(box, self, 'graph.showContinuousClassGraph', 'Show continuous class graph', callback=self.setShowContinuousClassGraph)
        OWGUI.spin(box, self, "numberOfBars", label="Number of bars", min=5, max=60, step=5, callback=self.setNumberOfBars, callbackOnReturn=True)

        box5 = OWGUI.widgetBox(self.SettingsTab, "Probability plot")
        self.showProb = OWGUI.checkBox(box5, self, 'showProbabilities', 'Show probabilities', callback = self.setShowProbabilities)
        self.targetQCB = OWGUI.comboBox(OWGUI.indentedBox(box5, sep=OWGUI.checkButtonOffsetHint(self.showProb)), self, "targetValue", label="Target value", valueType=int, callback=self.setTarget)

        box6 = OWGUI.widgetBox(box5, orientation = "horizontal")

        self.showYPaxisCheck = OWGUI.checkBox(box6, self, 'showYPaxisTitle', 'Axis title', callback = self.setShowYPaxisTitle)
        self.yPaxisEdit = OWGUI.lineEdit(box6, self, 'yPaxisTitle', callback = self.setYPaxisTitle, enterPlaceholder=True)
        self.confIntCheck = OWGUI.checkBox(box5, self, 'showConfidenceIntervals', 'Show confidence intervals', callback = self.setShowConfidenceIntervals)
        self.cbSmooth = OWGUI.checkBox(box5, self, 'smoothLines', 'Smooth probability lines', callback = self.setSmoothLines)
        self.showProb.disables = [self.showYPaxisCheck, self.yPaxisEdit, self.confIntCheck, self.targetQCB, self.cbSmooth]
        self.showProb.makeConsistent()


#        self.barSizeSlider = OWGUI.hSlider(box5, self, 'lineWidth', box='Line width', minValue=1, maxValue=9, step=1, callback=self.setLineWidth, ticks=1)
        
        OWGUI.rubber(self.SettingsTab)

        #add controls to self.controlArea widget

        self.icons = self.createAttributeIconDict()

        self.graph.numberOfBars = self.numberOfBars
        self.graph.barSize = self.barSize
        self.graph.setShowMainTitle(self.showMainTitle)
        self.graph.setShowXaxisTitle(self.showXaxisTitle)
        self.graph.setShowYLaxisTitle(self.showYaxisTitle)
        self.graph.setShowYRaxisTitle(self.showYPaxisTitle)
        self.graph.setMainTitle(self.mainTitle)
        self.graph.setXaxisTitle(self.xaxisTitle)
        self.graph.setYLaxisTitle(self.yaxisTitle)
        self.graph.setYRaxisTitle(self.yPaxisTitle)
        self.graph.showProbabilities = self.showProbabilities
        self.graph.showConfidenceIntervals = self.showConfidenceIntervals
        self.graph.smoothLines = self.smoothLines
        self.graph.lineWidth = self.lineWidth
        #self.graph.variableContinuous = self.VariableContinuous
        self.graph.targetValue = self.targetValue

    def sendReport(self):
        self.startReport("%s [%s: %s]" % (self.windowTitle(), self.attribute, self.targetValue))
        self.reportSettings("Visualized attribute",
                            [("Attribute", self.attribute),
                             ("Target class", self.targetValue)])
        self.reportRaw("<br/>")
        self.reportImage(self.graph.saveToFileDirect, QSize(600, 400))
        
    def setShowMainTitle(self):
        self.graph.setShowMainTitle(self.showMainTitle)

    def setMainTitle(self):
        self.graph.setMainTitle(self.mainTitle)

    def setShowXaxisTitle(self):
        self.graph.setShowXaxisTitle(self.showXaxisTitle)

    def setXaxisTitle(self):
        self.graph.setXaxisTitle(self.xaxisTitle)

    def setShowYaxisTitle(self):
        self.graph.setShowYLaxisTitle(self.showYaxisTitle)

    def setYaxisTitle(self):
        self.graph.setYLaxisTitle(self.yaxisTitle)

    def setShowYPaxisTitle(self):
        self.graph.setShowYRaxisTitle(self.showYPaxisTitle)

    def setYPaxisTitle(self):
        self.graph.setYRaxisTitle(self.yPaxisTitle)

    def setBarSize(self):
        self.graph.setBarSize(self.barSize)

    # Sets whether the probabilities are drawn or not
    def setShowProbabilities(self):
        self.graph.showProbabilities = self.showProbabilities
        self.graph.refreshProbGraph()
        self.graph.replot()

    def setShowContinuousClassGraph(self):
        self.graph.refreshPureVisibleOutcomes()

    #Sets the number of bars for histograms of continuous variables
    def setNumberOfBars(self):
        self.graph.setNumberOfBars(self.numberOfBars)

    # sets the line smoothing on and off
    def setSmoothLines(self):
        #self.SmoothLines = n
        #self.updateGraphSettings()
        pass

    # Sets the line thickness for probability
    def setLineWidth(self):
        #self.LineWidth = n
        #self.updateGraphSettings()
        pass

    # Sets whether the confidence intervals are shown
    def setShowConfidenceIntervals(self):
        self.graph.showConfidenceIntervals = self.showConfidenceIntervals
        #self.updateGraphSettings()
        self.graph.refreshProbGraph()
        self.graph.replot()

    def setTarget(self, *t):
        if t:
            self.targetValue = t[0]
        self.graph.setTargetValue(self.targetValue)

    def target(self, targetValue):
        self.targetValue = targetValue
        #self.updateGraphSettings()
        self.graph.refreshProbGraph()
        self.graph.replot()
        outcomeName = ""
        if self.data and self.data.domain.classVar:
            self.setYPaxisTitle("P( " + self.data.domain.classVar.name + " = " + targetValue + " )")

    def setData(self, data):
        self.closeContext()

        if data == None:
            self.variablesQCB.clear()
            self.targetQCB.clear()
            self.outcomes = []

            self.graph.setXlabels(None)
            self.graph.setYLlabels(None)
            self.graph.setShowYRaxisTitle(0)
            self.graph.setVisibleOutcomes(None)
            self.graph.setData(None, None)
            self.data = None
            return
        self.dataHasClass = bool(data.domain.classVar)
        if self.dataHasClass:
            self.dataHasDiscreteClass = data.domain.classVar.varType != orange.VarTypes.Continuous

        sameDomain = data and self.data and data.domain == self.data.domain

        if self.dataHasClass and self.dataHasDiscreteClass:
            self.data = orange.Preprocessor_dropMissingClasses(data)
        else:
            self.data = data

        if sameDomain:
            self.openContext("", self.data)
            self.graph.setData(self.data, self.graph.attributeName)

        else:
            self.graph.setData(None, None)
            self.graph.setTargetValue(None)
            self.graph.setVisibleOutcomes(None)
            # set targets
            self.targetQCB.clear()
            if self.data.domain.classVar and self.data.domain.classVar.varType == orange.VarTypes.Discrete:
                self.targetQCB.addItems([val for val in self.data.domain.classVar.values])
                self.setTarget(0)

            # set variable combo box
            self.variablesQCB.clear()
            variables = []
            for attr in self.data.domain.attributes:
                if attr.varType in [orange.VarTypes.Discrete, orange.VarTypes.Continuous]:
                    self.variablesQCB.addItem(self.icons[attr.varType], attr.name)
                    variables.append(attr)

            if self.data and variables:
                self.attribute = variables[0].name
                self.graph.setData(self.data, variables[0].name) # pick first variable
                #self.setVariable()

            self.targetValue = 0  # self.data.domain.classVar.values.index(str(targetVal))
            if self.dataHasClass and self.dataHasDiscreteClass:
                self.graph.setTargetValue(self.targetValue) #str(self.data.domain.classVar.values[0])) # pick first target
                self.setOutcomeNames(self.data.domain.classVar.values.native())
            else:
               self.setOutcomeNames([])

            self.openContext("", self.data)
            if self.data and variables:
                self.setVariable()

        for f in [self.setMainTitle, self.setTarget, self.setXaxisTitle, self.setYaxisTitle, self.setYPaxisTitle, self.outcomeSelectionChange]:
            f()


    def setOutcomeNames(self, list):
        "Sets the outcome target names."
        colors = ColorPaletteGenerator()
        self.outcomes = [(ColorPixmap(c), l) for c, l in zip(colors, list)]
        self.visibleOutcomes = range(len(list))

    def outcomeSelectionChange(self):
        "Sets which outcome values are represented in the graph."
        "Reacts to changes in outcome selection."
        self.graph.visibleOutcomes = [i in self.visibleOutcomes for i in range(self.outcomesQLB.count())]
        self.graph.refreshVisibleOutcomes()
        #self.graph.replot()
        #self.repaint()

    def setVariable(self):
        self.graph.setVariable(self.attribute)
        self.graph.refreshVisibleOutcomes()
        self.xaxisTitle = str(self.attribute)
        self.repaint()


if __name__ == "__main__":
    a = QApplication(sys.argv)
    owd = OWDistributionsQt()
##    a.setMainWidget(owd)
##    from pywin.debugger import set_trace
##    set_trace()
    owd.show()
    data=orange.ExampleTable("../../doc/datasets/housing.tab")
    owd.setData(data)
    a.exec_()
    owd.saveSettings()
