"""
<name>Distributions</name>
<description>Widget for comparing distributions of two datasets with same domain and different
examples.</description>
<category>Standard Visualizations</category>
<icon>icons/Distribution.png</icon>
<priority>1000</priority>
"""

#
# OWDistributions.py
# Shows data distributions, distribution of attribute values and distribution of classes for each attribute
#

from OWTools import *
from OWWidget import *
from OWVisGraph import *
import OWGUI
import math

class distribErrorBarQwtPlotCurve(QwtPlotCurve):
    def __init__(self, parent = None, text = None):
        QwtPlotCurve.__init__(self, parent, text)

    def draw(self, p, xMap, yMap, f, t):
        self.setPen( self.symbol().pen() )
        p.setPen( self.symbol().pen() )
        if self.style() == QwtCurve.UserCurve:
            p.setBackgroundMode(Qt.OpaqueMode)
            if t < 0: t = self.dataSize() - 1
            if divmod(f, 3)[1] != 0: f -= f % 3
            if divmod(t, 3)[1] == 0:  t += 1
            for i in range(f, t+1, 3):
                px = xMap.transform(self.x(i))
                pxl = xMap.transform(self.x(i) - 0.1)
                pxr = xMap.transform(self.x(i) + 0.1)
                py1 = yMap.transform(self.y(i + 0))
                py2 = yMap.transform(self.y(i + 1))
                py3 = yMap.transform(self.y(i + 2))
                p.drawLine(px, py1, px, py3)
                p.drawLine(pxl, py1, pxr, py1)
                p.drawLine(pxl, py3, pxr, py3)
                self.symbol().draw(p, px, py2)
        else:
            QwtPlotCurve.draw(self, p, xMap, yMap, f, t)


class OWDistributionGraph(OWVisGraph):
    def __init__(self, settingsWidget = None, parent = None, name = None):
        OWVisGraph.__init__(self, parent, name)
        self.parent = parent
        
        # initialize settings
        self.attributeName = ""
        self.variableContinuous = FALSE
        self.YLaxisTitle = "Frequency"
        
        self.numberOfBars = 5
        self.barSize = 50
        self.showProbabilities = 0
        self.showConfidenceIntervals = 0
        self.smoothLines = 0
        self.hdata = {}
        self.probGraphValues = []
        
        self.targetValue = None
        self.data = None
        self.visibleOutcomes = None

        self.settingsWidget = settingsWidget

        curve = distribErrorBarQwtPlotCurve(self, '')
        self.probCurveKey = self.insertCurve(curve)
        self.setCurveXAxis(self.probCurveKey, QwtPlot.xBottom)
        self.setCurveYAxis(self.probCurveKey, QwtPlot.yRight)

        self.probCurveUpperCIKey = self.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.probCurveLowerCIKey = self.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.curve(self.probCurveKey).setEnabled(FALSE)
        self.curve(self.probCurveUpperCIKey).setEnabled(FALSE)
        self.curve(self.probCurveLowerCIKey).setEnabled(FALSE)

    def sizeHint(self):
        return QSize(500, 500)

    def setVisibleOutcomes(self, outcomes):
        self.visibleOutcomes = outcomes

    def setTargetValue(self, target):
        self.targetValue = target
        self.refreshProbGraph()

    def setData(self, data, variable):
        self.data = data
        if data: self.dc = orange.DomainContingency(self.data)
        self.setVariable(variable)

    def setVariable(self, variable):
        self.attributeName = variable
        if variable: self.setXaxisTitle(variable)
        else:        self.setXaxisTitle("")

        if not self.data: return
        
        if self.data.domain[self.attributeName].varType == orange.VarTypes.Continuous:
            self.variableContinuous = TRUE
        else: self.variableContinuous = FALSE

        self.calcHistogramAndProbGraph()

        if self.variableContinuous:
            self.setXlabels(None)
        else:
            labels = self.data.domain[self.attributeName].values.native()
            self.setXlabels(labels)
            self.setAxisScale(QwtPlot.xBottom, -0.5, len(labels) - 0.5, 1)

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
            #self.replot()
            self.repaint()
        
    def calcHistogramAndProbGraph(self):
        "Calculates the histogram."
        if self.data == None:
            return

        if self.variableContinuous:
            "Continuous variable, break data into self.NumberOfBars subintervals"
            "use orange.EquiDistDiscretization(numberOfIntervals)"
            equiDist = orange.EquiDistDiscretization(numberOfIntervals = self.numberOfBars)
            d_variable = equiDist(self.attributeName, self.data)
            d_data = self.data.select([d_variable, self.data.domain.classVar])
            tmphdata = orange.DomainContingency(d_data)[0]
            dc = orange.DomainContingency(self.data) #!!!
            g = orange.ConditionalProbabilityEstimatorConstructor_loess(dc[self.attributeName]) #!!!
            # print [ps.variances for (x, ps) in g.probabilities.items()]
            self.probGraphValues = [(x, ps, [math.sqrt(v)*1.96 for v in ps.variances]) for (x, ps) in g.probabilities.items()]
            # calculate the weighted CI=math.sqrt(prob*(1-prob)/(0.0+self.sums[curcol])),
            # where self.sums[curcol] = g.probabilities.items()[example][1].cases

            # change the attribute value (which is discretized) into the subinterval start value
            # keep the same DomainContingency data
            curPos = d_variable.getValueFrom.transformer.firstVal
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
                self.probGraphValues.append( (x, ps, cis) )

    def refreshVisibleOutcomes(self):
        if not self.data or not self.visibleOutcomes: return
        keys = self.hdata.keys()
        if self.variableContinuous:
            keys.sort()

        self.removeCurves()

        currentBarsHeight = [0] * len(keys)
        colors = ColorPaletteHSV(len(self.visibleOutcomes))
        for oi in range(len(self.visibleOutcomes)):
            if self.visibleOutcomes[oi] == 1:
                #for all bars insert curve and
                cn = 0
                for key in keys:
                    subBarHeight = self.hdata[key][oi]
                    curve = subBarQwtPlotCurve(self)
                    curve.color = colors.getColor(oi)
                    ckey = self.insertCurve(curve)
                    self.setCurveStyle(ckey, QwtCurve.UserCurve)
                    if self.variableContinuous:
                        self.setCurveData(ckey, [key, key + self.subIntervalStep], [currentBarsHeight[cn], currentBarsHeight[cn] + subBarHeight])
                    else:
                        tmpx = cn - (self.barSize/2.0)/100.0
                        tmpx2 = cn + (self.barSize/2.0)/100.0
                        self.setCurveData(ckey, [tmpx, tmpx2], [currentBarsHeight[cn], currentBarsHeight[cn] + subBarHeight])
                    currentBarsHeight[cn] += subBarHeight
                    cn += 1

        curve = distribErrorBarQwtPlotCurve(self, '')
        self.probCurveKey = self.insertCurve(curve)
        self.setCurveXAxis(self.probCurveKey, QwtPlot.xBottom)
        self.setCurveYAxis(self.probCurveKey, QwtPlot.yRight)

        self.probCurveUpperCIKey = self.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.probCurveLowerCIKey = self.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.refreshProbGraph()

    def refreshProbGraph(self):
        if not self.data or self.targetValue == None: return
        if self.showProbabilities:
            self.enableYRaxis(1)
            self.setShowYRaxisTitle(self.showYRaxisTitle)
            self.setYRaxisTitle(self.YRaxisTitle)
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
                newSymbol = QwtSymbol(QwtSymbol.None, QBrush(Qt.color0), QPen(Qt.black, 2), QSize(0,0))
            else:
                newSymbol = QwtSymbol(QwtSymbol.Diamond, QBrush(Qt.color0), QPen(Qt.black, 2), QSize(7,7))

            self.setCurveData(self.probCurveKey, xs, mps)
            self.setCurveSymbol(self.probCurveKey, newSymbol)

            if self.variableContinuous:
                self.setCurveStyle(self.probCurveKey, QwtCurve.Lines)
                if self.showConfidenceIntervals:
                    self.setCurveData(self.probCurveUpperCIKey, xs, ups)
                    self.setCurveData(self.probCurveLowerCIKey, xs, lps)
            else:
                if self.showConfidenceIntervals:
                    self.setCurveStyle(self.probCurveKey, QwtCurve.UserCurve)
                else:
                    self.setCurveStyle(self.probCurveKey, QwtCurve.Dots)
        else:
            self.enableYRaxis(0)
            self.setShowYRaxisTitle(0)

        self.curve(self.probCurveKey).setEnabled(self.showProbabilities)
        self.curve(self.probCurveUpperCIKey).setEnabled(self.showConfidenceIntervals and self.showProbabilities)
        self.curve(self.probCurveLowerCIKey).setEnabled(self.showConfidenceIntervals and self.showProbabilities)
        self.curve(self.probCurveKey).itemChanged()
        self.curve(self.probCurveUpperCIKey).itemChanged()
        self.curve(self.probCurveLowerCIKey).itemChanged()
        self.repaint()

        
        
class OWDistributions(OWWidget):
    settingsList = ["numberOfBars", "barSize", "showProbabilities", "showConfidenceIntervals", "smoothLines", "lineWidth", "showMainTitle", "showXaxisTitle", "showYaxisTitle", "showYPaxisTitle"]

    def __init__(self,parent=None):
        "Constructor"
        OWWidget.__init__(self,
        parent,
        "&Distributions",
        """The Distribution Widget is an Orange Widget
that shows data distributions, 
distribution of attribute values and
distribution of classes for each attribute.
        """,
        FALSE,
        TRUE, icon = "Distribution.png")
        # settings
        self.numberOfBars = 5
        self.barSize = 50
        self.showProbabilities = 0
        self.showConfidenceIntervals = 0
        self.smoothLines = 0
        self.lineWidth = 1
        self.showMainTitle = 0
        self.showXaxisTitle = 1
        self.showYaxisTitle = 1
        self.showYPaxisTitle = 1

        #load settings
        self.loadSettings()

        # tmp values
        self.mainTitle = ""
        self.xaxisTitle = ""
        self.yaxisTitle = "frequency"
        self.yPaxisTitle = ""

        self.loadSettings()

        # GUI
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")
        
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWDistributionGraph(self, self.mainArea)
        self.graph.setYRlabels(None)
        self.graph.setAxisScale(QwtPlot.yRight, 0.0, 1.0, 0.1)
        self.box.addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # inputs
        # data and graph temp variables
        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1), ("Target Class value", int, self.target, 1)]
        
        self.data = None
        self.targetValue = 0
        self.visibleOutcomes = []
        self.outcomenames = []
        self.probGraphValues = []
       

        # GUI connections
        # options dialog connections
        self.numberOfBarsSlider = OWGUI.hSlider(self.SettingsTab, self, 'numberOfBars', box='Number of Bars', minValue=5, maxValue=60, step=5, callback=self.setNumberOfBars, ticks=5)
        self.numberOfBarsSlider.setTracking(0) # no change until the user stop dragging the slider

        self.barSizeSlider = OWGUI.hSlider(self.SettingsTab, self, 'barSize', box=' Bar Size ', minValue=30, maxValue=100, step=5, callback=self.setBarSize, ticks=10)

        box = OWGUI.widgetBox(self.SettingsTab, " General graph settings ")
        box.setMinimumWidth(170)
        box2 = OWGUI.widgetBox(box, orientation = "horizontal")
        OWGUI.checkBox(box2, self, 'showMainTitle', 'Show Main Title', callback = self.setShowMainTitle)
        OWGUI.lineEdit(box2, self, 'mainTitle', callback = self.setMainTitle)

        box3 = OWGUI.widgetBox(box, orientation = "horizontal")
        OWGUI.checkBox(box3, self, 'showXaxisTitle', 'Show X axis title', callback = self.setShowXaxisTitle)
        OWGUI.lineEdit(box3, self, 'xaxisTitle', callback = self.setXaxisTitle)

        box4 = OWGUI.widgetBox(box, orientation = "horizontal")
        OWGUI.checkBox(box4, self, 'showYaxisTitle', 'Show Y axis title', callback = self.setShowYaxisTitle)
        OWGUI.lineEdit(box4, self, 'yaxisTitle', callback = self.setYaxisTitle)
        
        box5 = OWGUI.widgetBox(self.SettingsTab, " Probability graph ")
        self.showProb = OWGUI.checkBox(box5, self, 'showProbabilities', ' Show Probabilities ', callback = self.setShowProbabilities)

        box6 = OWGUI.widgetBox(box5, orientation = "horizontal")

        self.showYPaxisCheck = OWGUI.checkBox(box6, self, 'showYPaxisTitle', 'Show Axis Title', callback = self.setShowYPaxisTitle)
        self.yPaxisEdit = OWGUI.lineEdit(box6, self, 'yPaxisTitle', callback = self.setYPaxisTitle)
        self.confIntCheck = OWGUI.checkBox(box5, self, 'showConfidenceIntervals', 'Show Confidence Intervals', callback = self.setShowConfidenceIntervals)
        self.showProb.disables = [self.showYPaxisCheck, self.yPaxisEdit, self.confIntCheck]
        self.showProb.makeConsistent()
        
        OWGUI.checkBox(box5, self, 'smoothLines', 'Smooth probability lines', callback = self.setSmoothLines)

        self.barSizeSlider = OWGUI.hSlider(box5, self, 'lineWidth', box=' Line width ', minValue=1, maxValue=9, step=1, callback=self.setLineWidth, ticks=1)
        
        #add controls to self.controlArea widget
        self.selvar = OWGUI.widgetBox(self.GeneralTab, " Variable ")
        self.target = OWGUI.widgetBox(self.GeneralTab, " Target value ")
        self.selout = OWGUI.widgetBox(self.GeneralTab, " Outcomes ")
        self.variablesQCB = QComboBox(self.selvar)
        self.targetQCB = QComboBox(self.target)
        self.outcomesQLB = QListBox(self.selout)
        self.outcomesQLB.setSelectionMode(QListBox.Multi)
        #connect controls to appropriate functions
        self.connect(self.variablesQCB, SIGNAL('activated (const QString &)'), self.setVariable)
        self.connect(self.targetQCB, SIGNAL('activated (const QString &)'), self.setTarget)
        self.connect(self.outcomesQLB, SIGNAL("selectionChanged()"), self.outcomeSelectionChange)
        
        self.activateLoadedSettings()

    def activateLoadedSettings(self):
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

    def setShowMainTitle(self):
        self.graph.setShowMainTitle(self.showMainTitle)

    def setMainTitle(self):
        self.graph.setMainTitle(self.mainTitle)

    def setShowXaxisTitle(self):
        self.graph.setShowXaxisTitle(self.showXaxisTitle)

    def setXaxisTitle(self):
        self.graph.setXaxisTitle(self.xaxisTitle)

    def setShowYaxisTitle(self):
        self.graph.setShowYLaxisTitle(self.showYaxisTitle )

    def setYaxisTitle(self):
        self.graph.setYLaxisTitle(self.yaxisTitle )

    def setShowYPaxisTitle(self):
        self.graph.setShowYRaxisTitle(self.showYPaxisTitle)

    def setYPaxisTitle(self):
        self.graph.setYRaxisTitle(self.yPaxisTitle)

    def setBarSize(self):
        self.graph.setBarSize(self.barSize )

    # Sets whether the probabilities are drawn or not
    def setShowProbabilities(self):
        self.graph.showProbabilities = self.showProbabilities 
        self.graph.refreshProbGraph()
        #self.graph.replot()
        self.repaint()

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
        #self.graph.replot()

    def setTarget(self, targetVal):
        self.targetValue = self.data.domain.classVar.values.index(str(targetVal))
        self.graph.setTargetValue(self.targetValue)


    def target(self, targetValue):
        self.targetValue = targetValue
        #self.updateGraphSettings()
        self.graph.refreshProbGraph()
        outcomeName = ""
        if self.data and self.data.domain.classVar:
            self.setYPaxisTitle("P( " + self.data.domain.classVar.name + " = " + targetValue + " )")

    def cdata(self, data):
        if data == None:
            self.setVariablesComboBox([])
            self.setOutcomeNames([])
            self.targetQCB.clear()
            self.graph.setXlabels(None)
            self.graph.setYLlabels(None)
            self.graph.setShowYRaxisTitle(0)
            self.graph.setVisibleOutcomes(None)
            self.graph.setData(None, None)
            self.data = None
            return

        self.data = orange.Preprocessor_dropMissingClasses(data)
        self.graph.setData(None, None)
        self.graph.setTargetValue(None)
        self.graph.setVisibleOutcomes(None)
        
        names = [attr.name for attr in self.data.domain.attributes]
        if self.graph.attributeName not in names:
            self.graph.attributeName = names[0]

        self.targetQCB.clear()
        if self.data.domain.classVar and self.data.domain.classVar.varType == orange.VarTypes.Discrete:
            for val in self.data.domain.classVar.values:
                self.targetQCB.insertItem(val)
            self.setTarget(self.data.domain.classVar.values[0])
                
        self.setVariablesComboBox(names, names.index(self.graph.attributeName))
        self.setOutcomeNames([])
        if self.data.domain.classVar:
            self.setOutcomeNames(self.data.domain.classVar.values.native())


    def setVariablesComboBox(self, list, defaultItem = 0):
        "Set the variables with the suplied list."
        self.variablesQCB.clear()
        for i in list:    self.variablesQCB.insertItem(i)
        if len(list) > 0:
            self.graph.setData(self.data, list[defaultItem])
            self.variablesQCB.setCurrentItem(defaultItem)
            self.setVariable(self.variablesQCB.text(defaultItem))


    def setOutcomeNames(self, list):
        "Sets the outcome target names."
        self.outcomesQLB.clear()
        colors = ColorPaletteHSV(len(list))
        for i in range(len(list)):
            self.outcomesQLB.insertItem(ColorPixmap(colors.getColor(i)), list[i])
        self.outcomesQLB.selectAll(TRUE)

    def outcomeSelectionChange(self):
        "Sets which outcome values are represented in the graph."
        "Reacts to changes in outcome selection."
        visibleOutcomes = []
        for i in range(self.outcomesQLB.numRows()):
            visibleOutcomes.append(self.outcomesQLB.isSelected(i))
        self.visibleOutcomes = visibleOutcomes
        self.graph.visibleOutcomes = visibleOutcomes
        self.graph.refreshVisibleOutcomes()
        #self.graph.replot()
        #self.repaint()

    def setVariable(self, varName):
        self.graph.setVariable(str(varName))
        self.graph.refreshVisibleOutcomes()
        self.xaxisTitle = str(varName)
        self.repaint()
    

if __name__ == "__main__":
    a = QApplication(sys.argv)
    owd = OWDistributions()
    a.setMainWidget(owd)
    owd.show()
    a.exec_loop()
    owd.saveSettings()