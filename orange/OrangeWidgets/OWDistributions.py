"""
<name>Distributions</name>
<description>The Distribution Widget shows data distributions, distribution of attribute values and
distribution of classes for each attribute.</description>
<category>Data</category>
<icon>icons/Distribution.png</icon>
<priority>2100</priority>
"""

#
# OWDistributions.py
# Distributions Widget
# Shows data distributions, distribution of attribute values and distribution of classes for each attribute
#

from OData import *
from OWTools import *
from OWWidget import *
from OWDistributionsOptions import *
from OWGraph import *

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


class OWDistributions(OWWidget):
    settingsList = ["NumberOfBars", "BarSize", "ShowProbabilities", "ShowConfidenceIntervals", "SmoothLines", "LineWidth"]

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
        TRUE,
        TRUE)
        # settings
        self.NumberOfBars = 5
        self.BarSize = 50
        self.ShowProbabilities = 0
        self.ShowConfidenceIntervals = 0
        self.SmoothLines = 0
        self.LineWidth = 1

        #load settings
        self.loadSettings()

        # GUI
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWGraph(self.mainArea)
        self.graph.setYRlabels(None)
        self.graph.setAxisScale(QwtPlot.yRight, 0.0, 1.0, 0.1)
        self.box.addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # inputs
        # data and graph temp variables
        self.addInput("cdata")
        self.addInput("target")
        self.data = None
        self.hdata = {} # keep track of distribution data for the selected target attribute for the values(outcomeNames) of the selected class(target)
        self.Variable = 0
        self.VariableContinuous = FALSE
        self.targetValue = 0
        self.visibleOutcomes = []
        self.outcomenames = []
        self.probGraphValues = []

        curve = distribErrorBarQwtPlotCurve(self.graph, '')
        self.probCurveKey = self.graph.insertCurve(curve)
        self.graph.setCurveXAxis(self.probCurveKey, QwtPlot.xBottom)
        self.graph.setCurveYAxis(self.probCurveKey, QwtPlot.yRight)

        self.probCurveUpperCIKey = self.graph.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.probCurveLowerCIKey = self.graph.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.graph.curve(self.probCurveKey).setEnabled(FALSE)
        self.graph.curve(self.probCurveUpperCIKey).setEnabled(FALSE)
        self.graph.curve(self.probCurveLowerCIKey).setEnabled(FALSE)

        # set values in options dialog to values in settings
        self.options = OWDistributionsOptions()
        self.activateLoadedSettings()

        #connect settingsbutton to show options
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)

        # GUI connections
            # options dialog connections
        self.connect(self.options.barSize, SIGNAL("valueChanged(int)"), self.setBarSize)
        self.connect(self.options.showprob, SIGNAL("stateChanged(int)"), self.setShowProbabilities)
        self.connect(self.options.numberOfBars, SIGNAL("valueChanged(int)"), self.setNumberOfBars)
        self.connect(self.options.smooth, SIGNAL("stateChanged(int)"), self.setSmoothLines)
        self.connect(self.options.lineWidth, SIGNAL("valueChanged(int)"), self.setLineWidth)
        self.connect(self.options.showcoin, SIGNAL("stateChanged(int)"), self.setShowConfidenceIntervals)
            # self connections

        #add controls to self.controlArea widget 
        self.selout = QVGroupBox(self.space)
        self.selvar = QVGroupBox(self.controlArea)
        self.selvar.setTitle("Variable")
        self.selout.setTitle("Outcomes")
        self.variablesQCB = QComboBox(self.selvar)
        self.outcomesQLB = QListBox(self.selout)
        self.outcomesQLB.setSelectionMode(QListBox.Multi)
        #connect controls to appropriate functions
        self.connect(self.outcomesQLB, SIGNAL("selectionChanged()"), self.outcomeSelectionChange)
        self.connect(self.variablesQCB, SIGNAL('activated (const QString &)'), self.setVariable)

    def setBarSize(self, n):
        self.BarSize = n

        if not(self.VariableContinuous):
            self.refreshVisibleOutcomes()
            self.graph.replot()
            self.repaint()

    def setShowProbabilities(self, n):
        "Sets whether the probabilities are drawn or not"
        self.ShowProbabilities = n
        self.refreshProbGraph()
        self.graph.replot()
        self.repaint()

    def setNumberOfBars(self, n):
        "Sets the number of bars for histograms of continuous variables"
        self.NumberOfBars = n

        if self.VariableContinuous:
            self.graph.replot()
            self.repaint()
            self.calcHistogramAndProbGraph()
            self.refreshVisibleOutcomes()
            self.graph.replot()
            self.repaint()

    def setSmoothLines(self, n):
        "sets the line smoothing on and off"
        self.SmoothLines = n

    def setLineWidth(self, n): 
        "Sets the line thickness for probability"
        self.LineWidth = n

    def setShowConfidenceIntervals(self,value):
        "Sets whether the confidence intervals are shown"
        self.ShowConfidenceIntervals = value
        self.refreshProbGraph()
        self.graph.replot()
        self.repaint()

    def activateLoadedSettings(self):
        "Sets options in the settings dialog"
        self.options.numberOfBars.setValue(self.NumberOfBars)
        self.setNumberOfBars(self.NumberOfBars)
        #
        self.options.barSize.setValue(self.BarSize)
        #
        self.options.showprob.setChecked(self.ShowProbabilities)
        self.setShowProbabilities(self.ShowProbabilities)
        #
        self.options.showcoin.setChecked(self.ShowConfidenceIntervals)
        #
        self.options.smooth.setChecked(self.SmoothLines)
        #
        self.options.lineWidth.setValue(self.LineWidth)

    def target(self, targetValue):
        self.targetValue = targetValue
        self.refreshProbGraph()

    def cdata(self, data):
        self.data = data

        if self.data == None:
            self.setVariablesComboBox([])
            self.setOutcomeNames([])
            self.graph.setXlabels(None)
            self.graph.setYLlabels(None)
            self.graph.setShowYRaxisTitle(0)
            return

        self.setVariablesComboBox(self.data.getVarNames())
        self.setOutcomeNames(self.data.getVarValues(self.data.getOutcomeName()))
        self.dc = self.data.getDC()
        self.setVariable(self.data.getVarNames()[0])

    def setVariablesComboBox(self, list):
        "Set the variables with the suplied list."
        self.variablesQCB.clear()
        for i in list:
            self.variablesQCB.insertItem(i)
        if len(list) > 0:
            self.variablesQCB.setCurrentItem(0)

#
#
# graph calculation and manipulation

    def setOutcomeNames(self, list):
        "Sets the outcome target names."
        self.outcomesQLB.clear()
        for i in range(len(list)):
            c = QColor()
            c.setHsv(i*360/len(list), 255, 255)
            self.outcomesQLB.insertItem(ColorPixmap(c), list[i])
        self.outcomesQLB.selectAll(TRUE)

    def outcomeSelectionChange(self):
        "Sets which outcome values are represented in the graph."
        "Reacts to changes in outcome selection."
        visibleOutcomes = []
        for i in range(self.outcomesQLB.numRows()):
            visibleOutcomes.append(self.outcomesQLB.isSelected(i))
        self.visibleOutcomes = visibleOutcomes
        self.refreshVisibleOutcomes()
        self.graph.replot()
        self.repaint()

    def setVariable(self, varName):
        self.Variable = self.data.getVarNames().index(str(varName))
        if self.data.getVarNames()[self.Variable] in self.data.getVarNames(CONTINUOUS):
            self.VariableContinuous = TRUE
        else:
            self.VariableContinuous = FALSE
        self.calcHistogramAndProbGraph()

        if self.VariableContinuous:
            self.graph.setXlabels(None)
        else:
            labels = self.data.getVarValues(self.Variable)
            self.graph.setXlabels(labels)
            self.graph.setAxisScale(QwtPlot.xBottom, -0.5, len(labels) - 0.5, 1)

        self.refreshVisibleOutcomes()
        self.graph.replot()
        self.repaint()

    def calcHistogramAndProbGraph(self):
        "Calculates the histogram."
        if self.data == None:
            return

        if self.VariableContinuous:
            "Continuous variable, break data into self.NumberOfBars subintervals"
            "use orange.EquiDistDiscretization(numberOfIntervals)"
            equiDist = orange.EquiDistDiscretization(numberOfIntervals = self.NumberOfBars)
            d_variable = equiDist(self.data.getVarNames()[self.Variable], self.data.data)
            d_data = self.data.data.select([d_variable, self.data.data.domain.classVar])
            tmphdata = orange.DomainContingency(d_data)[0]
            dc = orange.DomainContingency(self.data.data) #!!!
            g = orange.ConditionalProbabilityEstimatorConstructor_loess(dc[self.Variable]) #!!!
##            print [ps.variances for (x, ps) in g.probabilities.items()]
            self.probGraphValues = [(x, ps, [v*1.96 for v in ps.variances]) for (x, ps) in g.probabilities.items()]
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
            self.hdata = self.dc[self.Variable]
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
        keys = self.hdata.keys()
        if self.VariableContinuous:
            keys.sort()

        self.graph.removeCurves()

        currentBarsHeight = [0] * len(keys)
        for oi in range(len(self.visibleOutcomes)):
            newColor = QColor()
            newColor.setHsv(oi*360/len(self.visibleOutcomes), 255, 255)
            if self.visibleOutcomes[oi] == 1:
                #for all bars insert curve and
                cn = 0
                for key in keys:
                    subBarHeight = self.hdata[key][oi]
                    curve = subBarQwtPlotCurve(self.graph)
                    curve.color = newColor
                    ckey = self.graph.insertCurve(curve)
                    self.graph.setCurveStyle(ckey, QwtCurve.UserCurve)
                    if self.VariableContinuous:
                        self.graph.setCurveData(ckey, [key, key + self.subIntervalStep], [currentBarsHeight[cn], currentBarsHeight[cn] + subBarHeight])
                    else:
                        tmpx = cn - (self.BarSize/2.0)/100.0
                        tmpx2 = cn + (self.BarSize/2.0)/100.0
                        self.graph.setCurveData(ckey, [tmpx, tmpx2], [currentBarsHeight[cn], currentBarsHeight[cn] + subBarHeight])
                    currentBarsHeight[cn] += subBarHeight
                    cn += 1

        curve = distribErrorBarQwtPlotCurve(self.graph, '')
        self.probCurveKey = self.graph.insertCurve(curve)
        self.graph.setCurveXAxis(self.probCurveKey, QwtPlot.xBottom)
        self.graph.setCurveYAxis(self.probCurveKey, QwtPlot.yRight)

        self.probCurveUpperCIKey = self.graph.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.probCurveLowerCIKey = self.graph.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.refreshProbGraph()

    def refreshProbGraph(self):
        if self.ShowProbabilities:
            self.graph.enableYRaxis(1)
            xs = []
            ups = []
            mps = []
            lps = []
            cn = 0.0
            for (x, ps, cis) in self.probGraphValues:
                if self.VariableContinuous:
                    xs.append(x)
                    ups.append(ps[self.targetValue] + cis[self.targetValue])
                    mps.append(ps[self.targetValue] + 0.0)
                    lps.append(ps[self.targetValue] - cis[self.targetValue])
                else:
                    if self.ShowConfidenceIntervals:
                        xs.append(cn)
                        mps.append(ps[self.targetValue] + cis[self.targetValue])

                    xs.append(cn)
                    mps.append(ps[self.targetValue] + 0.0)

                    if self.ShowConfidenceIntervals:
                        xs.append(cn)
                        mps.append(ps[self.targetValue] - cis[self.targetValue])
                cn += 1.0

            ## (re)set the curves
            if self.VariableContinuous:
                newSymbol = QwtSymbol(QwtSymbol.None, QBrush(Qt.color0), QPen(Qt.black, 2), QSize(0,0))
            else:
                newSymbol = QwtSymbol(QwtSymbol.Diamond, QBrush(Qt.color0), QPen(Qt.black, 2), QSize(7,7))

            self.graph.setCurveData(self.probCurveKey, xs, mps)
            self.graph.setCurveSymbol(self.probCurveKey, newSymbol)

            if self.VariableContinuous:
                self.graph.setCurveStyle(self.probCurveKey, QwtCurve.Lines)
                if self.ShowConfidenceIntervals:
                    self.graph.setCurveData(self.probCurveUpperCIKey, xs, ups)
                    self.graph.setCurveData(self.probCurveLowerCIKey, xs, lps)
            else:
                if self.ShowConfidenceIntervals:
                    self.graph.setCurveStyle(self.probCurveKey, QwtCurve.UserCurve)
                else:
                    self.graph.setCurveStyle(self.probCurveKey, QwtCurve.Dots)
        else:
            self.graph.enableYRaxis(0)
            self.graph.setShowYRaxisTitle(0)

        self.graph.curve(self.probCurveKey).setEnabled(self.ShowProbabilities)
        self.graph.curve(self.probCurveUpperCIKey).setEnabled(self.ShowConfidenceIntervals and self.ShowProbabilities)
        self.graph.curve(self.probCurveLowerCIKey).setEnabled(self.ShowConfidenceIntervals and self.ShowProbabilities)
        self.graph.curve(self.probCurveKey).itemChanged()
        self.graph.curve(self.probCurveUpperCIKey).itemChanged()
        self.graph.curve(self.probCurveLowerCIKey).itemChanged()
        self.graph.replot()
        self.repaint()

if __name__ == "__main__":
    a = QApplication(sys.argv)
    owd = OWDistributions()
    a.setMainWidget(owd)
    owd.show()
    a.exec_loop()
    owd.saveSettings()
