"""
<name>Distributions</name>
<description>The Distribution Widget shows data distributions, distribution of attribute values and
distribution of classes for each attribute.</description>
<category>Data</category>
<icon>icons/Distribution.png</icon>
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

class subBarQwtCurve(QwtCurve):
    def __init__(self, parent = None, text = None):
        QwtCurve.__init__(self, parent, text)
        self.color = Qt.black

    def draw(self, p, xMap, yMap, f, t):
        p.setBackgroundMode(Qt.OpaqueMode)
        p.setBackgroundColor(self.color)
        p.setBrush(self.color)
        p.setPen(Qt.black)
        if t < 0: t = self.dataSize() - 1
        if divmod(f, 2)[1] != 0: f -= 1
        if divmod(t, 2)[1] == 0:  t += 1
        for i in range(f, t+1, 2):
            px1 = xMap.transform(self.x(i))
            py1 = yMap.transform(self.y(i))
            px2 = xMap.transform(self.x(i+1))
            py2 = yMap.transform(self.y(i+1))
#            print "draw from ", px1, ",", py1, "to", px2, ",", py2
            p.drawRect(px1, py1, (px2 - px1), (py2 - py1))

class subBarQwtPlotCurve(QwtPlotCurve, subBarQwtCurve): # there must be a better way to do this
    def dummy():
        None

class OWDistributions(OWWidget):
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
        self.settingsList = ["NumberOfBars", "BarSize", "ShowProbabilities", "ShowConfidenceIntervals", "SmoothLines", "LineWidth"]
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
        self.probCurveKey = self.graph.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.probCurveUpperCIKey = self.graph.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.probCurveLowerCIKey = self.graph.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.graph.curve(self.probCurveKey).setEnabled(FALSE)
        self.graph.curve(self.probCurveUpperCIKey).setEnabled(FALSE)
        self.graph.curve(self.probCurveLowerCIKey).setEnabled(FALSE)

        # set values in options dialog to values in settings
        self.options = OWDistributionsOptions()
        self.setOptions()

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

    def setOptions(self):
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
##        newColor = QColor()
##        newColor.setHsv(5*360/12, 255, 255)
##        curve = subBarQwtPlotCurve(self.graph)
##        curve.setPen(QPen(newColor))
##        ckey = self.graph.insertCurve(curve)
##        self.graph.setCurveStyle(ckey, QwtCurve.UserCurve)
##        self.graph.setCurveData(ckey, [10, 11], [5, 7])

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
        self.probCurveKey = self.graph.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.probCurveUpperCIKey = self.graph.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.probCurveLowerCIKey = self.graph.insertCurve('', QwtPlot.xBottom, QwtPlot.yRight)
        self.refreshProbGraph()

    def refreshProbGraph(self):
        if self.ShowProbabilities:
            if self.VariableContinuous:
                newSymbol = QwtSymbol()
            else:
                newSymbol = QwtSymbol() #QwtSymbol.Cross, QBrush(Qt.red), QPen(Qt.red), QSize(1,100))

            self.graph.enableYRaxis(1)
            xs = []
            ups = []
            mps = []
            lps = []
            cn = 0.0
            for (x, ps, cis) in self.probGraphValues:
                if self.VariableContinuous:
                    xs.append(x)
                else:
                    xs.append(cn)
                ups.append(ps[self.targetValue] + cis[self.targetValue])
                mps.append(ps[self.targetValue] + 0.0)
                lps.append(ps[self.targetValue] - cis[self.targetValue])
                cn += 1.0

            self.graph.setCurveData(self.probCurveKey, xs, mps)
            self.graph.setCurveSymbol(self.probCurveKey, newSymbol)
            if self.ShowConfidenceIntervals:
                self.graph.setCurveData(self.probCurveUpperCIKey, xs, ups)
                self.graph.setCurveData(self.probCurveLowerCIKey, xs, lps)
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
