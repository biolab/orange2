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
import math
from OData import * 

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
        
        self.targetValue = 0
        self.data = None

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


    def setData(self, data, variable):
        self.data = data
        self.dc = orange.DomainContingency(self.data)
        self.setVariable(variable)

    def setVariable(self, variable):
        self.attributeName = str(variable)
        self.setXaxisTitle(str(variable))
        if not self.data: return
        
        if self.data.domain[self.attributeName].varType == orange.VarTypes.Continuous:
            self.variableContinuous = TRUE
        else:
            self.variableContinuous = FALSE

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
        if not self.data: return
        keys = self.hdata.keys()
        if self.variableContinuous:
            keys.sort()

        self.removeCurves()

        currentBarsHeight = [0] * len(keys)
        i = 0
        for oi in range(len(self.visibleOutcomes)):
            newColor = QColor()
            newColor.setHsv(self.colorHueValues[i]*360, 255, 255)
            i += 1
            if self.visibleOutcomes[oi] == 1:
                #for all bars insert curve and
                cn = 0
                for key in keys:
                    subBarHeight = self.hdata[key][oi]
                    curve = subBarQwtPlotCurve(self)
                    curve.color = newColor
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
        if not self.data: return
        if self.showProbabilities:
            self.enableYRaxis(1)
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
        #self.replot()

        
        
class OWDistributions(OWWidget):
    settingsList = ["NumberOfBars", "BarSize", "ShowProbabilities", "ShowConfidenceIntervals", "SmoothLines", "LineWidth", "ShowMainTitle", "ShowXaxisTitle", "ShowYaxisTitle", "ShowYPaxisTitle"]

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
        TRUE)
        # settings
        self.NumberOfBars = 5
        self.BarSize = 50
        self.ShowProbabilities = 0
        self.ShowConfidenceIntervals = 0
        self.SmoothLines = 0
        self.LineWidth = 1
        self.ShowMainTitle = 0
        self.ShowXaxisTitle = 1
        self.ShowYaxisTitle = 1
        self.ShowYPaxisTitle = 1

        #load settings
        self.loadSettings()

        # tmp values
        self.MainTitle = ""
        self.XaxisTitle = ""
        self.YaxisTitle = "frequency"
        self.YPaxisTitle = ""

        # GUI
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = OWDistributionsOptions(self, "Settings")
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
       

        # set values in options dialog to values in settings
        self.activateLoadedSettings()

        # GUI connections
        # options dialog connections
        self.connect(self.SettingsTab.barSize, SIGNAL("valueChanged(int)"), self.setBarSize)
        self.connect(self.SettingsTab.gSetMainTitleCB, SIGNAL("stateChanged(int)"), self.setShowMainTitle)
        self.connect(self.SettingsTab.gSetMainTitleLE, SIGNAL("textChanged(const QString &)"), self.setMainTitle)
        self.connect(self.SettingsTab.gSetXaxisCB, SIGNAL("stateChanged(int)"), self.setShowXaxisTitle)
        self.connect(self.SettingsTab.gSetXaxisLE, SIGNAL("textChanged(const QString &)"), self.setXaxisTitle)
        self.connect(self.SettingsTab.gSetYaxisCB, SIGNAL("stateChanged(int)"), self.setShowYaxisTitle)
        self.connect(self.SettingsTab.gSetYaxisLE, SIGNAL("textChanged(const QString &)"), self.setYaxisTitle)
        self.connect(self.SettingsTab.gSetYPaxisCB, SIGNAL("stateChanged(int)"), self.setShowYPaxisTitle)
        self.connect(self.SettingsTab.gSetYPaxisLE, SIGNAL("textChanged(const QString &)"), self.setYPaxisTitle)
        self.connect(self.SettingsTab.showprob, SIGNAL("stateChanged(int)"), self.setShowProbabilities)
        self.connect(self.SettingsTab.numberOfBars, SIGNAL("valueChanged(int)"), self.setNumberOfBars)
        self.connect(self.SettingsTab.smooth, SIGNAL("stateChanged(int)"), self.setSmoothLines)
        self.connect(self.SettingsTab.lineWidth, SIGNAL("valueChanged(int)"), self.setLineWidth)
        self.connect(self.SettingsTab.showcoin, SIGNAL("stateChanged(int)"), self.setShowConfidenceIntervals)
        # self connections

        #add controls to self.controlArea widget
        self.selvar = QVGroupBox(self.GeneralTab)
        self.selout = QVGroupBox(self.GeneralTab)
        self.selvar.setTitle("Variable")
        self.selout.setTitle("Outcomes")
        self.variablesQCB = QComboBox(self.selvar)
        self.outcomesQLB = QListBox(self.selout)
        self.outcomesQLB.setSelectionMode(QListBox.Multi)
        #connect controls to appropriate functions
        self.connect(self.outcomesQLB, SIGNAL("selectionChanged()"), self.outcomeSelectionChange)
        self.connect(self.variablesQCB, SIGNAL('activated (const QString &)'), self.setVariable)

    def updateGraphSettings(self):
        self.graph.numberOfBars = self.NumberOfBars
        self.graph.barSize = self.BarSize
        self.graph.showProbabilities = self.ShowProbabilities
        self.graph.showConfidenceIntervals = self.ShowConfidenceIntervals
        self.graph.smoothLines = self.SmoothLines
        self.graph.lineWidth = self.LineWidth
        self.graph.variableContinuous = self.VariableContinuous
        self.graph.targetValue = self.targetValue

    def setShowMainTitle(self, b):
        self.ShowMainTitle = b
        self.graph.setShowMainTitle(b)

    def setMainTitle(self, t):
        self.MainTitle = t
        if self.SettingsTab.gSetMainTitleLE.text() <> t:
            self.SettingsTab.gSetMainTitleLE.setText(QString(t))
        self.graph.setMainTitle(str(t))

    def setShowXaxisTitle(self, b):
        self.ShowXaxisTitle = b
        self.graph.setShowXaxisTitle(b)

    def setXaxisTitle(self, t):
        self.XaxisTitle = t
        if self.SettingsTab.gSetXaxisLE.text() <> t:
            self.SettingsTab.gSetXaxisLE.setText(QString(t))
        self.graph.setXaxisTitle(str(t))

    def setShowYaxisTitle(self, b):
        self.ShowYaxisTitle = b
        self.graph.setShowYLaxisTitle(b)

    def setYaxisTitle(self, t):
        self.YaxisTitle = t
        if self.SettingsTab.gSetYaxisLE.text() <> t:
            self.SettingsTab.gSetYaxisLE.setText(QString(t))
        self.graph.setYLaxisTitle(str(t))


    def setShowYPaxisTitle(self, b):
        self.ShowYPaxisTitle = b
        self.graph.setShowYRaxisTitle(b)

    def setYPaxisTitle(self, t):
        self.YPaxisTitle = t
        if self.SettingsTab.gSetYPaxisLE.text() <> t:
            self.SettingsTab.gSetYPaxisLE.setText(QString(t))
        self.graph.setYRaxisTitle(str(t))

    def setBarSize(self, n):
        self.BarSize = n
        self.graph.setBarSize(n)
        

    def setShowProbabilities(self, n):
        "Sets whether the probabilities are drawn or not"
        self.ShowProbabilities = n
        self.graph.showProbabilities = n
        self.graph.refreshProbGraph()
        #self.graph.replot()
        self.repaint()

    def setNumberOfBars(self, n):
        "Sets the number of bars for histograms of continuous variables"
        self.NumberOfBars = n
        self.graph.setNumberOfBars(n)
       

    def setSmoothLines(self, n):
        "sets the line smoothing on and off"
        self.SmoothLines = n
        #self.updateGraphSettings()

    def setLineWidth(self, n): 
        "Sets the line thickness for probability"
        self.LineWidth = n
        #self.updateGraphSettings()
        

    def setShowConfidenceIntervals(self,value):
        "Sets whether the confidence intervals are shown"
        self.ShowConfidenceIntervals = value
        self.graph.showConfidenceIntervals = value
        #self.updateGraphSettings()
        self.graph.refreshProbGraph()
        #self.graph.replot()

    def activateLoadedSettings(self):
        "Sets options in the settings dialog"
        self.SettingsTab.numberOfBars.setValue(self.NumberOfBars)
        self.setNumberOfBars(self.NumberOfBars)
        self.SettingsTab.barSize.setValue(self.BarSize)
        self.SettingsTab.gSetMainTitleLE.setText(self.MainTitle)
        self.SettingsTab.gSetMainTitleCB.setChecked(self.ShowMainTitle)
        
        self.SettingsTab.gSetXaxisLE.setText(self.XaxisTitle)
        self.SettingsTab.gSetXaxisCB.setChecked(self.ShowXaxisTitle)

        self.SettingsTab.gSetYaxisLE.setText(self.YaxisTitle)
        self.SettingsTab.gSetYaxisCB.setChecked(self.ShowYaxisTitle)

        self.SettingsTab.gSetYPaxisLE.setText(self.YPaxisTitle)
        self.SettingsTab.gSetYPaxisCB.setChecked(self.ShowYPaxisTitle)
        self.SettingsTab.showprob.setChecked(self.ShowProbabilities)
        self.SettingsTab.showcoin.setChecked(self.ShowConfidenceIntervals)
        self.SettingsTab.smooth.setChecked(self.SmoothLines)
        self.SettingsTab.lineWidth.setValue(self.LineWidth)

        self.setMainTitle(self.MainTitle)
        self.setShowMainTitle(self.ShowMainTitle)
        self.setXaxisTitle(self.XaxisTitle)
        self.setShowXaxisTitle(self.ShowXaxisTitle)
        self.setYaxisTitle(self.YaxisTitle)
        self.setShowYaxisTitle(self.ShowYaxisTitle)
        self.setYPaxisTitle(self.YPaxisTitle)
        self.setShowYPaxisTitle(self.ShowYPaxisTitle)
        self.setShowProbabilities(self.ShowProbabilities)
        #self.updateGraphSettings()

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
            self.graph.setXlabels(None)
            self.graph.setYLlabels(None)
            self.graph.setShowYRaxisTitle(0)
            self.graph.data = None
            self.data = None
            return

        self.data = orange.Preprocessor_dropMissingClasses(data)
        
        names = []
        for attr in self.data.domain.attributes:
            names.append(attr.name)
        if self.graph.attributeName not in names:
            self.graph.attributeName = names[0]
        
        self.setVariablesComboBox(names, names.index(self.graph.attributeName))
        self.setOutcomeNames([])
        if self.data.domain.classVar:
            self.setOutcomeNames(self.data.domain.classVar.values.native())
        self.graph.setData(self.data, self.graph.attributeName)


    def setVariablesComboBox(self, list, defaultItem = 0):
        "Set the variables with the suplied list."
        self.variablesQCB.clear()
        for i in list:    self.variablesQCB.insertItem(i)
        if len(list) > 0:
            self.variablesQCB.setCurrentItem(defaultItem)
            self.setVariable(self.variablesQCB.text(defaultItem))

#
#
# graph calculation and manipulation

    def setOutcomeNames(self, list):
        "Sets the outcome target names."
        self.outcomesQLB.clear()
        for i in range(len(list)):
            c = QColor()
            c.setHsv(self.graph.colorHueValues[i]*360, 255, 255)
            self.outcomesQLB.insertItem(ColorPixmap(c), list[i])
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
        self.setXaxisTitle(str(varName))
        self.repaint()
    


class OWDistributionsOptions(QVGroupBox):
    def __init__(self,parent=None,name=None):
        QVGroupBox.__init__(self, parent, name)
        self.dist=QVGroupBox(self)
        self.dist.setTitle("Distribution Graph")
        self.nb=QHGroupBox(self.dist)
        self.nb.setTitle("Number of Bars")
        QToolTip.add(self.nb,"Number of bars for graphs\nof continuous variables")
        self.numberOfBars=QSlider(5,60,5,5,QSlider.Horizontal,self.nb)
        self.numberOfBars.setTickmarks(QSlider.Below)
        self.numberOfBars.setTracking(0) # no change until the user stop dragging the slider
        self.nbLCD=QLCDNumber(2,self.nb)
        self.nbLCD.display(5)
        self.connect(self.numberOfBars,SIGNAL("valueChanged(int)"),self.nbLCD,SLOT("display(int)"))
        self.bx=QHGroupBox(self.dist)
        self.bx.setTitle("Bar Size")
        QToolTip.add(self.bx,"The size of bars\nin percentage\nof available space (for discrete variables)")
        self.barSize=QSlider(30,100,10,50,QSlider.Horizontal,self.bx)
        self.barSize.setTickmarks(QSlider.Below)
        self.barSize.setLineStep(10)
#        self.barSize.setTracking(0) # no change until the user stop dragging the slider
        self.bxLCD=QLCDNumber(3,self.bx)
        self.bxLCD.display(50)
        self.connect(self.barSize,SIGNAL("valueChanged(int)"),self.bxLCD,SLOT("display(int)"))

        self.graphSettings = QVButtonGroup("General graph settings", self)
        QToolTip.add(self.graphSettings, "Enable/disable axis title")
        self.gSetMainTitle = QHBox(self.graphSettings, "main title group")
        self.gSetMainTitleCB = QCheckBox('Show Main Title', self.gSetMainTitle)
        self.gSetMainTitleLE = QLineEdit('Main Title', self.gSetMainTitle)
        self.gSetXaxis = QHBox(self.graphSettings, "X Axis Group")
        self.gSetXaxisCB = QCheckBox('Show X Axis Title ', self.gSetXaxis)
        self.gSetXaxisLE = QLineEdit('X Axis Title', self.gSetXaxis)
        self.gSetYaxis = QHBox(self.graphSettings, "Y Axis Group")
        self.gSetYaxisCB = QCheckBox('Show Y Axis Title ', self.gSetYaxis)
        self.gSetYaxisLE = QLineEdit('Y Axis Title', self.gSetYaxis)

        self.pg=QVGroupBox(self)
        self.pg.setTitle("Probability graph")
        self.showprob=QCheckBox("Show Probabilities",self.pg)
        self.gSetYPaxis = QHBox(self.pg, "y prob. axis group")
        self.gSetYPaxisCB = QCheckBox('Show Axis Title ', self.gSetYPaxis)
        self.gSetYPaxisLE = QLineEdit('Axis Title', self.gSetYPaxis)
        self.showcoin=QCheckBox("Show Confidence Intervals",self.pg)
        self.smooth=QCheckBox("Smooth probability lines",self.pg)
        self.lw=QHGroupBox(self.pg)
        self.lw.setTitle("Line width")
        QToolTip.add(self.lw,"The width of lines in pixels")
        self.lineWidth=QSlider(1,9,1,1,QSlider.Horizontal,self.lw)
        self.lineWidth.setTickmarks(QSlider.Below)
#        self.lineWidth.setTracking(0) # no change signaled until the user stop dragging the slider
        self.lwLCD=QLCDNumber(1,self.lw)
        self.lwLCD.display(1)
        self.smooth.setEnabled(0)
        self.connect(self.lineWidth,SIGNAL("valueChanged(int)"),self.lwLCD,SLOT("display(int)"))
        self.connect(self.showprob,SIGNAL("stateChanged(int)"),self.prob2CI)
              
    def prob2CI(self,state):
        if state==0:
            self.showcoin.setChecked(0)
        self.showcoin.setEnabled(state)
        self.gSetYPaxis.setEnabled(state)


if __name__ == "__main__":
    a = QApplication(sys.argv)
    owd = OWDistributions()
    a.setMainWidget(owd)
    data = orange.ExampleTable("titanic")
    owd.cdata(data)
    owd.show()
    a.exec_loop()
    owd.saveSettings()