"""
<name>2D Interactions</name>
<description>Shows interactions of two attributes (works for discrete and numerical attributes)</description>
<category>Visualization</category>
<icon>pics\2DInteractions.png</icon>
"""
# OW2DInteractions.py
#
# 2D Interactions is an Orange Widget that
# shows interactions of two attributes 
# (works for both discrete and numerica attributes).
# 

from OWWidget import *
from OW2DInteractionsOptions import *
from random import betavariate 
from OWGraph import *
from OData import *

class OW2DInteractions(OWWidget):
    def __init__(self,parent=None):
        self.spreadType=["none","uniform","triangle","beta"]
        OWWidget.__init__(self,
        parent,
        "2D &Interactions",
        """2D Interactions is an Orange Widget that
shows interactions of two attributes 
(works for both discrete and numerical attributes).

""",
        TRUE,
        TRUE)

        #set default settings
        self.settingsList = ["PointWidth", "RandomSpreadType", "ShowMainGraphTitle", "ShowXAxisTitle",
                             "ShowYAxisTitle", "ShowVerticalGridlines", "ShowHorizontalGridlines",
                             "ShowLegend", "GraphGridColor", "GraphCanvasColor"]
        self.PointWidth = 3
        self.RandomSpreadType = "uniform"
        self.ShowMainGraphTitle = FALSE
        self.ShowXAxisTitle = TRUE
        self.ShowYAxisTitle = TRUE
        self.ShowVerticalGridlines = TRUE
        self.ShowHorizontalGridlines = TRUE
        self.ShowLegend = TRUE
        self.GraphGridColor = str(Qt.black.name())
        self.GraphCanvasColor = str(Qt.white.name())

        #load settings
        self.loadSettings()

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWGraph(self.mainArea)
        self.box.addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # graph main tmp variables
        self.addInput("cdata")
        self.data = None
        self.xAxis = 0
        self.yAxis = 1
        self.outcome = 0
        self.outcomenames = []
        self.visibleOutcomes = []
        self.curveKeys = []
        self.curveColors = []

        # add a settings dialog and initialize its values
        self.options = OW2DInteractionsOptions()
        self.setOptions()

        #connect settingsbutton to show options
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)

        #connect GUI controls of options in options dialog to settings
        self.connect(self.options.widthSlider, SIGNAL("valueChanged(int)"), self.setPointWidth)
        self.connect(self.options.spreadButtons, SIGNAL("clicked(int)"), self.setSpreadType)
        self.connect(self.options.gSetMainTitleCB, SIGNAL("toggled(bool)"), self.setShowMainGraphTitle)
        self.connect(self.options.gSetMainTitleLE, SIGNAL("textChanged(const QString &)"), self.setMainGraphTitle)
        self.connect(self.options.gSetXaxisCB, SIGNAL("toggled(bool)"), self.setShowXaxisTitle)
        self.connect(self.options.gSetYaxisCB, SIGNAL("toggled(bool)"), self.setShowYaxisTitle)
        self.connect(self.options.gSetVgridCB, SIGNAL("toggled(bool)"), self.setShowVgridAxis)
        self.connect(self.options.gSetHgridCB, SIGNAL("toggled(bool)"), self.setShowHgridAxis)
        self.connect(self.options.gSetLegendCB, SIGNAL("toggled(bool)"), self.setShowLegend)
        self.connect(self.options, PYSIGNAL("gridColorChange(QColor &)"), self.setGridColor)
        self.connect(self.options, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)

        #add controls to self.controlArea widget 
        self.selx = QVGroupBox(self.controlArea)
        self.sely = QVGroupBox(self.controlArea)
        self.selout = QVGroupBox(self.space)
        self.selx.setTitle("X Axis")
        self.sely.setTitle("Y Axis")
        self.selout.setTitle("Outcomes")
        self.xaQCB = QComboBox(self.selx)
        self.yaQCB = QComboBox(self.sely)
        self.outcomesQLB = QListBox(self.selout)
        self.outcomesQLB.setSelectionMode(QListBox.Multi)
        #connect controls to appropriate functions
        self.connect(self.xaQCB, SIGNAL('activated ( const QString & )'), self.xAxisChange)
        self.connect(self.yaQCB, SIGNAL('activated ( const QString & )'), self.yAxisChange)
        self.connect(self.outcomesQLB, SIGNAL("selectionChanged()"), self.outcomeSelectionChange)

    def setPointWidth(self, n):
        self.PointWidth = n
        for curveIndex in range(len(self.curveKeys)):
            newSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(self.curveColors[curveIndex]), QPen(self.curveColors[curveIndex]), QSize(n, n))
            self.graph.setCurveSymbol(self.curveKeys[curveIndex], newSymbol)
        self.graph.replot()
        self.repaint()

    def setSpreadType(self, n):
        self.RandomSpreadType = self.spreadType[n]
        self.calcCurves()
        self.refreshVisibleCurves()
        self.graph.replot()
        self.repaint()

    def setShowMainGraphTitle(self, b):
        self.ShowMainGraphTitle = b
        self.graph.setShowMainTitle(b)

    def setMainGraphTitle(self, t):
        if self.options.gSetMainTitleLE.text() <> t:
            self.options.gSetMainTitleLE.setText(QString(t))
        self.graph.setMainTitle(str(t))

    def setShowXaxisTitle(self, b):
        self.ShowXAxisTitle = b
        self.graph.setShowXaxisTitle(b)

    def setXaxisTitle(self, t):
        self.graph.setXaxisTitle(str(t))

    def setShowYaxisTitle(self, b):
        self.ShowYAxisTitle = b
        self.graph.setShowYLaxisTitle(b)

    def setYaxisTitle(self, t):
        self.graph.setYLaxisTitle(str(t))

    def setShowVgridAxis(self, b):
        self.ShowVerticalGridlines = b
        self.graph.enableGridXB(b)

    def setShowHgridAxis(self, b):
        self.ShowHorizontalGridlines = b
        self.graph.enableGridYL(b)

    def setShowLegend(self, b):
        self.ShowLegend = b
        self.graph.enableGraphLegend(b)

    def setGridColor(self, c):
        self.GraphGridColor = str(c.name())
        self.graph.setGridColor(c)

    def setCanvasColor(self, c):
        self.GraphCanvasColor = str(c.name())
        self.graph.setCanvasColor(c)

    def setOptions(self):
        self.options.widthSlider.setValue(self.PointWidth)
        self.options.widthLCD.display(self.PointWidth)
        self.setPointWidth(self.PointWidth)
        #
        self.options.spreadButtons.setButton(self.spreadType.index(self.RandomSpreadType))
        self.setSpreadType(self.spreadType.index(self.RandomSpreadType))
        #
        self.options.gSetMainTitleCB.setChecked(self.ShowMainGraphTitle)
        self.setShowMainGraphTitle(self.ShowMainGraphTitle)
        #
        self.options.gSetXaxisCB.setChecked(self.ShowXAxisTitle)
        self.setShowXaxisTitle(self.ShowXAxisTitle)
        #
        self.options.gSetYaxisCB.setChecked(self.ShowYAxisTitle)
        self.setShowYaxisTitle(self.ShowYAxisTitle)
        #
        self.options.gSetVgridCB.setChecked(self.ShowVerticalGridlines)
        self.setShowVgridAxis(self.ShowVerticalGridlines)
        #
        self.options.gSetHgridCB.setChecked(self.ShowHorizontalGridlines)
        self.setShowHgridAxis(self.ShowHorizontalGridlines)
        #
        self.options.gSetLegendCB.setChecked(self.ShowLegend)
        self.setShowLegend(self.ShowLegend)
        #
        self.options.gSetGridColor.setNamedColor(str(self.GraphGridColor))
        self.setGridColor(self.options.gSetGridColor)
        #
        self.options.gSetCanvasColor.setNamedColor(str(self.GraphCanvasColor))
        self.setCanvasColor(self.options.gSetCanvasColor)

    def cdata(self, data):
        self.data = data

        if self.data == None:
            self.setMainGraphTitle('')
            self.setComboBoxes([])
            self.setOutcomeNames([])
            self.setXAxis(None)
            self.setYAxis(None)
            self.repaint()
            return

        self.setMainGraphTitle(data.title)
        self.setComboBoxes(self.data.getVarNames())
        self.setOutcomeNames(self.data.getVarValues(self.data.getOutcomeName()))
        self.setXAxis(self.data.getVarNames()[0])
        self.setYAxis(self.data.getVarNames()[1])
        list = self.data.getPotentialOutcomes()
        self.outcome = self.data.getVarNames(None,TRUE).index(list[-1])
        self.calcCurves()
        self.refreshVisibleCurves()
        self.graph.replot()
        self.repaint()

    def setComboBoxes(self, list):
        self.xaQCB.clear()
        self.yaQCB.clear()
        for i in list:
            self.xaQCB.insertItem(i)
            self.yaQCB.insertItem(i)
        self.xaQCB.setCurrentItem(0)
        if len(list) > 0:
            self.yaQCB.setCurrentItem(1)
        else:
            self.yaQCB.setCurrentItem(0)

#
#
# graph calculation and manipulation

    def setOutcomeNames(self, list):
        self.outcomenames = list

        # create the appropriate number of curves, one for every outcomename
        # first remove old curves and self.outcomes(QListBox) items if any
        self.graph.removeCurves()
        self.outcomesQLB.clear()
        if len(self.outcomenames) == 0: return

        # create new curves and QListBox items
        self.curveKeys = []
        self.curveColors = []
        for curveIndex in range(len(self.outcomenames)):
            # insert QListBox item
            newColor = QColor()
            newColor.setHsv(curveIndex*360/len(list), 255, 255)
            self.outcomesQLB.insertItem(ColorPixmap(newColor), list[curveIndex])

            # insert curve
            self.curveColors.append(newColor)
            newSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(newColor), QPen(newColor), QSize(self.PointWidth, self.PointWidth))
            newCurveKey = self.graph.insertCurve(self.outcomenames[curveIndex])
            self.curveKeys.append(newCurveKey)
            self.graph.setCurveStyle(newCurveKey, QwtCurve.Dots)
            self.graph.setCurveSymbol(newCurveKey, newSymbol)

        self.outcomesQLB.selectAll(TRUE)

    def outcomeSelectionChange(self):
        "Reacts to changes in outcome selection and sets which outcome values are represented in the graph."
        visibleOutcomes = []
        for i in range(self.outcomesQLB.numRows()):
            visibleOutcomes.append(self.outcomesQLB.isSelected(i))
        self.visibleOutcomes = visibleOutcomes
        self.refreshVisibleCurves()
        self.graph.replot()
        self.repaint()

    def setXAxis(self, xa):
        self.setXaxisTitle(xa)
        if xa == None:
            self.xAxis = None
            self.graph.setXlabels([' '])
            self.graph.setAxisScale(QwtPlot.xBottom, -0.5, 0.5, 1)
        else:
            xa = str(xa) #QString->python string
            self.xAxis = self.data.getVarNames().index(xa) 

            if self.data.data.domain[xa].varType == orange.VarTypes.Continuous:
                self.graph.setXlabels(None)
            else:
                xlabels = self.data.getVarValues(self.data.getVarNames().index(xa))
                self.graph.setXlabels(xlabels)
                # reset the xAxis so all values (+- 0.5) changed by rndCorrection() are shown on graph
                self.graph.setAxisScale(QwtPlot.xBottom, -0.5, len(xlabels) - 0.5, 1)

    def setYAxis(self, ya):
        self.setYaxisTitle(ya)
        if ya == None:
            self.yAxis = None
            self.graph.setYLlabels([' '])
            self.graph.setAxisScale(QwtPlot.yLeft, -0.5, 0.5, 1)
        else:
            ya = str(ya) #QString->python string
            self.yAxis = self.data.getVarNames().index(ya)

            if self.data.data.domain[ya].varType == orange.VarTypes.Continuous:
                self.graph.setYLlabels(None)
            else:
                ylabels = self.data.getVarValues(self.data.getVarNames().index(ya))
                self.graph.setYLlabels(ylabels)
                # reset the yAxis so all values (+- 0.5) changef by rndCorrection() are shown on graph
                self.graph.setAxisScale(QwtPlot.yLeft, -0.5, len(ylabels) - 0.5, 1)

    def xAxisChange(self, newxa):
        self.setXAxis(newxa)
        self.calcCurves()
        self.refreshVisibleCurves()
        self.graph.replot()
        self.repaint()

    def yAxisChange(self, newya):
        self.setYAxis(newya)
        self.calcCurves()
        self.refreshVisibleCurves()
        self.graph.replot()
        self.repaint()

    def calcCurves(self):
        if self.data == None:
            return

        # initialize the data point sets
        curveDataPoints = []
        for curveIndex in range(len(self.outcomenames)):
            curveDataPoints.append( {'x':[], 'y':[]} )

        # calculate data points
        for i in self.data.table:
            if i[self.outcome].isSpecial():
                continue
            ins = self.outcomenames.index(str(i[self.outcome]))
            if i[self.xAxis].isSpecial():
                continue
            if i[self.yAxis].isSpecial():
                continue
            curveDataPoints[ins]['x'].append( i[self.xAxis] + self.rndCorrection() )
            curveDataPoints[ins]['y'].append( i[self.yAxis] + self.rndCorrection() )

        # put data into curves
        for curveIndex in range(len(self.curveKeys)):
            self.graph.setCurveData(self.curveKeys[curveIndex], curveDataPoints[curveIndex]['x'], curveDataPoints[curveIndex]['y'])

    def refreshVisibleCurves(self):
        for curveIndex in range(min(len(self.curveKeys), len(self.visibleOutcomes))):
            curveKey = self.curveKeys[curveIndex]
            if self.graph.curve(curveKey) <> 0:
                self.graph.curve(curveKey).setEnabled(self.visibleOutcomes[curveIndex] == 1)

    def rndCorrection(self):
        """
        returns a number from 0 to 1, self.RandomSpreadType defines which distribution is to be used.
        function is used to plot data points for categorical variables
        """    
        if self.RandomSpreadType  == 'none': 
            return 0.0
        elif self.RandomSpreadType  == 'uniform': 
            return random() - 0.5
        elif self.RandomSpreadType  == 'triangle': 
            b = (1 - betavariate(1,1)) * 0.5; return choice((-b,b))
        elif self.RandomSpreadType  == 'beta': 
            b = (1 - betavariate(1,2)) * 0.5; return choice((-b,b))


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OW2DInteractions()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
