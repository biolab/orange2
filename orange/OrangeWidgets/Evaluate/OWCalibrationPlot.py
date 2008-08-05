"""
<name>Calibration Plot</name>
<description>Displays calibration plot based on evaluation of classifiers.</description>
<contact>Tomaz Curk</contact>
<icon>CalibrationPlot.png</icon>
<priority>1030</priority>
"""
import orngOrangeFoldersQt4
from OWColorPalette import ColorPixmap
from OWWidget import *
from OWGraph import *
import OWGUI

import orngTest, orngStat
import statc, math

class singleClassCalibrationPlotGraph(OWGraph):
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
        self.setYLaxisTitle('actual probability')
        self.setShowYLaxisTitle(1)
        self.setXaxisTitle('estimated probability')
        self.setShowXaxisTitle(1)
        self.setShowMainTitle(1)
        self.setMainTitle(title)
        self.dres = None
        self.numberOfClasses = None
        self.targetClass = None
        self.rugHeight = 0.02

        self.removeCurves()

    def setData(self, classifierColor, dres, targetClass):
        self.classifierColor = classifierColor
        self.dres = dres
        self.targetClass = targetClass

        classifiersNum = len(self.dres.classifierNames)
        self.removeCurves()
        self.classifierColor = classifierColor
        self.classifierNames = self.dres.classifierNames
        self.numberOfClasses = len(self.dres.classValues)

        for cNum in range(classifiersNum):
            curve = self.addCurve('', pen=QPen(self.classifierColor[cNum], 3))
            self.classifierCalibrationCKeys.append(curve)

            curve = errorBarQwtPlotCurve('', connectPoints = 0, tickXw = 0.0)
            curve.attach(self)
            curve.setSymbol(QwtSymbol(QwtSymbol.NoSymbol, QBrush(Qt.color0), QPen(self.classifierColor[cNum], 1), QSize(0,0)))
            curve.setStyle(QwtPlotCurve.UserCurve)
            self.classifierYesClassRugCKeys.append(curve)

            curve = errorBarQwtPlotCurve('', connectPoints = 0, tickXw = 0.0)
            curve.attach(self)
            curve.setSymbol(QwtSymbol(QwtSymbol.NoSymbol, QBrush(Qt.color0), QPen(self.classifierColor[cNum], 1), QSize(0,0)))
            curve.setStyle(QwtPlotCurve.UserCurve)
            self.classifierNoClassRugCKeys.append(curve)

            self.showClassifiers.append(0)

        ## compute curves for targetClass
        if (self.dres <> None): ## check that targetClass in range
            if self.targetClass < 0:
                self.targetClass = 0
            if self.targetClass >= self.numberOfClasses:
                self.targetClass = self.numberOfClasses - 1
            if self.targetClass < 0:
                self.targetClass = None ## no classes, no target

        if (self.dres == None) or (self.targetClass == None):
            self.setMainTitle("")
            for curve in self.classifierCalibrationCKeys + self.classifierYesClassRugCKeys + self.classifierNoClassRugCKeys:
                curve.setData([], [])
            return

        self.setMainTitle(self.dres.classValues[self.targetClass])
        calibrationCurves = orngStat.computeCalibrationCurve(self.dres, self.targetClass)

        classifier = 0
        for (curve, yesClassRugPoints, noClassRugPoints) in calibrationCurves:
            x = [px for (px, py) in curve]
            y = [py for (px, py) in curve]
            curve = self.classifierCalibrationCKeys[classifier]
            curve.setData(x, y)

            x = []
            y = []
            for (px, py) in yesClassRugPoints:
                n = py > 0.0 ##py
                if n:
                    py = 1.0
                    x.append(px)
                    y.append(py - self.rugHeight*n / 2.0)

                    x.append(px)
                    y.append(py)

                    x.append(px)
                    y.append(py - self.rugHeight*n)
            curve = self.classifierYesClassRugCKeys[classifier]
            curve.setData(x, y)

            x = []
            y = []
            for (px, py) in noClassRugPoints:
                n = py > 0.0 ##py
                if n:
                    py = 0.0
                    x.append(px)
                    y.append(py + self.rugHeight*n / 2.0)

                    x.append(px)
                    y.append(py + self.rugHeight*n)

                    x.append(px)
                    y.append(py)
            curve = self.classifierNoClassRugCKeys[classifier]
            curve.setData(x, y)
            classifier += 1

        self.updateCurveDisplay()

    def removeCurves(self):
        OWGraph.clear(self)
        self.classifierColor = []
        self.classifierNames = []
        self.showClassifiers = []
        self.showDiagonal = 0
        self.showRugs = 1

        self.classifierCalibrationCKeys = []
        self.classifierYesClassRugCKeys = []
        self.classifierNoClassRugCKeys = []

        ## diagonal curve
        self.diagonalCKey = self.addCurve("", pen = QPen(Qt.black, 1), style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = [0.0, 1.0], yData = [0.0, 1.0])

    def updateCurveDisplay(self):
        self.diagonalCKey.setVisible(self.showDiagonal)

        for cNum in range(len(self.showClassifiers)):
            showCNum = (self.showClassifiers[cNum] <> 0)
            self.classifierCalibrationCKeys[cNum].setVisible(showCNum)
            b = showCNum and self.showRugs
            self.classifierYesClassRugCKeys[cNum].setVisible(b)
            self.classifierNoClassRugCKeys[cNum].setVisible(b)
        self.updateLayout()
        self.update()

    def setCalibrationCurveWidth(self, v):
        for cNum in range(len(self.showClassifiers)):
            self.classifierCalibrationCKeys[cNum].setPen(QPen(self.classifierColor[cNum], v))
        self.update()

    def setShowClassifiers(self, list):
        self.showClassifiers = list
        self.updateCurveDisplay()

    def setShowDiagonal(self, v):
        self.showDiagonal = v
        self.updateCurveDisplay()

    def setShowRugs(self, v):
        self.showRugs = v
        self.updateCurveDisplay()

    def sizeHint(self):
        return QSize(170, 170)

class OWCalibrationPlot(OWWidget):
    settingsList = ["CalibrationCurveWidth", "ShowDiagonal", "ShowRugs"]
    contextHandlers = {"": EvaluationResultsContextHandler("", "targetClass", "selectedClassifiers")}
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Calibration Plot", 1)

        # inputs
        self.inputs=[("Evaluation Results", orngTest.ExperimentResults, self.results, Default)]

        #set default settings
        self.CalibrationCurveWidth = 3
        self.ShowDiagonal = TRUE
        self.ShowRugs = TRUE
        #load settings
        self.loadSettings()

        # temp variables
        self.dres = None
        self.targetClass = None
        self.numberOfClasses = 0
        self.graphs = []
        self.classifierColor = None
        self.numberOfClassifiers = 0
        self.classifiers = []
        self.selectedClassifiers = []

        # GUI
        self.graphsGridLayoutQGL = QGridLayout(self)
        import sip
        sip.delete(self.mainArea.layout())
        self.mainArea.setLayout(self.graphsGridLayoutQGL)

        ## save each ROC graph in separate file
        self.graph = None
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFile)

        ## general tab
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.generalTab = OWGUI.createTabPage(self.tabs, "General")
        self.settingsTab = OWGUI.createTabPage(self.tabs, "Settings")

        self.splitQS = QSplitter()
        self.splitQS.setOrientation(Qt.Vertical)

        ## target class
        self.classCombo = OWGUI.comboBox(self.generalTab, self, 'targetClass', box='Target class', items=[], callback=self.target)
        OWGUI.separator(self.generalTab)

        ## classifiers selection (classifiersQLB)
        self.classifiersQVGB = OWGUI.widgetBox(self.generalTab, "Classifiers")
        self.classifiersQLB = OWGUI.listBox(self.classifiersQVGB, self, "selectedClassifiers", selectionMode = QListWidget.MultiSelection, callback = self.classifiersSelectionChange)
        self.unselectAllClassifiersQLB = OWGUI.button(self.classifiersQVGB, self, "(Un)select all", callback = self.SUAclassifiersQLB)

        ## settings tab
        OWGUI.hSlider(self.settingsTab, self, 'CalibrationCurveWidth', box='Calibration Curve Width', minValue=1, maxValue=9, step=1, callback=self.setCalibrationCurveWidth, ticks=1)
        OWGUI.checkBox(self.settingsTab, self, 'ShowDiagonal', 'Show Diagonal Line', tooltip='', callback=self.setShowDiagonal)
        OWGUI.checkBox(self.settingsTab, self, 'ShowRugs', 'Show Rugs', tooltip='', callback=self.setShowRugs)
        self.settingsTab.layout().addStretch(100)

    def setCalibrationCurveWidth(self):
        for g in self.graphs:
            g.setCalibrationCurveWidth(self.CalibrationCurveWidth)

    def setShowDiagonal(self):
        for g in self.graphs:
            g.setShowDiagonal(self.ShowDiagonal)

    def setShowRugs(self):
        for g in self.graphs:
            g.setShowRugs(self.ShowRugs)

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

    def classifiersSelectionChange(self):
        list = []
        for i in range(self.classifiersQLB.count()):
            if self.classifiersQLB.item(i).isSelected():
                list.append( 1 )
            else:
                list.append( 0 )
        for g in self.graphs:
            g.setShowClassifiers(list)
    ##

    def calcAllClassGraphs(self):
        cl = 0
        for g in self.graphs:
            g.setData(self.classifierColor, self.dres, cl)

            ## user settings
            g.setCalibrationCurveWidth(self.CalibrationCurveWidth)
            g.setShowDiagonal(self.ShowDiagonal)
            g.setShowRugs(self.ShowRugs)
            cl += 1

    def removeGraphs(self):
        for g in self.graphs:
            g.removeCurves()

    def saveToFile(self):
        if self.graph:
            self.graph.saveToFile()

    def target(self):
        for g in self.graphs:
            g.hide()

        if (self.targetClass <> None) and (len(self.graphs) > 0):
            if self.targetClass >= len(self.graphs):
                self.targetClass = len(self.graphs) - 1
            if self.targetClass < 0:
                self.targetClass = 0
            self.graph = self.graphs[self.targetClass]
            self.graph.show()
            self.graphsGridLayoutQGL.addWidget(self.graph, 0, 0)
        else:
            self.graph = None

    def results(self, dres):
        self.closeContext()

        self.targeClass = None
        self.classifiersQLB.clear()
        self.removeGraphs()
        self.classCombo.clear()

        self.dres = dres

        self.graphs = []
        if self.dres <> None:
            self.numberOfClasses = len(self.dres.classValues)
            ## one graph for each class
            for i in range(self.numberOfClasses):
                graph = singleClassCalibrationPlotGraph(self.mainArea)
                graph.hide()
                self.graphs.append(graph)
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

            self.calcAllClassGraphs()

            ## update graphics
            ## classifiersQLB
            for i in range(self.numberOfClassifiers):
                newColor = self.classifierColor[i]
                self.classifiersQLB.addItem(QListWidgetItem(ColorPixmap(newColor), self.dres.classifierNames[i]))
            self.classifiersQLB.selectAll()
        else:
            self.numberOfClasses = 0
            self.classifierColor = None
            self.targetClass = None ## no results, no target
            
        if not self.targetClass:
            self.targetClass = 0
            
        self.openContext("", self.dres)
        self.target()

if __name__ == "__main__":
    a = QApplication(sys.argv)
    owdm = OWCalibrationPlot()
    owdm.show()
    a.exec_()
