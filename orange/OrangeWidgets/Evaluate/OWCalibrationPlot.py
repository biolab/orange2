"""
<name>Calibration Plot</name>
<description>Displays calibration plot based on evaluation of classifiers.</description>
<contact>Tomaz Curk</contact>
<icon>CalibrationPlot.png</icon>
<priority>1030</priority>
"""

from OWTools import *
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
            ckey = self.insertCurve('')
            self.setCurvePen(ckey, QPen(self.classifierColor[cNum], 3))
            self.classifierCalibrationCKeys.append(ckey)

            newSymbol = QwtSymbol(QwtSymbol.None, QBrush(Qt.color0), QPen(self.classifierColor[cNum], 1), QSize(0,0))
            curve = errorBarQwtPlotCurve(self, '', connectPoints = 0, tickXw = 0.0)
            ckey = self.insertCurve(curve)
            self.setCurveSymbol(ckey, newSymbol)
            self.setCurveStyle(ckey, QwtCurve.UserCurve)
            self.classifierYesClassRugCKeys.append(ckey)

            newSymbol = QwtSymbol(QwtSymbol.None, QBrush(Qt.color0), QPen(self.classifierColor[cNum], 1), QSize(0,0))
            curve = errorBarQwtPlotCurve(self, '', connectPoints = 0, tickXw = 0.0)
            ckey = self.insertCurve(curve)
            self.setCurveSymbol(ckey, newSymbol)
            self.setCurveStyle(ckey, QwtCurve.UserCurve)
            self.classifierNoClassRugCKeys.append(ckey)

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
            for ckey in self.classifierCalibrationCKeys:
                self.setCurveData(ckey, [], [])
            for ckey in self.classifierYesClassRugCKeys:
                self.setCurveData(ckey, [], [])
            for ckey in self.classifierNoClassRugCKeys:
                self.setCurveData(ckey, [], [])
            return

        self.setMainTitle(self.dres.classValues[self.targetClass])
        calibrationCurves = orngStat.computeCalibrationCurve(self.dres, self.targetClass)

        classifier = 0
        for (curve, yesClassRugPoints, noClassRugPoints) in calibrationCurves:
            x = [px for (px, py) in curve]
            y = [py for (px, py) in curve]
            ckey = self.classifierCalibrationCKeys[classifier]
            self.setCurveData(ckey, x, y)

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
            ckey = self.classifierYesClassRugCKeys[classifier]
            self.setCurveData(ckey, x, y)

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
            ckey = self.classifierNoClassRugCKeys[classifier]
            self.setCurveData(ckey, x, y)
            classifier += 1

        self.updateCurveDisplay()

    def removeCurves(self):
        OWGraph.removeCurves(self)
        self.classifierColor = []
        self.classifierNames = []
        self.showClassifiers = []
        self.showDiagonal = 0
        self.showRugs = 1

        self.classifierCalibrationCKeys = []
        self.classifierYesClassRugCKeys = []
        self.classifierNoClassRugCKeys = []

        ## diagonal curve
        self.diagonalCKey = self.insertCurve('')
        self.setCurvePen(self.diagonalCKey, QPen(Qt.black, 1))
        self.setCurveData(self.diagonalCKey, [0.0, 1.0], [0.0, 1.0])

    def updateCurveDisplay(self):
        self.curve(self.diagonalCKey).setEnabled(self.showDiagonal)

        for cNum in range(len(self.showClassifiers)):
            showCNum = (self.showClassifiers[cNum] <> 0)
            self.curve(self.classifierCalibrationCKeys[cNum]).setEnabled(showCNum)
            b = showCNum and self.showRugs
            self.curve(self.classifierYesClassRugCKeys[cNum]).setEnabled(b)
            self.curve(self.classifierNoClassRugCKeys[cNum]).setEnabled(b)

        self.updateLayout()
        self.update()

    def setCalibrationCurveWidth(self, v):
        for cNum in range(len(self.showClassifiers)):
            self.setCurvePen(self.classifierCalibrationCKeys[cNum], QPen(self.classifierColor[cNum], v))
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

        # GUI
        self.graphsGridLayoutQGL = QGridLayout(self.mainArea)
        ## save each ROC graph in separate file
        self.graph = None
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFile)

        ## general tab
        self.tabs = QTabWidget(self.controlArea, 'tabWidget')
        self.generalTab = QVGroupBox(self)
        self.splitQS = QSplitter()
        self.splitQS.setOrientation(Qt.Vertical)

        ## classifiers selection (classifiersQLB)
        self.classifiersQVGB = QVGroupBox(self.generalTab)
        self.classifiersQVGB.setTitle("Classifiers")
        self.classifiersQLB = QListBox(self.classifiersQVGB)
        self.classifiersQLB.setSelectionMode(QListBox.Multi)
        self.unselectAllClassifiersQLB = QPushButton("(Un)select All", self.classifiersQVGB)
        self.connect(self.classifiersQLB, SIGNAL("selectionChanged()"), self.classifiersSelectionChange)
        self.connect(self.unselectAllClassifiersQLB, SIGNAL("clicked()"), self.SUAclassifiersQLB)
        self.tabs.insertTab(self.generalTab, "General")

        ## settings tab
        self.settingsTab = QVGroupBox(self)
        OWGUI.hSlider(self.settingsTab, self, 'CalibrationCurveWidth', box='Calibration curve width', minValue=1, maxValue=9, step=1, callback=self.setCalibrationCurveWidth, ticks=1)
        OWGUI.checkBox(self.settingsTab, self, 'ShowDiagonal', 'Show diagonal line', tooltip='', callback=self.setShowDiagonal)
        OWGUI.checkBox(self.settingsTab, self, 'ShowRugs', 'Show rug', tooltip='', callback=self.setShowRugs)
        self.tabs.insertTab(self.settingsTab, "Settings")

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
            if qlb.isSelected(i):
                selected = 1
                break
        qlb.selectAll(not(selected))

    def SUAclassifiersQLB(self):
        self.selectUnselectAll(self.classifiersQLB)

    def classifiersSelectionChange(self):
        list = []
        for i in range(self.classifiersQLB.count()):
            if self.classifiersQLB.isSelected(i):
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

    def target(self, targetClass):
        self.targetClass = targetClass

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
        self.dres = dres
        self.classifiersQLB.clear()
        self.removeGraphs()

        self.graphs = []
        if self.dres <> None:
            self.numberOfClasses = len(self.dres.classValues)
            ## one graph for each class
            for i in range(self.numberOfClasses):
                graph = singleClassCalibrationPlotGraph(self.mainArea)
                graph.hide()
                self.graphs.append(graph)

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
                self.classifiersQLB.insertItem(ColorPixmap(newColor), self.dres.classifierNames[i])
            self.classifiersQLB.selectAll(1)
        else:
            self.numberOfClasses = 0
            self.classifierColor = None
            self.targetClass = None ## no results, no target
        if not self.targetClass:
            self.targetClass = 0
        self.target(self.targetClass)

if __name__ == "__main__":
    a = QApplication(sys.argv)
    owdm = OWCalibrationPlot()
    a.setMainWidget(owdm)
    owdm.show()
    a.exec_loop()
