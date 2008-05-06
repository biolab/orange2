"""
<name>Scatterplot</name>
<description>Scatterplot visualization.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/ScatterPlot.png</icon>
<priority>1000</priority>
"""
# ScatterPlot.py
#
# Show data using scatterplot
#

from OWWidget import *
from OWScatterPlotGraph import *
from OWkNNOptimization import *
import orngVizRank
##from OWClusterOptimization import *
import OWGUI, OWToolbars, OWDlgs
from orngScaleData import *
from OWGraph import OWGraph


###########################################################################################
##### WIDGET : Scatterplot visualization
###########################################################################################
class OWScatterPlot(OWWidget):
    settingsList = ["graph.pointWidth", "graph.showXaxisTitle", "graph.showYLaxisTitle", "showGridlines", "graph.showAxisScale",
                    "graph.showLegend", "graph.jitterSize", "graph.jitterContinuous", "graph.showFilledSymbols", "graph.showProbabilities",
                    "graph.showDistributions", "autoSendSelection", "toolbarSelection",
                    "colorSettings", "selectedSchemaIndex", "VizRankLearnerName", "showProbabilitiesDetails"]
    jitterSizeNums = [0.0, 0.1,   0.5,  1,  2 , 3,  4 , 5 , 7 ,  10,   15,   20 ,  30 ,  40 ,  50 ]

    contextHandlers = {"": DomainContextHandler("", ["attrX", "attrY",
                                                     (["attrColor", "attrShape", "attrSize"], DomainContextHandler.Optional),
                                                     ("attrLabel", DomainContextHandler.Optional + DomainContextHandler.IncludeMetaAttributes)])}

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "ScatterPlot", TRUE)

        self.inputs =  [("Examples", ExampleTable, self.setData, Default), ("Example Subset", ExampleTable, self.setSubsetData), ("Attribute selection", AttributeList, self.setShownAttributes), ("Evaluation Results", orngTest.ExperimentResults, self.setTestResults), ("VizRank Learner", orange.Learner, self.setVizRankLearner)]
        self.outputs = [("Selected Examples", ExampleTable), ("Unselected Examples", ExampleTable)]

        # local variables
        self.showGridlines = 0
        self.autoSendSelection = 1
        self.toolbarSelection = 0
        self.classificationResults = None
        self.outlierValues = None
        self.colorSettings = None
        self.selectedSchemaIndex = 0
        self.showProbabilitiesDetails = 0

        self.boxGeneral = 1

        self.graph = OWScatterPlotGraph(self, self.mainArea, "ScatterPlot")
        self.vizrank = OWVizRank(self, self.signalManager, self.graph, orngVizRank.SCATTERPLOT, "ScatterPlot")
##        self.clusterDlg = ClusterOptimization(self, self.signalManager, self.graph, "ScatterPlot")
        self.optimizationDlg = self.vizrank

        self.data = None
        self.unprocessedSubsetData = None

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = QTabWidget(self.controlArea, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "Main")
        self.tabs.insertTab(self.SettingsTab, "Settings")

        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        #x attribute
        self.attrX = ""
        self.attrXCombo = OWGUI.comboBox(self.GeneralTab, self, "attrX", "X-axis attribute", callback = self.majorUpdateGraph, sendSelectedValue = 1, valueType = str)

        # y attribute
        self.attrY = ""
        self.attrYCombo = OWGUI.comboBox(self.GeneralTab, self, "attrY", "Y-axis attribute", callback = self.majorUpdateGraph, sendSelectedValue = 1, valueType = str)

        # coloring
        self.showColorLegend = 0
        self.attrColor = ""
        box = OWGUI.widgetBox(self.GeneralTab, "Colors")
        OWGUI.checkBox(box, self, 'showColorLegend', 'Show color legend', callback = self.updateGraph)
        self.attrColorCombo = OWGUI.comboBox(box, self, "attrColor", callback = self.updateGraph, sendSelectedValue=1, valueType = str, emptyString = "(Same color)")

        # labelling
        self.attrLabel = ""
        self.attrLabelCombo = OWGUI.comboBox(self.GeneralTab, self, "attrLabel", "Point labelling", callback = self.updateGraph, sendSelectedValue = 1, valueType = str, emptyString = "(No labels)")

        # shaping
        self.attrShape = ""
        self.attrShapeCombo = OWGUI.comboBox(self.GeneralTab, self, "attrShape", "Shape", callback = self.updateGraph, sendSelectedValue=1, valueType = str, emptyString = "(Same shape)")

        # sizing
        self.attrSize = ""
        self.attrSizeCombo = OWGUI.comboBox(self.GeneralTab, self, "attrSize", "Size", callback = self.updateGraph, sendSelectedValue=1, valueType = str, emptyString = "(Same size)")

##        # cluster dialog
##        self.clusterDlg.label1.hide()
##        self.clusterDlg.optimizationTypeCombo.hide()
##        self.clusterDlg.attributeCountCombo.hide()
##        self.clusterDlg.attributeLabel.hide()
##        self.graph.clusterOptimization = self.clusterDlg


        self.optimizationButtons = OWGUI.widgetBox(self.GeneralTab, "Optimization dialogs", orientation = "horizontal")
        OWGUI.button(self.optimizationButtons, self, "VizRank", callback = self.vizrank.reshow, tooltip = "Opens VizRank dialog in which you can search for interesting projections with different subsets of attributes.", debuggingEnabled = 0)

##        OWGUI.button(self.optimizationButtons, self, "Cluster", callback = self.clusterDlg.reshow, debuggingEnabled = 0)
##        self.connect(self.clusterDlg.startOptimizationButton , SIGNAL("clicked()"), self.optimizeClusters)
##        self.connect(self.clusterDlg.resultList, SIGNAL("selectionChanged()"),self.showSelectedCluster)
##        self.graph.clusterOptimization = self.clusterDlg

        # zooming / selection
        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        # ####################################
        # SETTINGS TAB
        # point width
        OWGUI.hSlider(self.SettingsTab, self, 'graph.pointWidth', box='Symbol size', minValue=1, maxValue=20, step=1, callback = self.pointSizeChange)

        # #####
        # jittering options
        box2 = OWGUI.widgetBox(self.SettingsTab, "Jittering options")
        box3 = OWGUI.widgetBox(box2, orientation = "horizontal")
        self.jitterLabel = QLabel('Jittering size (% of size)'+'  ', box3)
        self.jitterSizeCombo = OWGUI.comboBox(box3, self, "graph.jitterSize", callback = self.resetGraphData, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)
        OWGUI.checkBox(box2, self, 'graph.jitterContinuous', 'Jitter continuous attributes', callback = self.resetGraphData, tooltip = "Does jittering apply also on continuous attributes?")

        # general graph settings
        box4 = OWGUI.collapsableWidgetBox(self.SettingsTab, "General graph settings", self, "boxGeneral")
        OWGUI.checkBox(box4, self, 'graph.showXaxisTitle', 'X axis title', callback = self.graph.setShowXaxisTitle)
        OWGUI.checkBox(box4, self, 'graph.showYLaxisTitle', 'Y axis title', callback = self.graph.setShowYLaxisTitle)
        OWGUI.checkBox(box4, self, 'graph.showAxisScale', 'Show axis scale', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showLegend', 'Show legend', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showFilledSymbols', 'Show filled symbols', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'showGridlines', 'Show gridlines', callback = self.setShowGridlines)
##        OWGUI.checkBox(box4, self, 'graph.showClusters', 'Show clusters', callback = self.updateGraph, tooltip = "Show a line boundary around a significant cluster")

        box5 = OWGUI.widgetBox(box4, orientation = "horizontal")
        OWGUI.checkBox(box5, self, 'graph.showProbabilities', 'Show probabilities'+'  ', callback = self.updateGraph, tooltip = "Show a background image with class probabilities")
        hider = OWGUI.widgetHider(box5, self, "showProbabilitiesDetails", tooltip = "Show/hide extra settings")
        rubb = OWGUI.rubber(box5)
        rubb.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum))

        box6 = OWGUI.widgetBox(box4, orientation = "horizontal")
        OWGUI.separator(box6, width = 20)
        OWGUI.label(box6, self, "Granularity:"+"  ")
        OWGUI.hSlider(box6, self, 'graph.squareGranularity', minValue=1, maxValue=10, step=1, callback = self.updateGraph)

        box7 = OWGUI.widgetBox(box4, orientation = "horizontal")
        OWGUI.separator(box7, 17)
        OWGUI.checkBox(box7, self, 'graph.spaceBetweenCells', 'Show space between cells', callback = self.updateGraph)
        hider.setWidgets([box6, box7])

        box4.syncControls()

        self.colorButtonsBox = OWGUI.widgetBox(self.SettingsTab, "Colors", orientation = "horizontal")
        OWGUI.button(self.colorButtonsBox, self, "Set Colors", self.setColors, tooltip = "Set the canvas background color, grid color and color palette for coloring continuous variables", debuggingEnabled = 0)

        box5 = OWGUI.widgetBox(self.SettingsTab, "Tooltips settings")
        OWGUI.comboBox(box5, self, "graph.tooltipKind", items = ["Don't Show Tooltips", "Show Visible Attributes", "Show All Attributes"], callback = self.updateGraph)

        OWGUI.checkBox(self.SettingsTab, self, 'autoSendSelection', 'Auto send selected data', box = "Data selection", callback = self.setAutoSendSelection, tooltip = "Send signals with selected data whenever the selection changes.")
        self.graph.selectionChangedCallback = self.setAutoSendSelection

        OWGUI.rubber(self.SettingsTab)
        self.SettingsTab.setMinimumWidth(max(self.GeneralTab.sizeHint().width(), self.SettingsTab.sizeHint().width())+20)
        self.icons = self.createAttributeIconDict()

        self.debugSettings = ["attrX", "attrY", "attrColor", "attrLabel", "attrShape", "attrSize"]
        self.activateLoadedSettings()
        self.resize(700, 550)
        self.wdChildDialogs = [self.vizrank]        # used when running widget debugging


    def activateLoadedSettings(self):
        dlg = self.createColorDialog()
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette()
        self.graph.setCanvasBackground(dlg.getColor("Canvas"))
        self.graph.setGridPen(QPen(dlg.getColor("Grid")))

        self.graph.enableGridXB(self.showGridlines)
        self.graph.enableGridYL(self.showGridlines)

        apply([self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection, self.zoomSelectToolbar.actionPolygonSelection][self.toolbarSelection], [])

    def settingsFromWidgetCallback(self, handler, context):
        context.selectionPolygons = []
        for key in self.graph.selectionCurveKeyList:
            curve = self.graph.curve(key)
            xs = [curve.x(i) for i in range(curve.dataSize())]
            ys = [curve.y(i) for i in range(curve.dataSize())]
            context.selectionPolygons.append((xs, ys))

    def settingsToWidgetCallback(self, handler, context):
        selections = getattr(context, "selectionPolygons", [])
        for (xs, ys) in selections:
            c = SelectionCurve(self.graph)
            c.setData(xs,ys)
            key = self.graph.insertCurve(c)
            self.graph.selectionCurveKeyList.append(key)

    # ##############################################################################################################################################################
    # SCATTERPLOT SIGNALS
    # ##############################################################################################################################################################

    def resetGraphData(self):
        orngScaleScatterPlotData.setData(self.graph, self.data)
        self.majorUpdateGraph()

    # receive new data and update all fields
    def setData(self, data, clearResults = 1, onlyDataSubset = 0):
        if data:
            name = getattr(data, "name", "")
            data = data.filterref(orange.Filter_hasClassValue())
            data.name = name
            if len(data) == 0 or len(data.domain) == 0:        # if we don't have any examples or attributes then this is not a valid data set
                data = None
        if self.data != None and data != None and self.data.checksum() == data.checksum():
            return    # check if the new data set is the same as the old one

        self.closeContext()
        self.graph.clear()

        exData = self.data
        self.data = data
        self.graph.insideColors = None
##        self.graph.clusterClosure = None
        self.classificationResults = None
        self.outlierValues = None

        self.vizrank.setData(data)
##        self.clusterDlg.setData(data, clearResults)

        sameDomain = self.data and exData and exData.domain.checksum() == self.data.domain.checksum() # preserve attribute choice if the domain is the same
        if not sameDomain:
            self.initAttrValues()

        self.openContext("", data)

        if data and self.unprocessedSubsetData:        # if we first received subset data we now have to call setSubsetData to process it
            self.setSubsetData(self.unprocessedSubsetData)
            self.unprocessedSubsetData = None


    # set an example table with a data subset subset of the data. if called by a visual classifier, the update parameter will be 0
    def setSubsetData(self, data):
        self.warning(10)

        # if we haven't yet received the main data we just remember this data and return
        if not self.data:
            self.unprocessedSubsetData = data
            return

        # check if the new data set is the same as the old one
        if self.graph.rawSubsetData != None and data != None and self.graph.rawSubsetData.checksum() == data.checksum():
            return

        try:
            subsetData = data.select(self.data.domain)
        except:
            subsetData = None
            self.warning(10, data and "'Examples' and 'Example Subset' data do not have copatible domains. Unable to draw 'Example Subset' data." or "")

        self.vizrank.setSubsetData(subsetData)
##        self.clusterDlg.setSubsetData(data)
#        qApp.processEvents()            # TODO: find out why scatterplot crashes if we remove this line and send a subset of data that is not in self.rawData - as in cluster argumentation

    # this is called by OWBaseWidget after setData and setSubsetData are called. this way the graph is updated only once
    def handleNewSignals(self):
        self.updateGraph()
        self.sendSelections()


    # receive information about which attributes we want to show on x and y axis
    def setShownAttributes(self, list):
        if not self.data or not list or len(list) < 2: return
        self.attrX = list[0]
        self.attrY = list[1]
        self.majorUpdateGraph()


    # visualize the results of the classification
    def setTestResults(self, results):
        self.classificationResults = None
        if isinstance(results, orngTest.ExperimentResults) and len(results.results) > 0 and len(results.results[0].probabilities) > 0:
            self.classificationResults = [results.results[i].probabilities[0][results.results[i].actualClass] for i in range(len(results.results))]
            self.classificationResults = (self.classificationResults, "Probability of correct classification = %.2f%%")

        self.updateGraph()

    # set the learning method to be used in VizRank
    def setVizRankLearner(self, learner):
        self.vizrank.externalLearner = learner

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        (selected, unselected) = self.graph.getSelectionsAsExampleTables([self.attrX, self.attrY])
        self.send("Selected Examples",selected)
        self.send("Unselected Examples",unselected)


    # ##############################################################################################################################################################
    # KNN OPTIMIZATION BUTTON EVENTS
    # ##############################################################################################################################################################

##    # ################################################################################################
##    # find projections that have tight clusters of points that belong to the same class value
##    def optimizeClusters(self):
##        if self.data == None: return
##
##        self.clusterDlg.clearResults()
##        self.clusterDlg.clusterStabilityButton.setOn(0)
##        self.clusterDlg.pointStability = None
##        self.clusterDlg.disableControls()
##
##        try:
##            attributeNameOrder = self.clusterDlg.getEvaluatedAttributes(self.data)
##            self.graph.getOptimalClusters(attributeNameOrder, self.clusterDlg.addResult)    # evaluate projections
##        except:
##            type, val, traceback = sys.exc_info()
##            sys.excepthook(type, val, traceback)  # print the exception
##
##        self.clusterDlg.enableControls()
##        self.clusterDlg.finishedAddingResults()


    def showSelectedAttributes(self):
        val = self.vizrank.getSelectedProjection()
        if not val: return
        (accuracy, other_results, tableLen, attrs, tryIndex, generalDict) = val

        if self.data.domain.classVar:
            self.attrColor = self.data.domain.classVar.name

        self.majorUpdateGraph(attrs)


##    def showSelectedCluster(self):
##        val = self.clusterDlg.getSelectedCluster()
##        if not val: return
##        (value, closure, vertices, attrList, classValue, enlargedClosure, other, strList) = val
##
##        if self.clusterDlg.clusterStabilityButton.isOn():
##            validData = self.graph.getValidList([self.graph.attributeNames.index(self.attrX), self.graph.attributeNames.index(self.attrY)])
##            insideColors = (numpy.compress(validData, self.clusterDlg.pointStability), "Point inside a cluster in %.2f%%")
##        else: insideColors = None
##
##        self.majorUpdateGraph(attrList, insideColors, (closure, enlargedClosure, classValue))


    # ##############################################################################################################################################################
    # ATTRIBUTE SELECTION
    # ##############################################################################################################################################################

    def getShownAttributeList(self):
        return [self.attrX, self.attrY]

    def initAttrValues(self):
        self.attrXCombo.clear()
        self.attrYCombo.clear()
        self.attrColorCombo.clear()
        self.attrLabelCombo.clear()
        self.attrShapeCombo.clear()
        self.attrSizeCombo.clear()

        if self.data == None: return

        self.attrColorCombo.insertItem("(Same color)")
        self.attrLabelCombo.insertItem("(No labels)")
        self.attrShapeCombo.insertItem("(Same shape)")
        self.attrSizeCombo.insertItem("(Same size)")

        #labels are usually chosen from meta variables, put them on top
        for metavar in [self.data.domain.getmeta(mykey) for mykey in self.data.domain.getmetas().keys()]:
            self.attrLabelCombo.insertItem(self.icons[metavar.varType], metavar.name)

        contList = []
        discList = []
        for attr in self.data.domain:
            self.attrXCombo.insertItem(self.icons[attr.varType], attr.name)
            self.attrYCombo.insertItem(self.icons[attr.varType], attr.name)
            self.attrColorCombo.insertItem(self.icons[attr.varType], attr.name)
            self.attrSizeCombo.insertItem(self.icons[attr.varType], attr.name)
            if attr.varType == orange.VarTypes.Discrete: self.attrShapeCombo.insertItem(self.icons[attr.varType], attr.name)
            self.attrLabelCombo.insertItem(self.icons[attr.varType], attr.name)

        self.attrX = str(self.attrXCombo.text(0))
        if self.attrYCombo.count() > 1: self.attrY = str(self.attrYCombo.text(1))
        else:                           self.attrY = str(self.attrYCombo.text(0))

        if self.data.domain.classVar:
            self.attrColor = self.data.domain.classVar.name
        else:
            self.attrColor = ""
        self.attrShape = ""
        self.attrSize= ""
        self.attrLabel = ""

    def majorUpdateGraph(self, attrList = None, insideColors = None, clusterClosure = None, **args):
        self.graph.removeAllSelections()
        self.updateGraph(attrList, insideColors, clusterClosure, **args)

    def updateGraph(self, attrList = None, insideColors = None, clusterClosure = None, **args):
        self.graph.zoomStack = []
        if not self.data:
            return

        if attrList and len(attrList) == 2:
            self.attrX = attrList[0]
            self.attrY = attrList[1]

        hasDiscreteClass = self.data.domain.classVar and self.data.domain.classVar.varType == orange.VarTypes.Discrete
        if hasDiscreteClass and (self.vizrank.showKNNCorrectButton.isOn() or self.vizrank.showKNNWrongButton.isOn()):
            kNNExampleAccuracy, probabilities = self.vizrank.kNNClassifyData(self.graph.createProjectionAsExampleTable([self.graph.attributeNameIndex[self.attrX], self.graph.attributeNameIndex[self.attrY]]))
            if self.vizrank.showKNNCorrectButton.isOn(): kNNExampleAccuracy = ([1.0 - val for val in kNNExampleAccuracy], "Probability of wrong classification = %.2f%%")
            else: kNNExampleAccuracy = (kNNExampleAccuracy, "Probability of correct classification = %.2f%%")
        else:
            kNNExampleAccuracy = None

        self.graph.insideColors = insideColors or self.classificationResults or kNNExampleAccuracy or self.outlierValues
##        self.graph.clusterClosure = clusterClosure

        self.graph.updateData(self.attrX, self.attrY, self.attrColor, self.attrShape, self.attrSize, self.showColorLegend, self.attrLabel)
        self.graph.repaint()


    # ##############################################################################################################################################################
    # SCATTERPLOT SETTINGS
    # ##############################################################################################################################################################
    def saveSettings(self):
        OWWidget.saveSettings(self)
        self.vizrank.saveSettings()

    #update status on progress bar - gets called by OWScatterplotGraph
    def updateProgress(self, current, total):
        self.progressBar.setTotalSteps(total)
        self.progressBar.setProgress(current)

    def pointSizeChange(self):
        if self.attrSize == "":
            self.replotCurves()
        else:
            self.updateGraph()

    def replotCurves(self):
        for key in self.graph.curveKeys():
            symbol = self.graph.curveSymbol(key)
            self.graph.setCurveSymbol(key, QwtSymbol(symbol.style(), symbol.brush(), symbol.pen(), QSize(self.graph.pointWidth, self.graph.pointWidth)))
        self.graph.repaint()

    def setShowGridlines(self):
        self.graph.enableGridXB(self.showGridlines)
        self.graph.enableGridYL(self.showGridlines)

    def setAutoSendSelection(self):
        if self.autoSendSelection:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(0)
            self.sendSelections()
        else:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(1)

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_loop():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.graph.contPalette = dlg.getContinuousPalette("contPalette")
            self.graph.discPalette = dlg.getDiscretePalette()
            self.graph.setCanvasBackground(dlg.getColor("Canvas"))
            self.graph.setGridPen(QPen(dlg.getColor("Grid")))
            self.updateGraph()

    def createColorDialog(self):
        c = OWDlgs.ColorPalette(self, "Color Palette")
        c.createDiscretePalette("Discrete Palette")
        c.createContinuousPalette("contPalette", "Continuous palette")
        box = c.createBox("otherColors", "Other colors")
        c.createColorButton(box, "Canvas", "Canvas color", Qt.white)
        box.addSpace(5)
        c.createColorButton(box, "Grid", "Grid color", Qt.black)
        box.addSpace(5)
        box.adjustSize()
        c.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return c

    def destroy(self, dw = 1, dsw = 1):
##        self.clusterDlg.hide()
        self.vizrank.hide()
        OWWidget.destroy(self, dw, dsw)

    def hasDiscreteClass(self, data = -1):
        if data == -1: data = self.data
        return data and data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete


#test widget appearance
if __name__=="__main__":
#    a=QApplication(sys.argv)
    a=QApplication([])
    ow=OWScatterPlot()
    a.setMainWidget(ow)
    ow.show()
    ow.setData(orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\wine.tab"))
    #ow.setData(orange.ExampleTable("..\\..\\doc\\datasets\\wine.tab"))
    ow.handleNewSignals()
    a.exec_loop()
    #save settings
    ow.saveSettings()
