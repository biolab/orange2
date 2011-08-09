"""
<name>Scatterplot Qt</name>
<description>Scatterplot visualization.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/ScatterPlot.png</icon>
<priority>1000</priority>
"""
# ScatterPlotQt.py
#
# Show data using scatterplot
#
from OWWidget import *
from OWScatterPlotGraphQt import *
from OWkNNOptimization import *
import orngVizRank
import OWGUI, OWToolbars, OWColorPalette
from orngScaleData import *
from plot.owcurve import *

###########################################################################################
##### WIDGET : Scatterplot visualization
###########################################################################################
class OWScatterPlotQt(OWWidget):
    settingsList = ["graph.pointWidth", "graph.showXaxisTitle", "graph.showYLaxisTitle", "showGridlines", "graph.showAxisScale", "graph.useAntialiasing",
                    "graph.showLegend", "graph.jitterSize", "graph.jitterContinuous", "graph.showFilledSymbols", "graph.showProbabilities",
                    "graph.alphaValue", "graph.showDistributions", "autoSendSelection", "toolbarSelection", "graph.sendSelectionOnUpdate",
                    "colorSettings", "selectedSchemaIndex", "VizRankLearnerName"]
    jitterSizeNums = [0.0, 0.1,   0.5,  1,  2 , 3,  4 , 5 , 7 ,  10,   15,   20 ,  30 ,  40 ,  50 ]

    contextHandlers = {"": DomainContextHandler("", ["attrX", "attrY",
                                                     (["attrColor", "attrShape", "attrSize"], DomainContextHandler.Optional),
                                                     ("attrLabel", DomainContextHandler.Optional + DomainContextHandler.IncludeMetaAttributes)])}

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Scatter Plot", TRUE)

        self.inputs =  [("Examples", ExampleTable, self.setData, Default), ("Example Subset", ExampleTable, self.setSubsetData), ("Attribute selection", AttributeList, self.setShownAttributes), ("Evaluation Results", orngTest.ExperimentResults, self.setTestResults), ("VizRank Learner", orange.Learner, self.setVizRankLearner)]
        self.outputs = [("Selected Examples", ExampleTable), ("Unselected Examples", ExampleTable)]

        self.graph = OWScatterPlotGraphQt(self, self.mainArea, "ScatterPlotQt")
        self.vizrank = OWVizRank(self, self.signalManager, self.graph, orngVizRank.SCATTERPLOT, "ScatterPlotQt")
        self.optimizationDlg = self.vizrank

        # local variables
        self.showGridlines = 0
        self.autoSendSelection = 1
        self.toolbarSelection = 0
        self.classificationResults = None
        self.outlierValues = None
        self.colorSettings = None
        self.selectedSchemaIndex = 0
        self.graph.sendSelectionOnUpdate = 0
        self.attributeSelectionList = None

        self.data = None
        self.subsetData = None

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.GeneralTab = OWGUI.createTabPage(self.tabs, "Main")
        self.SettingsTab = OWGUI.createTabPage(self.tabs, "Settings", canScroll = True)

        #add a graph widget
        self.mainArea.layout().addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        #x attribute
        self.attrX = ""
        self.attrXCombo = OWGUI.comboBox(self.GeneralTab, self, "attrX", "X-axis Attribute", callback = self.majorUpdateGraph, sendSelectedValue = 1, valueType = str)

        # y attribute
        self.attrY = ""
        self.attrYCombo = OWGUI.comboBox(self.GeneralTab, self, "attrY", "Y-axis Attribute", callback = self.majorUpdateGraph, sendSelectedValue = 1, valueType = str)

        # coloring
        self.attrColor = ""
        box = OWGUI.widgetBox(self.GeneralTab, "Point Color")
        self.attrColorCombo = OWGUI.comboBox(box, self, "attrColor", callback = self.updateGraph, sendSelectedValue=1, valueType = str, emptyString = "(Same color)")

        box = OWGUI.widgetBox(self.GeneralTab, "Additional Point Properties")
        # labelling
        self.attrLabel = ""
        self.attrLabelCombo = OWGUI.comboBox(box, self, "attrLabel", label = "Point label:", callback = self.updateGraph, sendSelectedValue = 1, valueType = str, emptyString = "(No labels)", indent = 10)

        # shaping
        self.attrShape = ""
        self.attrShapeCombo = OWGUI.comboBox(box, self, "attrShape", label = "Point shape:", callback = self.updateGraph, sendSelectedValue=1, valueType = str, emptyString = "(Same shape)", indent = 10)

        # sizing
        self.attrSize = ""
        self.attrSizeCombo = OWGUI.comboBox(box, self, "attrSize", label = "Point size:", callback = self.updateGraph, sendSelectedValue=1, valueType = str, emptyString = "(Same size)", indent = 10)

        self.optimizationButtons = OWGUI.widgetBox(self.GeneralTab, "Optimization dialogs", orientation = "horizontal")
        OWGUI.button(self.optimizationButtons, self, "VizRank", callback = self.vizrank.reshow, tooltip = "Opens VizRank dialog, where you can search for interesting projections with different subsets of attributes", debuggingEnabled = 0)

        g = self.graph.gui

        # zooming / selection
        self.zoomSelectToolbar = g.zoom_select_toolbar(self.GeneralTab, buttons = g.default_zoom_select_buttons + [g.Spacing, g.ShufflePoints])
        self.zoomSelectToolbar.buttons[g.SendSelection].clicked.connect(self.sendSelections)

        # ####################################
        # SETTINGS TAB
        # point width
        g.point_properties_box(self.SettingsTab)

        # #####
        # jittering options
        box2 = OWGUI.widgetBox(self.SettingsTab, "Jittering Options")
        self.jitterSizeCombo = OWGUI.comboBox(box2, self, "graph.jitterSize", label = 'Jittering size (% of size)'+'  ', orientation = "horizontal", callback = self.resetGraphData, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)
        OWGUI.checkBox(box2, self, 'graph.jitterContinuous', 'Jitter continuous attributes', callback = self.resetGraphData, tooltip = "Does jittering apply also on continuous attributes?")

        # general graph settings
        box4 = OWGUI.widgetBox(self.SettingsTab, "General Graph Settings")
        OWGUI.checkBox(box4, self, 'graph.showXaxisTitle', 'X axis title', callback = self.graph.setShowXaxisTitle)
        OWGUI.checkBox(box4, self, 'graph.showYLaxisTitle', 'Y axis title', callback = self.graph.setShowYLaxisTitle)
        OWGUI.checkBox(box4, self, 'graph.showAxisScale', 'Show axis scale', callback = self.updateGraph)
        
        g.add_widgets([g.ShowLegend, g.ShowFilledSymbols, g.ShowGridLines, g.UseAnimations, g.Antialiasing], box4)
        
        box5 = OWGUI.widgetBox(box4, orientation = "horizontal")
        OWGUI.checkBox(box5, self, 'graph.showProbabilities', 'Show probabilities'+'  ', callback = self.updateGraph, tooltip = "Show a background image with class probabilities")
        smallWidget = OWGUI.SmallWidgetLabel(box5, pixmap = 1, box = "Advanced settings", tooltip = "Show advanced settings")
        #OWGUI.rubber(box5)

        box6 = OWGUI.widgetBox(smallWidget.widget, orientation = "horizontal")
        box7 = OWGUI.widgetBox(smallWidget.widget, orientation = "horizontal")

        OWGUI.widgetLabel(box6, "Granularity:"+"  ")
        OWGUI.hSlider(box6, self, 'graph.squareGranularity', minValue=1, maxValue=10, step=1, callback = self.updateGraph)

        OWGUI.checkBox(box7, self, 'graph.spaceBetweenCells', 'Show space between cells', callback = self.updateGraph)

        self.colorButtonsBox = OWGUI.widgetBox(self.SettingsTab, "Colors", orientation = "horizontal")
        OWGUI.button(self.colorButtonsBox, self, "Set Colors", self.setColors, tooltip = "Set the canvas background color, grid color and color palette for coloring continuous variables", debuggingEnabled = 0)

        box5 = OWGUI.widgetBox(self.SettingsTab, "Tooltips Settings")
        OWGUI.comboBox(box5, self, "graph.tooltipKind", items = ["Don't Show Tooltips", "Show Visible Attributes", "Show All Attributes"], callback = self.updateGraph)

        box = OWGUI.widgetBox(self.SettingsTab, "Auto Send Selected Data When...")
        OWGUI.checkBox(box, self, 'autoSendSelection', 'Adding/Removing selection areas', callback = self.selectionChanged, tooltip = "Send selected data whenever a selection area is added or removed")
        OWGUI.checkBox(box, self, 'graph.sendSelectionOnUpdate', 'Moving/Resizing selection areas', tooltip = "Send selected data when a user moves or resizes an existing selection area")
        self.graph.autoSendSelectionCallback = self.selectionChanged

        self.GeneralTab.layout().addStretch(100)
        self.SettingsTab.layout().addStretch(100)
        self.icons = self.createAttributeIconDict()

        self.debugSettings = ["attrX", "attrY", "attrColor", "attrLabel", "attrShape", "attrSize"]
        self.wdChildDialogs = [self.vizrank]        # used when running widget debugging

        dlg = self.createColorDialog()
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette("discPalette")
        self.graph.setCanvasBackground(dlg.getColor("Canvas"))
        self.graph.gridCurve.setPen(QPen(dlg.getColor("Grid")))
        self.graph.palette.grid_style.color = dlg.getColor("Grid")

        self.graph.enableGridXB(self.showGridlines)
        self.graph.enableGridYL(self.showGridlines)

        #self.SettingsTab.resize(self.SettingsTab.sizeHint())

        self.resize(700, 550)


    def settingsFromWidgetCallback(self, handler, context):
        context.selectionPolygons = []
        for curve in self.graph.selectionCurveList:
            xs = [curve.x(i) for i in range(curve.dataSize())]
            ys = [curve.y(i) for i in range(curve.dataSize())]
            context.selectionPolygons.append((xs, ys))

    def settingsToWidgetCallback(self, handler, context):
        selections = getattr(context, "selectionPolygons", [])
        for (xs, ys) in selections:
            c = SelectionCurve("")
            c.setData(xs,ys)
            c.attach(self.graph)
            self.graph.selectionCurveList.append(c)

    # ##############################################################################################################################################################
    # SCATTERPLOT SIGNALS
    # ##############################################################################################################################################################

    def resetGraphData(self):
        self.graph.rescaleData()
        self.majorUpdateGraph()

    # receive new data and update all fields
    def setData(self, data):
        if data is not None and (len(data) == 0 or len(data.domain) == 0):
            data = None
        if self.data and data and self.data.checksum() == data.checksum():
            return    # check if the new data set is the same as the old one

        self.closeContext()
        sameDomain = self.data and data and data.domain.checksum() == self.data.domain.checksum() # preserve attribute choice if the domain is the same
        self.data = data
        self.vizrank.clearResults()
        if not sameDomain:
            self.initAttrValues()
        self.graph.insideColors = None
        self.classificationResults = None
        self.outlierValues = None
        self.openContext("", self.data)

    # set an example table with a data subset subset of the data. if called by a visual classifier, the update parameter will be 0
    def setSubsetData(self, subsetData):
        self.subsetData = subsetData
        self.vizrank.clearArguments()

    # this is called by OWBaseWidget after setData and setSubsetData are called. this way the graph is updated only once
    def handleNewSignals(self):
        self.graph.setData(self.data, self.subsetData)
        self.vizrank.resetDialog()
        if self.attributeSelectionList and 0 not in [self.graph.attributeNameIndex.has_key(attr) for attr in self.attributeSelectionList]:
            self.attrX = self.attributeSelectionList[0]
            self.attrY = self.attributeSelectionList[1]
        self.attributeSelectionList = None
        self.updateGraph()
        self.sendSelections()


    # receive information about which attributes we want to show on x and y axis
    def setShownAttributes(self, list):
        if list and len(list[:2]) == 2:
            self.attributeSelectionList = list[:2]
        else:
            self.attributeSelectionList = None


    # visualize the results of the classification
    def setTestResults(self, results):
        self.classificationResults = None
        if isinstance(results, orngTest.ExperimentResults) and len(results.results) > 0 and len(results.results[0].probabilities) > 0:
            self.classificationResults = [results.results[i].probabilities[0][results.results[i].actualClass] for i in range(len(results.results))]
            self.classificationResults = (self.classificationResults, "Probability of correct classification = %.2f%%")


    # set the learning method to be used in VizRank
    def setVizRankLearner(self, learner):
        self.vizrank.externalLearner = learner

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        (selected, unselected) = self.graph.getSelectionsAsExampleTables([self.attrX, self.attrY])
        self.send("Selected Examples",selected)
        self.send("Unselected Examples",unselected)


    # ##############################################################################################################################################################
    # CALLBACKS FROM VIZRANK DIALOG
    # ##############################################################################################################################################################

    def showSelectedAttributes(self):
        val = self.vizrank.getSelectedProjection()
        if not val: return
        if self.data.domain.classVar:
            self.attrColor = self.data.domain.classVar.name
        self.majorUpdateGraph(val[3])

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

        if not self.data: return

        self.attrColorCombo.addItem("(Same color)")
        self.attrLabelCombo.addItem("(No labels)")
        self.attrShapeCombo.addItem("(Same shape)")
        self.attrSizeCombo.addItem("(Same size)")

        #labels are usually chosen from meta variables, put them on top
        for metavar in [self.data.domain.getmeta(mykey) for mykey in self.data.domain.getmetas().keys()]:
            self.attrLabelCombo.addItem(self.icons[metavar.varType], metavar.name)

        contList = []
        discList = []
        for attr in self.data.domain:
            if attr.varType in [orange.VarTypes.Discrete, orange.VarTypes.Continuous]:
                self.attrXCombo.addItem(self.icons[attr.varType], attr.name)
                self.attrYCombo.addItem(self.icons[attr.varType], attr.name)
                self.attrColorCombo.addItem(self.icons[attr.varType], attr.name)
                self.attrSizeCombo.addItem(self.icons[attr.varType], attr.name)
            if attr.varType == orange.VarTypes.Discrete: 
                self.attrShapeCombo.addItem(self.icons[attr.varType], attr.name)
            self.attrLabelCombo.addItem(self.icons[attr.varType], attr.name)

        self.attrX = str(self.attrXCombo.itemText(0))
        if self.attrYCombo.count() > 1: self.attrY = str(self.attrYCombo.itemText(1))
        else:                           self.attrY = str(self.attrYCombo.itemText(0))

        if self.data.domain.classVar and self.data.domain.classVar.varType in [orange.VarTypes.Discrete, orange.VarTypes.Continuous]:
            self.attrColor = self.data.domain.classVar.name
        else:
            self.attrColor = ""
        self.attrShape = ""
        self.attrSize= ""
        self.attrLabel = ""

    def majorUpdateGraph(self, attrList = None, insideColors = None, **args):
        self.graph.removeAllSelections()
        self.updateGraph(attrList, insideColors, **args)

    def updateGraph(self, attrList = None, insideColors = None, **args):
        self.graph.zoomStack = []
        if not self.graph.haveData:
            return

        if attrList and len(attrList) == 2:
            self.attrX = attrList[0]
            self.attrY = attrList[1]

        if self.graph.dataHasDiscreteClass and (self.vizrank.showKNNCorrectButton.isChecked() or self.vizrank.showKNNWrongButton.isChecked()):
            kNNExampleAccuracy, probabilities = self.vizrank.kNNClassifyData(self.graph.createProjectionAsExampleTable([self.graph.attributeNameIndex[self.attrX], self.graph.attributeNameIndex[self.attrY]]))
            if self.vizrank.showKNNCorrectButton.isChecked(): kNNExampleAccuracy = ([1.0 - val for val in kNNExampleAccuracy], "Probability of wrong classification = %.2f%%")
            else: kNNExampleAccuracy = (kNNExampleAccuracy, "Probability of correct classification = %.2f%%")
        else:
            kNNExampleAccuracy = None

        self.graph.insideColors = insideColors or self.classificationResults or kNNExampleAccuracy or self.outlierValues
        self.graph.updateData(self.attrX, self.attrY, self.attrColor, self.attrShape, self.attrSize, self.attrLabel)


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
        
    def setShowGridlines(self):
        self.graph.enableGridXB(self.showGridlines)
        self.graph.enableGridYL(self.showGridlines)

    def selectionChanged(self):
        self.zoomSelectToolbar.buttons[OWPlotGUI.SendSelection].setEnabled(not self.autoSendSelection)
        if self.autoSendSelection:
            self.sendSelections()

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.graph.contPalette = dlg.getContinuousPalette("contPalette")
            self.graph.discPalette = dlg.getDiscretePalette("discPalette")
            self.graph.setCanvasBackground(dlg.getColor("Canvas"))
            self.graph.setGridColor(dlg.getColor("Grid"))
            self.updateGraph()

    def createColorDialog(self):
        c = OWColorPalette.ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        c.createContinuousPalette("contPalette", "Continuous Palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, "Canvas", "Canvas color", Qt.white)
        box.layout().addSpacing(5)
        c.createColorButton(box, "Grid", "Grid color", QColor(215,215,215))
        box.layout().addSpacing(5)
        c.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return c

    def closeEvent(self, ce):
        self.vizrank.close()
        OWWidget.closeEvent(self, ce)


    def sendReport(self):
        self.startReport("%s [%s - %s]" % (self.windowTitle(), self.attrX, self.attrY))
        self.reportSettings("Visualized attributes",
                            [("X", self.attrX),
                             ("Y", self.attrY),
                             self.attrColor and ("Color", self.attrColor),
                             self.attrLabel and ("Label", self.attrLabel),
                             self.attrShape and ("Shape", self.attrShape),
                             self.attrSize and ("Size", self.attrSize)])
        self.reportSettings("Settings",
                            [("Symbol size", self.graph.pointWidth),
                             ("Transparency", self.graph.alphaValue),
                             ("Jittering", self.graph.jitterSize),
                             ("Jitter continuous attributes", OWGUI.YesNo[self.graph.jitterContinuous])])
        self.reportSection("Graph")
        self.reportImage(self.graph.saveToFileDirect, QSize(400, 400))

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlotQt()
    ow.show()
    data = orange.ExampleTable(r"../../doc/datasets/brown-selected.tab")
    ow.setData(data)
    #ow.setData(orange.ExampleTable("..\\..\\doc\\datasets\\wine.tab"))
    ow.handleNewSignals()
    a.exec_()
    #save settings
    ow.saveSettings()
