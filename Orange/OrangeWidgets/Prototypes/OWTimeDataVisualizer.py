"""
<name>Time Data Visualizer</name>
<description>Visualization of time data.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/ScatterPlot.png</icon>
<priority>3300</priority>
"""
# Time Data Visualizer.py
#
# Show mulitple line visualizations
#
from OWWidget import *
from OWTimeDataVisualizerGraph import *
import OWGUI, OWToolbars, OWColorPalette
from orngScaleData import *
from OWGraph import OWGraph


###########################################################################################
##### WIDGET : Scatterplot visualization
###########################################################################################
class OWTimeDataVisualizer(OWWidget):
    settingsList = ["graph.pointWidth", "graph.showXaxisTitle",
                    "graph.showLegend", "graph.useAntialiasing", 'graph.drawPoints', 'graph.drawLines',
                    "graph.alphaValue", "autoSendSelection", "toolbarSelection", "graph.trackExamples"
                    "colorSettings", "selectedSchemaIndex"]
    contextHandlers = {"": DomainContextHandler("", ["graph.timeAttr", "graph.attributes", "graph.shownAttributeIndices"], loadImperfect = 0)}

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Time Data Visualizer", TRUE)

        self.inputs =  [("Data", ExampleTable, self.setData, Default), ("Data Subset", ExampleTable, self.setSubsetData), ("Features", AttributeList, self.setShownAttributes)]
        self.outputs = [("Selected Data", ExampleTable), ("Other Data", ExampleTable)]

        # local variables
        self.autoSendSelection = 1
        self.toolbarSelection = 0
        self.colorSettings = None
        self.selectedSchemaIndex = 0

        self.graph = OWTimeDataVisualizerGraph(self, self.mainArea, "Time Data Visualizer")

        self.data = None
        self.subsetData = None

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.GeneralTab = OWGUI.createTabPage(self.tabs, "Main")
        self.SettingsTab = OWGUI.createTabPage(self.tabs, "Settings")

        #add a graph widget
        self.mainArea.layout().addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        #x attribute
        self.attrX = ""
        box = OWGUI.widgetBox(self.GeneralTab, "Time Attribute")
        # add an option to use the index of the example as time stamp
        #OWGUI.checkBox(box, self, 'graph.use', 'X axis title', callback = self.graph.setShowXaxisTitle)
        self.timeCombo = OWGUI.comboBox(box, self, "graph.timeAttr", callback = self.majorUpdateGraph, sendSelectedValue = 1, valueType = str)

        self.colorCombo = OWGUI.comboBox(self.GeneralTab, self, "graph.colorAttr", "Point Color", callback = self.updateGraph, sendSelectedValue = 1, valueType = str)

        # y attributes
        box = OWGUI.widgetBox(self.GeneralTab, "Visualized attributes")
        OWGUI.listBox(box, self, "graph.shownAttributeIndices", "graph.attributes", selectionMode = QListWidget.ExtendedSelection, sizeHint = QSize(150, 250))
        OWGUI.button(box, self, "Update listbox changes", callback = self.majorUpdateGraph)


        # zooming / selection
        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        # ####################################
        # SETTINGS TAB
        # point width
        pointBox = OWGUI.widgetBox(self.SettingsTab, "Point properties")
        OWGUI.hSlider(pointBox, self, 'graph.pointWidth', label = "Symbol size:   ", minValue=1, maxValue=10, step=1, callback = self.pointSizeChange)
        OWGUI.hSlider(pointBox, self, 'graph.alphaValue', label = "Transparency: ", minValue=0, maxValue=255, step=10, callback = self.alphaChange)

        # general graph settings
        box4 = OWGUI.widgetBox(self.SettingsTab, "General graph settings")
        OWGUI.checkBox(box4, self, 'graph.drawLines', 'Draw lines', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.drawPoints', 'Draw points (slower)', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.trackExamples', 'Track examples', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showGrayRects', 'Show gray rectangles', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showXaxisTitle', 'Show x axis title', callback = self.graph.setShowXaxisTitle)
        OWGUI.checkBox(box4, self, 'graph.showLegend', 'Show legend', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.useAntialiasing', 'Use antialiasing', callback = self.antialiasingChange)

        self.colorButtonsBox = OWGUI.widgetBox(self.SettingsTab, "Colors", orientation = "horizontal")
        OWGUI.button(self.colorButtonsBox, self, "Set Colors", self.setColors, tooltip = "Set the canvas background color, grid color and color palette for coloring continuous variables", debuggingEnabled = 0)

        box5 = OWGUI.widgetBox(self.SettingsTab, "Tooltips settings")
        OWGUI.comboBox(box5, self, "graph.tooltipKind", items = ["Don't Show Tooltips", "Show Visible Attributes", "Show All Attributes"], callback = self.updateGraph)

        OWGUI.checkBox(self.SettingsTab, self, 'autoSendSelection', 'Auto send selected data', box = "Data selection", callback = self.setAutoSendSelection, tooltip = "Send signals with selected data whenever the selection changes")
        self.graph.selectionChangedCallback = self.setAutoSendSelection

        OWGUI.rubber(self.GeneralTab)
        OWGUI.rubber(self.SettingsTab)
        self.icons = self.createAttributeIconDict()

        self.activateLoadedSettings()
        self.resize(700, 550)


    def activateLoadedSettings(self):
        dlg = self.createColorDialog()
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette("discPalette")
        self.graph.setCanvasBackground(dlg.getColor("Canvas"))

        apply([self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection, self.zoomSelectToolbar.actionPolygonSelection][self.toolbarSelection], [])

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
    def setData(self, data, onlyDataSubset = 0):
        if data and (len(data) == 0 or len(data.domain) == 0):
            data = None
        if self.data != None and data != None and self.data.checksum() == data.checksum():
            return    # check if the new data set is the same as the old one

        self.closeContext()
        sameDomain = self.data and data and data.domain.checksum() == self.data.domain.checksum() # preserve attribute choice if the domain is the same

        self.data = data
        if not sameDomain:
            self.initAttrValues()

        self.openContext("", data)


    # set an example table with a data subset subset of the data. if called by a visual classifier, the update parameter will be 0
    def setSubsetData(self, data):
        if self.subsetData != None and data != None and self.subsetData.checksum() == data.checksum():
            return    # check if the new data set is the same as the old one

        try:
            subsetData = data.select(self.data.domain)
            self.warning(10)
        except:
            subsetData = None
            self.warning(10, data and "'Examples' and 'Example Subset' data do not have compatible domains. Unable to draw 'Example Subset' data." or "")

        self.subsetData = subsetData


    # this is called by OWBaseWidget after setData and setSubsetData are called. this way the graph is updated only once
    def handleNewSignals(self):
        self.graph.clear()
        self.graph.setData(self.data, self.subsetData)
        self.majorUpdateGraph()
        self.sendSelections()


    # receive information about which attributes we want to show on x and y axis
    def setShownAttributes(self, list):
        if not self.data or not list: return
        self.graph.shownAttributeIndices = [i for i in range(len(self.data.domain)) if self.data.domain[i].name in list]
        self.majorUpdateGraph()


    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        pass
#        (selected, unselected) = self.graph.getSelectionsAsExampleTables([self.attrX, self.attrY])
#        self.send("Selected Data",selected)
#        self.send("Other Data",unselected)


    # ##############################################################################################################################################################
    # ATTRIBUTE SELECTION
    # ##############################################################################################################################################################

    def initAttrValues(self):
        self.graph.attributes = []
        self.graph.shownAttributeIndices = []
        self.timeCombo.clear()
        self.colorCombo.clear()
        self.graph.timeAttr = None
        self.graph.colorAttr = None
        self.colorCombo.addItem("(Same color)")

        if self.data == None: return

        domain = self.data.domain
        for attr in domain:
            self.timeCombo.addItem(self.icons[attr.varType], attr.name)
            self.colorCombo.addItem(self.icons[attr.varType], attr.name)
        self.graph.attributes = [(domain[a].name, domain[a].varType) for a in domain]
        self.graph.shownAttributeIndices = range(min(10, len(self.graph.attributes)))
        self.graph.timeAttr = domain[0].name
        self.graph.colorAttr = "(Same color)"


    def majorUpdateGraph(self):
        self.graph.zoomStack = []
        self.graph.removeAllSelections()
        self.updateGraph(setScale = 1)

    def updateGraph(self, **args):
        if not self.data:
            return

        self.graph.updateData(**args)


    # ##############################################################################################################################################################
    # SCATTERPLOT SETTINGS
    # ##############################################################################################################################################################

    def pointSizeChange(self):
        for curve in self.graph.itemList():
            if type(curve) == QwtPlotCurve:
                curve.symbol().setSize(self.graph.pointWidth)
        self.graph.replot()

    def alphaChange(self):
        for curve in self.graph.itemList():
            if type(curve) == QwtPlotCurve:
                brushColor = curve.symbol().brush().color()
                penColor = curve.symbol().pen().color()
                penColor2 = curve.pen().color()
                brushColor.setAlpha(self.graph.alphaValue)
                penColor.setAlpha(self.graph.alphaValue)
                penColor2.setAlpha(self.graph.alphaValue)
                curve.symbol().setBrush(QBrush(brushColor))
                curve.symbol().setPen(QPen(penColor))
                curve.setPen(QPen(penColor2))
        self.graph.replot()


    def antialiasingChange(self):
        for curve in self.graph.itemList():
            if type(curve) == QwtPlotCurve:
                curve.setRenderHint(QwtPlotItem.RenderAntialiased, self.graph.useAntialiasing)
        self.graph.replot()

    def setAutoSendSelection(self):
        if self.autoSendSelection:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(0)
            self.sendSelections()
        else:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(1)

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.graph.contPalette = dlg.getContinuousPalette("contPalette")
            self.graph.discPalette = dlg.getDiscretePalette("discPalette")
            self.graph.setCanvasBackground(dlg.getColor("Canvas"))
            self.graph.setGridColor(QPen(dlg.getColor("Grid")))
            self.updateGraph()

    def createColorDialog(self):
        c = OWColorPalette.ColorPaletteDlg(self, "Color palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        c.createContinuousPalette("contPalette", "Continuous Palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, "Canvas", "Canvas color", Qt.white)
        box.layout().addSpacing(5)
        c.createColorButton(box, "Grid", "Grid color", Qt.black)
        box.layout().addSpacing(5)
        #box.adjustSize()
        c.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return c

    def closeEvent(self, ce):
        OWWidget.closeEvent(self, ce)

    def sendReport(self):
        self.reportSettings("",
                            [("Time attribute", self.graph.timeAttr),
                             self.graph.colorAttr != "(Same color)" and ("Color", self.graph.colorAttr)])
        self.reportRaw("<br/>")
        self.reportImage(self.graph.saveToFileDirect, QSize(400, 400))

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlot()
    ow.show()
    ow.setData(orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\wine.tab"))
    #ow.setData(orange.ExampleTable("..\\..\\doc\\datasets\\wine.tab"))
    ow.handleNewSignals()
    a.exec_()
    #save settings
    ow.saveSettings()
