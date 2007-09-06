"""
<name>Parallel coordinates</name>
<description>Parallel coordinates (multiattribute) visualization.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/ParallelCoordinates.png</icon>
<priority>3200</priority>
"""
# ParallelCoordinates.py
#
# Show data using parallel coordinates visualization method
#
import orngOrangeFoldersQt4
from OWVisWidget import *
from OWParallelGraph import *
import OWToolbars, OWGUI, OWDlgs, orngVisFuncts
from sys import getrecursionlimit, setrecursionlimit

###########################################################################################
##### WIDGET : Parallel coordinates visualization
###########################################################################################
class OWParallelCoordinates(OWVisWidget):
    settingsList = ["graph.jitterSize", "graph.showDistributions",
                    "graph.showAttrValues", "graph.globalValueScaling", "linesDistance",
                    "graph.useSplines", "graph.alphaValue", "graph.alphaValueSubset", "graph.lineTracking", "graph.enabledLegend", "autoSendSelection",
                    "toolbarSelection", "graph.showStatistics", "colorSettings", "selectedSchemaIndex", "showAllAttributes"]
    jitterSizeNums = [0, 2,  5,  10, 15, 20, 30]
    linesDistanceNums = [10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Parallel Coordinates", TRUE)

        #add a graph widget
        self.graph = OWParallelGraph(self, self.mainArea)
        self.mainArea.layout().addWidget(self.graph)
        self.slider = QSlider(Qt.Horizontal, self.mainArea)
        self.mainArea.layout().addWidget(self.slider)
        self.sliderRange = 0
        self.slider.setRange(0, 0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.isResizing = 0
        self.showAllAttributes = 0

        self.inputs = [("Examples", ExampleTable, self.setData, Default), ("Example Subset", ExampleTable, self.setSubsetData), ("Attribute Selection List", AttributeList, self.setShownAttributes)]
        self.outputs = [("Selected Examples", ExampleTable), ("Unselected Examples", ExampleTable), ("Attribute Selection List", AttributeList)]

        #set default settings
        self.data = None
        self.subsetData = None
        self.linesDistance = 60
        self.autoSendSelection = 1
        self.attrDiscOrder = "None"
        self.attrContOrder = "None"
        self.projections = None
        self.correlationDict = {}
        self.middleLabels = "Correlations"
        self.attributeSelectionList = None
        self.toolbarSelection = 0
        self.colorSettings = None
        self.selectedSchemaIndex = 0

        self.graph.jitterSize = 10
        self.graph.showDistributions = 1
        self.graph.showStatistics = 0
        self.graph.showAttrValues = 1
        self.graph.globalValueScaling = 0
        self.graph.useSplines = 0
        self.graph.lineTracking = 0
        self.graph.enabledLegend = 1
        self.setSliderIndex = -1

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.GeneralTab = OWGUI.createTabPage(self.tabs, "Main")
        self.SettingsTab = OWGUI.createTabPage(self.tabs, "Settings")

        self.createShowHiddenLists(self.GeneralTab, callback = self.updateGraph)
        self.connect(self.shownAttribsLB, SIGNAL('doubleClicked(QListBoxItem *)'), self.flipAttribute)

        self.optimizationDlg = ParallelOptimization(self, signalManager = self.signalManager)
        self.optimizationDlgButton = OWGUI.button(self.GeneralTab, self, "Optimization dialog", callback = self.optimizationDlg.reshow, debuggingEnabled = 0)

        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        #connect controls to appropriate functions
        self.connect(self.slider, SIGNAL("valueChanged(int)"), self.updateGraphSlider)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # ####################################
        # SETTINGS functionality
        boxY = OWGUI.widgetBox(self.SettingsTab, "Transparency")
        OWGUI.hSlider(boxY, self, 'graph.alphaValue', label = "Examples: ", minValue=0, maxValue=255, step=10, callback = self.updateGraph, tooltip = "Alpha value used for drawing example lines")
        OWGUI.hSlider(boxY, self, 'graph.alphaValue2', label = "Rest:     ", minValue=0, maxValue=255, step=10, callback = self.updateGraph, tooltip = "Alpha value used to draw statistics, example subsets, ...")

        boxX = OWGUI.widgetBox(self.SettingsTab, "Line Properties")
        OWGUI.comboBox(boxX, self, "graph.jitterSize", label = 'Jittering size (% of size):  ', orientation='horizontal', callback = self.setJitteringSize, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)
        OWGUI.comboBox(boxX, self, "linesDistance", label = 'Minimum axis distance:  ', orientation='horizontal', callback = self.updateGraph, items = self.linesDistanceNums, tooltip = "What is the minimum distance between two adjecent attribute axis", sendSelectedValue = 1, valueType = int)

        # visual settings
        box = OWGUI.widgetBox(self.SettingsTab, "Visual Settings")
        OWGUI.checkBox(box, self, 'graph.showAttrValues', 'Show attribute values', callback = self.updateValues)
        OWGUI.checkBox(box, self, 'graph.useAntiAliasing', 'Use antialiasing (slower)', callback = self.updateValues)
        OWGUI.checkBox(box, self, 'graph.useSplines', 'Show splines', callback = self.updateValues, tooltip  = "Show lines using splines")
        OWGUI.checkBox(box, self, 'graph.lineTracking', 'Line tracking', callback = self.updateValues, tooltip = "Show nearest example with a wider line. The rest of the lines \nwill be shown in lighter colors.")
        OWGUI.checkBox(box, self, 'graph.enabledLegend', 'Show legend', callback = self.updateValues)
        OWGUI.checkBox(box, self, 'graph.globalValueScaling', 'Global Value Scaling', callback = self.setGlobalValueScaling)

        box3 = OWGUI.widgetBox(self.SettingsTab, "Statistics")
        OWGUI.comboBox(box3, self, "graph.showStatistics", items = ["No statistics", "Means, deviations", "Median, quartiles"], callback = self.updateValues, sendSelectedValue = 0, valueType = int)
        OWGUI.checkBox(box3, self, 'graph.showDistributions', 'Show distributions', callback = self.updateValues, tooltip = "Show bars with distribution of class values (only for discrete attributes)")

        OWGUI.comboBox(self.SettingsTab, self, "middleLabels", box = "Middle Labels", items = ["Off", "Correlations", "VizRank"], callback = self.updateGraph, tooltip = "What information do you wish to view on top in the middle of coordinate axes?", sendSelectedValue = 1, valueType = str)

        hbox4 = OWGUI.widgetBox(self.SettingsTab, "Colors", orientation = "horizontal")
        OWGUI.button(hbox4, self, "Set Colors", self.setColors, tooltip = "Set the canvas background color and color palette for coloring continuous variables", debuggingEnabled = 0)

        box2 = OWGUI.widgetBox(self.SettingsTab, "Sending Selection")
        OWGUI.checkBox(box2, self, 'autoSendSelection', 'Auto send selected data', callback = self.selectionChanged, tooltip = "Send signals with selected data whenever the selection changes.")

        self.SettingsTab.layout().addStretch(100)

        self.graph.selectionChangedCallback = self.selectionChanged
        self.icons = self.createAttributeIconDict()

        # add a settings dialog and initialize its values
        self.activateLoadedSettings()
        self.resize(900, 700)


    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        dlg = self.createColorDialog()
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette()
        self.graph.setCanvasBackground(dlg.getColor("Canvas"))
        apply([self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection, self.zoomSelectToolbar.actionPolygonSelection][self.toolbarSelection], [])
        self.cbShowAllAttributes()

    def flipAttribute(self, item):
        if self.graph.flipAttribute(str(item.text())):
            self.updateGraph()
            self.information(0)
        else:
            self.information(0, "Didn't flip the attribute. To flip a continuous attribute uncheck 'Global value scaling' checkbox.")

    def updateGraph(self, *args):
        attrs = self.getShownAttributeList()
        maxAttrs = self.mainArea.width() / self.linesDistance
        if len(attrs) > maxAttrs:
            rest = len(attrs) - maxAttrs
            if self.sliderRange != rest:
                self.slider.setRange(0, rest)
                self.sliderRange = rest
            elif self.isResizing:
                self.isResizing = 0
                return  # if we resized widget and it doesn't change the number of attributes that are shown then we return
            start = min(self.slider.value(), len(attrs)-maxAttrs)
            if self.setSliderIndex != -1:
                if self.setSliderIndex == 0: start = 0
                else:                        start = min(len(attrs)-maxAttrs, self.setSliderIndex - (maxAttrs+1)/2)
                start = max(start, 0)
                self.setSliderIndex = -1
                self.slider.setValue(start)
        else:
            self.slider.setRange(0,0)
            self.sliderRange = 0
            maxAttrs = len(attrs)
            start = 0

        #self.graph.updateData(attrs[start:start+maxAttrs], self.buildMidLabels(attrs[start:start+maxAttrs]))
        self.graph.updateData(attrs, self.buildMidLabels(attrs), start, start + maxAttrs)


    def updateGraphSlider(self, *args):
        attrs = self.getShownAttributeList()
        maxAttrs = self.mainArea.width() / self.linesDistance
        start = min(self.slider.value(), len(attrs)-maxAttrs)
        self.graph.setAxisScale(QwtPlot.xBottom, start, start + maxAttrs - 1, 1)
        self.graph.replot()

    # build a list of strings that will be shown in the middle of the parallel axis
    def buildMidLabels(self, attrs):
        labels = []
        if self.middleLabels == "Off" or self.data == None or len(self.data) == 0: return None
        elif self.middleLabels == "Correlations":
            for i in range(len(attrs)-1):
                corr = None
                if (attrs[i], attrs[i+1]) in self.correlationDict.keys():   corr = self.correlationDict[(attrs[i], attrs[i+1])]
                elif (attrs[i+1], attrs[i]) in self.correlationDict.keys(): corr = self.correlationDict[(attrs[i+1], attrs[i])]
                else:
                    corr = orngVisFuncts.computeCorrelation(self.data, attrs[i], attrs[i+1])
                    self.correlationDict[(attrs[i], attrs[i+1])] = corr
                if corr and len(self.graph.attributeFlipInfo.keys()) > 0 and (self.graph.attributeFlipInfo[attrs[i]] != self.graph.attributeFlipInfo[attrs[i+1]]): corr = -corr
                if corr: labels.append("%2.3f" % (corr))
                else: labels.append("")
        elif self.middleLabels == "VizRank":
            for i in range(len(attrs)-1):
                val = self.optimizationDlg.getVizRankVal(attrs[i], attrs[i+1])
                if val: labels.append("%2.2f%%" % (val))
                else: labels.append("")
        return labels


    # #############################################

    # had to override standart show to call updateGraph. otherwise self.mainArea.width() gives incorrect value
    def show(self):
        OWWidget.show(self)
        self.updateGraph()

    # ------------- SIGNALS --------------------------
    # receive new data and update all fields
    def setData(self, data):
        if data:
            name = getattr(data, "name", "")
            data = data.filterref(orange.Filter_hasClassValue())
            data.name = name
            if len(data) == 0 or len(data.domain) == 0:        # if we don't have any examples or attributes then this is not a valid data set
                data = None
        if self.data != None and data != None and self.data.checksum() == data.checksum():
            return    # check if the new data set is the same as the old one

        self.projections = None
        self.correlationDict = {}

        sameDomain = self.data and data and data.domain.checksum() == self.data.domain.checksum() # preserve attribute choice if the domain is the same
        self.data = data
        self.optimizationDlg.setData(self.data)

        # preserve attribute choice if the domain is the same
        if not sameDomain:
            self.setShownAttributeList(self.data, self.attributeSelectionList)
        self.resetAttrManipulation()    # update up down buttons

    def setSubsetData(self, data):
        self.warning(10)
        if self.subsetData != None and data != None and self.subsetData.checksum() == data.checksum():
            return    # check if the new data set is the same as the old one

        try:
            if data:
                subsetData = data.select(self.data.domain)
            else:
                subsetData = None
        except:
            subsetData = None
            self.warning(10, "'Examples' and 'Example Subset' data do not have copatible domains. Unable to draw 'Example Subset' data.")

        self.subsetData = subsetData


    # attribute selection signal - info about which attributes to show
    def setShownAttributes(self, attributeSelectionList):
        self.attributeSelectionList = attributeSelectionList
        if self.data and self.attributeSelectionList:
            for attr in self.attributeSelectionList:
                if not self.graph.attributeNameIndex.has_key(attr):  # this attribute list belongs to a new dataset that has not come yet
                    return

            self.setShownAttributeList(self.data, self.attributeSelectionList)
            self.attributeSelectionList = None

    # this is called by OWBaseWidget after setData and setSubsetData are called. this way the graph is updated only once
    def handleNewSignals(self):
        self.graph.clear()
        self.graph.setData(self.data, self.subsetData)
        self.updateGraph()
        self.selectionChanged()
        self.sendSelections()

    # ------------------------------------------------
    def sendShownAttributes(self, attrList = None):
        if not attrList:
            attrList = self.getShownAttributeList()
        self.send("Attribute Selection List", attrList)

    def selectionChanged(self):
        self.zoomSelectToolbar.buttonSendSelections.setEnabled(not self.autoSendSelection)
        if self.autoSendSelection:
            self.sendSelections()

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        return                    # TO DO: remove this return so that we will send selected items
        if not self.data:
            self.send("Selected Examples", None)
            self.send("Unselected Examples", None)
            return

        (selected, unselected) = self.graph.getSelectionsAsExampleTables()

        self.send("Selected Examples", selected)
        self.send("Unselected Examples", unselected)

    # ------------------------------------------------

    def updateValues(self):
        self.isResizing = 0
        self.updateGraph()

    def resizeEvent(self, e):
        OWWidget.resizeEvent(self,e)
        self.isResizing = 1
        # self.updateGraph()  # had to comment, otherwise python throws an exception

    # jittering options
    def setJitteringSize(self):
        self.isResizing = 0
        self.graph.setData(self.data)
        self.updateGraph()

    def setGlobalValueScaling(self):
        self.isResizing = 0
        self.graph.setData(self.data, self.subsetData)
        self.updateGraph()

    # update attribute ordering
    def updateShownAttributeList(self):
        self.isResizing = 0
        self.setShownAttributeList(self.data)
        self.updateGraph()

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.graph.contPalette = dlg.getContinuousPalette("contPalette")
            self.graph.discPalette = dlg.getDiscretePalette()
            self.graph.setCanvasBackground(dlg.getColor("Canvas"))
            self.updateGraph()

    def createColorDialog(self):
        c = OWDlgs.ColorPalette(self, "Color Palette")
        c.createDiscretePalette("Discrete Palette")
        c.createContinuousPalette("contPalette", "Continuous palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, "Canvas", "Canvas color", QColor(QColor(Qt.white)))
        box.layout().addSpacing(5)
        c.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return c

    def saveSettings(self):
        OWWidget.saveSettings(self)
        self.optimizationDlg.saveSettings()

    def closeEvent(self, ce):
        self.optimizationDlg.hide()
        OWWidget.closeEvent(self, ce)



CORRELATION = 0
VIZRANK = 1
#
# Find attribute subsets that are interesting to visualize using parallel coordinates
class ParallelOptimization(OWWidget):
    resultListList = [50, 100, 200, 500, 1000]
    qualityMeasure =  ["Classification accuracy", "Average correct", "Brier score"]
    testingMethod = ["Leave one out", "10-fold cross validation", "Test on learning set"]

    settingsList = ["attributeCount", "fileBuffer", "lastSaveDirName", "optimizationMeasure",
                    "numberOfAttributes", "orderAllAttributes", "optimizationMeasure"]

    def __init__(self, parallelWidget, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Parallel Optimization Dialog", FALSE)
        self.setCaption("Parallel Optimization Dialog")
        self.parallelWidget = parallelWidget

        self.optimizationMeasure = 0
        self.attributeCount = 5
        self.numberOfAttributes = 6
        self.fileName = ""
        self.lastSaveDirName = os.getcwd() + "/"
        self.fileBuffer = []
        self.projections = []
        self.allResults = []
        self.canOptimize = 0
        self.orderAllAttributes = 1 # do we wish to order all attributes or find just an interesting subset
        self.worstVal = -1  # used in heuristics to stop the search in uninteresting parts of the graph
        self.datasetName = ""

        self.loadSettings()

        self.measureBox = OWGUI.radioButtonsInBox(self.controlArea, self, "optimizationMeasure", ["Correlation", "VizRank"], box = "Select optimization measure", callback = self.updateGUI)
        self.vizrankSettingsBox = OWGUI.widgetBox(self.controlArea, "VizRank settings")
        self.optimizeBox = OWGUI.widgetBox(self.controlArea, "Optimize")
        self.manageBox = OWGUI.widgetBox(self.controlArea, "Manage results")
        self.resultsBox = OWGUI.widgetBox(self.mainArea, "Results")

        self.resultList = OWGUI.listBox(self.resultsBox, self)
        self.resultList.setMinimumSize(200,200)
        self.connect(self.resultList, SIGNAL("selectionChanged()"), self.showSelectedAttributes)

        # remove non-existing files
        names = []
        for i in range(len(self.fileBuffer)-1, -1, -1):
            (short, longName) = self.fileBuffer[i]
            if not os.path.exists(longName):
                self.fileBuffer.remove((short, longName))
            else: names.append(short)
        names.append("(None)")
        self.fileName = "(None)"

        self.hbox1 = OWGUI.widgetBox(self.vizrankSettingsBox, "VizRank projections file", orientation = "horizontal")
        self.vizrankFileCombo = OWGUI.comboBox(self.hbox1, self, "fileName", items = names, tooltip = "File that contains information about interestingness of scatterplots \ngenerated by VizRank method in scatterplot widget", callback = self.changeProjectionFile, sendSelectedValue = 1, valueType = str)
        self.browseButton = OWGUI.button(self.hbox1, self, "...", callback = self.loadProjections)
        self.browseButton.setMaximumWidth(20)

        self.resultsInfoBox = OWGUI.widgetBox(self.vizrankSettingsBox, "VizRank parameters")
        self.kNeighborsLabel = OWGUI.widgetLabel(self.resultsInfoBox, "Number of neighbors (k):")
        self.percentDataUsedLabel = OWGUI.widgetLabel(self.resultsInfoBox, "Percent of data used:")
        self.testingMethodLabel = OWGUI.widgetLabel(self.resultsInfoBox, "Testing method used:")
        self.qualityMeasureLabel = OWGUI.widgetLabel(self.resultsInfoBox, "Quality measure used:")

        #self.numberOfAttributesCombo = OWGUI.comboBoxWithCaption(self.optimizeBox, self, "numberOfAttributes", "Number of visualized attributes: ", tooltip = "Projections with this number of attributes will be evaluated", items = [x for x in range(3, 12)], sendSelectedValue = 1, valueType = int)
        self.allAttributesRadio = QRadioButton("Order all attributes", self.optimizeBox)
        self.optimizeBox.layout().addWidget(self.allAttributesRadio)
        self.connect(self.allAttributesRadio, SIGNAL("clicked()"), self.setAllAttributeRadio)
        box = OWGUI.widgetBox(self.optimizeBox, orientation = "horizontal")
        self.subsetAttributeRadio = QRadioButton("find subsets of      ", box)
        self.optimizeBox.layout().addWidget(self.subsetAttributeRadio)
        self.connect(self.subsetAttributeRadio, SIGNAL("clicked()"), self.setSubsetAttributeRadio)
        self.subsetAttributeEdit = OWGUI.lineEdit(box, self, "numberOfAttributes", valueType = int)
        label  = OWGUI.widgetLabel(box, "   attributes")

        self.startOptimizationButton = OWGUI.button(self.optimizeBox, self, "Start optimization", callback = self.startOptimization)
        f = self.startOptimizationButton.font()
        f.setBold(1)
        self.startOptimizationButton.setFont(f)
        self.stopOptimizationButton = OWGUI.button(self.optimizeBox, self, "Stop evaluation", callback = self.stopOptimizationClick)
        self.stopOptimizationButton.setFont(f)
        self.stopOptimizationButton.hide()
        self.connect(self.stopOptimizationButton , SIGNAL("clicked()"), self.stopOptimizationClick)

        self.clearButton = OWGUI.button(self.manageBox, self, "Clear results", self.clearResults)
        self.loadButton  = OWGUI.button(self.manageBox, self, "Load", self.loadResults)
        self.saveButton  = OWGUI.button(self.manageBox, self, "Save", self.saveResults)
        self.closeButton = OWGUI.button(self.manageBox, self, "Close dialog", self.hide)

        self.changeProjectionFile()
        self.updateGUI()
        self.activateLoadedSettings()

    def activateLoadedSettings(self):
        if self.orderAllAttributes: self.setAllAttributeRadio()
        else:                       self.setSubsetAttributeRadio()

    def updateGUI(self):
        self.vizrankSettingsBox.setEnabled(self.optimizationMeasure)

    # if user clicks new attribute list in optimization dialog, we update shown attributes
    def showSelectedAttributes(self):
        attrList = self.getSelectedAttributes()
        if not attrList: return

        self.parallelWidget.setShownAttributeList(self.parallelWidget.data, attrList)
        self.parallelWidget.graph.removeAllSelections()

        self.parallelWidget.middleLabels = (self.optimizationMeasure == VIZRANK and "VizRank") or "Correlations"
        self.parallelWidget.updateGraph()

    def setAllAttributeRadio(self):
        self.orderAllAttributes = 1
        self.allAttributesRadio.setChecked(1)
        self.subsetAttributeRadio.setChecked(0)
        self.subsetAttributeEdit.setEnabled(0)

    def setSubsetAttributeRadio(self):
        self.orderAllAttributes = 0
        self.allAttributesRadio.setChecked(0)
        self.subsetAttributeRadio.setChecked(1)
        self.subsetAttributeEdit.setEnabled(1)

    # return list of selected attributes
    def getSelectedAttributes(self):
        if self.resultList.count() == 0: return None
        return self.allResults[self.resultList.currentItem()][1]


    def setData(self, data):
        if hasattr(data, "name"):
            self.datasetName = data.name
        else: self.datasetName = ""

    # called when optimization is in progress
    def canContinueOptimization(self):
        return self.canOptimize

    def getWorstVal(self):
        return self.worstVal

    def stopOptimizationClick(self):
        self.canOptimize = 0

    # get vizrank value for attributes attr1 and attr2
    def getVizRankVal(self, attr1, attr2):
        if not self.projections: return None
        for (val, [a1, a2]) in self.projections:
            if (attr1 == a1 and attr2 == a2) or (attr1 == a2 and attr2 == a1): return val
        return None

    def changeProjectionFile(self):
        for (short, long) in self.fileBuffer:
            if short == self.fileName:
                self.loadProjections(long)
                return

    # load projections from a file
    def loadProjections(self, name = None):
        self.projections = []
        self.kNeighborsLabel.setText("Number of neighbors (k): " )
        self.percentDataUsedLabel.setText("Percent of data used:" )
        self.testingMethodLabel.setText("Testing method used:" )
        self.qualityMeasureLabel.setText("Quality measure used:" )

        if name == None:
            name = str(QFileDialog.getOpenFileName(self, "Open Projections",  self.lastSaveDirName, "Interesting projections (*.proj)"))
            if name == "": return

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        file = open(name, "rt")
        settings = eval(file.readline()[:-1])
        if settings.has_key("parentName") and settings["parentName"].lower() != "scatterplot":
            QMessageBox.critical( None, "Optimization Dialog", 'Unable to load projection file. Only projection file generated by scatterplot is compatible. \nThis file was created using %s method'%(settings["parentName"]), QMessageBox.Ok)
            file.close()
            return

        if type(eval(file.readline()[:-1])) != list:    # second line must contain a list of classes that we tried to separate
            QMessageBox.critical(None,'Old version of projection file','This file was saved with an older version of k-NN Optimization Dialog. The new version of dialog offers \nsome additional functionality and therefore you have to compute the projection quality again.',QMessageBox.Ok)
            file.close()
            return

        try:
            line = file.readline()[:-1]; ind = 0    # first line is a settings line
            (acc, other_results, lenTable, attrList, tryIndex, strList) = eval(line)
            if len(attrList) != 2:
                QMessageBox.information(self, "Incorrect file", "File should contain projections with 2 attributes!", QMessageBox.Ok)
                file.close()
                return

            while (line != ""):
                (acc, other_results, lenTable, attrList, tryIndex, strList) = eval(line)
                self.projections += [(acc, attrList)]
                line = file.readline()[:-1]
        except:
            self.projections = []
            file.close()
            QMessageBox.information(self, "Incorrect file", "Incorrect file format!", QMessageBox.Ok)
            return

        file.close()

        if (shortFileName, name) in self.fileBuffer:
            self.fileBuffer.remove((shortFileName, name))

        self.fileBuffer.insert(0, (shortFileName, name))


        if len(self.fileBuffer) > 10:
            self.fileBuffer.remove(self.fileBuffer[-1])

        self.vizrankFileCombo.clear()
        for i in range(len(self.fileBuffer)):
            self.vizrankFileCombo.addItem(self.fileBuffer[i][0])
        self.fileName = shortFileName

        self.kNeighborsLabel.setText("Number of neighbors (k): %s" % (str(settings["kValue"])))
        self.percentDataUsedLabel.setText("Percent of data used: %d %%" % (settings["percentDataUsed"]))
        self.testingMethodLabel.setText("Testing method used: %s" % (self.testingMethod[settings["testingMethod"]]))
        self.qualityMeasureLabel.setText("Quality measure used: %s" % (self.qualityMeasure[settings["qualityMeasure"]]))


    def addProjection(self, val, attrList):
        index = self.findTargetIndex(val, max)
        self.allResults.insert(index, (val, attrList))
        self.resultList.addItem("%.3f - %s" % (val, str(attrList)), index)


    def findTargetIndex(self, accuracy, funct):
        # use bisection to find correct index
        top = 0; bottom = len(self.allResults)

        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if funct(accuracy, self.allResults[mid][0]) == accuracy: bottom = mid
            else: top = mid

        if len(self.allResults) == 0: return 0
        if funct(accuracy, self.allResults[top][0]) == accuracy:
            return top
        else:
            return bottom


    def startOptimization(self):
        self.clearResults()
        if self.parallelWidget.data == None: return

        if self.optimizationMeasure == VIZRANK and self.fileName == "":
            QMessageBox.information(self, "No projection file", "If you wish to optimize using VizRank you first have to load a projection file \ncreated by VizRank using Scatterplot widget.", QMessageBox.Ok)
            return
        if self.parallelWidget.data == None:
            QMessageBox.information(self, "Missing data set", "A data set has to be loaded in order to perform optimization.", QMessageBox.Ok)
            return

        attrInfo = []
        if self.optimizationMeasure == CORRELATION:
            attrList = [attr.name for attr in self.parallelWidget.data.domain.attributes]
            attrInfo = orngVisFuncts.computeCorrelationBetweenAttributes(self.parallelWidget.data, attrList)
            #attrInfo = orngVisFuncts.computeCorrelationInsideClassesBetweenAttributes(self.parallelWidget.data, attrList)
        elif self.optimizationMeasure == VIZRANK:
            for (val, [a1, a2]) in self.projections:
                attrInfo.append((val, a1, a2))

            # check if all attributes in loaded projection file are actually present in this data set
            attrs = [attr.name for attr in self.parallelWidget.data.domain.attributes]
            for (v, a1, a2) in attrInfo:
                if a1 not in attrs:
                    print "attribute " + a1 + " was not found in the data set. You probably loaded wrong file with VizRank projections."
                    return
                if a2 not in attrs:
                    print "attribute " + a2 + " was not found in the data set. You probably loaded wrong file with VizRank projections."
                    return

        if len(attrInfo) == 0:
            print "len(attrInfo) == 0. No attribute pairs. Unable to optimize."; return

        self.worstVal = -1
        self.canOptimize = 1
        self.startOptimizationButton.hide()
        self.stopOptimizationButton.show()
        #qApp.processEvents()        # allow processing of other events

        if self.orderAllAttributes:
            orngVisFuncts.optimizeAttributeOrder(attrInfo, len(self.parallelWidget.data.domain.attributes), self, qApp)
        else:
            orngVisFuncts.optimizeAttributeOrder(attrInfo, self.numberOfAttributes, self, qApp)

        self.stopOptimizationButton.hide()
        self.startOptimizationButton.show()


    # ################################
    # MANAGE RESULTS
    def updateShownProjections(self, *args):
        self.resultList.clear()
        for i in range(len(self.allResults)):
            self.resultList.addItem("%.2f - %s" % (self.allResults[i][0], str(self.allResults[i][1])), i)
        if self.resultList.count() > 0: self.resultList.setCurrentRow(0)

    def clearResults(self):
        self.allResults = []
        self.resultList.clear()


    def saveResults(self, filename = None):
        if filename == None:
            filename = ""
            if self.datasetName != "":
                filename = os.path.splitext(os.path.split(self.datasetName)[1])[0]
            if self.optimizationMeasure == CORRELATION: filename += " - " + "correlation"
            else:                                       filename += " - " + "vizrank"

            name = str(QFileDialog.getSaveFileName(self, "Save Parallel Projections",  os.path.join(self.lastSaveDirName, filename), "Parallel projections (*.papr)"))
            if name == "": return
        else:
            name = filename

        # take care of extension
        if os.path.splitext(name)[1] != ".papr": name += ".papr"

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        # open, write and save file
        file = open(name, "wt")
        for val in self.allResults:
            file.write(str(val) + "\n")
        file.close()

    def loadResults(self):
        self.clearResults()

        name = str(QFileDialog.getOpenFileName(self, "Open Parallel Projections",  self.lastSaveDirName, "Parallel projections (*.papr)"))
        if name == "": return

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        file = open(name, "rt")
        line = file.readline()[:-1]; ind = 0
        while (line != ""):
            (val, attrList) = eval(line)
            self.allResults.insert(ind, (val, attrList))
            self.resultList.addItem("%.2f - %s" % (val, str(attrList)), ind)
            line = file.readline()[:-1]
            ind+=1
        file.close()

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWParallelCoordinates()
    ow.show()
    data = orange.ExampleTable(r"e:\Development\Orange Datasets\UCI\wine.tab")
    ow.setData(data)
    ow.handleNewSignals()
    a.exec_()