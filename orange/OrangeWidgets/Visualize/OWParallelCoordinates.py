"""
<name>Parallel coordinates</name>
<description>Shows data using parallel coordianates visualization method</description>
<category>Visualization</category>
<icon>icons/ParallelCoordinates.png</icon>
<priority>3200</priority>
"""
# ParallelCoordinates.py
#
# Show data using parallel coordinates visualization method
# 

from OWWidget import *
from OWParallelGraph import *
import OWVisAttrSelection 
import OWToolbars
import OWGUI
from sys import getrecursionlimit, setrecursionlimit

###########################################################################################
##### WIDGET : Parallel coordinates visualization
###########################################################################################
class OWParallelCoordinates(OWWidget):
    settingsList = ["attrContOrder", "attrDiscOrder", "graphCanvasColor", "jitterSize", "showDistributions", "showAttrValues", "hidePureExamples", "globalValueScaling", "linesDistance", "useSplines", "lineTracking", "showLegend", "autoSendSelection", "sendShownAttributes"]
    attributeContOrder = ["None","ReliefF", "Fisher discriminant"]
    attributeDiscOrder = ["None","ReliefF","GainRatio", "Oblivious decision graphs"]
    jitterSizeNums = [0, 2,  5,  10, 15, 20, 30]
    linesDistanceNums = [20, 30, 40, 50, 60, 70, 80, 100, 120, 150]

    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Parallel Coordinates", "Show data using parallel coordinates visualization method", FALSE, TRUE, icon = "ParallelCoordinates.png")
        self.resize(700,700)

        self.inputs = [("Examples", ExampleTable, self.data, 1), ("Selection", list, self.selection, 1)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass), ("Attribute selection", list)]
    
        #set default settings
        self.data = None

        self.jitterSize = 10
        self.linesDistance = 60
        
        self.showDistributions = 1
        self.showAttrValues = 1
        self.hidePureExamples = 1
        
        self.globalValueScaling = 0
        self.useSplines = 0
        self.lineTracking = 0
        self.showLegend = 1
        self.autoSendSelection = 1
        self.sendShownAttributes = 0
        self.attrDiscOrder = "None"
        self.attrContOrder = "None"
        self.graphCanvasColor = str(Qt.white.name())
        self.projections = None
        self.correlationDict = {}
        self.middleLabels = "Correlations"

        self.setSliderIndex = -1

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")

        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWParallelGraph(self, self.mainArea)
        self.slider = QSlider(QSlider.Horizontal, self.mainArea)
        self.sliderRange = 0
        self.slider.setRange(0, 0)
        self.slider.setTickmarks(QSlider.Below)
        self.isResizing = 0 
        self.box.addWidget(self.graph)
        self.box.addWidget(self.slider)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        #add controls to self.controlArea widget
        self.targetGroup = QVGroupBox(self.GeneralTab)
        self.targetGroup.setTitle(" Target class value: ")
        self.targetValueCombo = QComboBox(self.targetGroup)
        self.connect(self.targetValueCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)
        
        self.shownAttribsGroup = QVGroupBox(self.GeneralTab)
        self.addRemoveGroup = QHButtonGroup(self.GeneralTab)
        self.hiddenAttribsGroup = QVGroupBox(self.GeneralTab)
        self.shownAttribsGroup.setTitle("Shown attributes")
        self.hiddenAttribsGroup.setTitle("Hidden attributes")

        self.shownAttribsLB = QListBox(self.shownAttribsGroup)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)
        
        self.hbox = QHBox(self.shownAttribsGroup)
        self.buttonUPAttr = QPushButton("Attr UP", self.hbox)
        self.buttonDOWNAttr = QPushButton("Attr DOWN", self.hbox)

        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        self.optimizationDlg = ParallelOptimization(self)
        self.connect(self.optimizationDlg.resultList, SIGNAL("selectionChanged()"), self.showSelectedAttributes)
        self.optimizationDlgButton = OWGUI.button(self.GeneralTab, self, "Optimization dialog", callback = self.optimizationDlg.reshow)

        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        #connect controls to appropriate functions
        self.connect(self.buttonUPAttr, SIGNAL("clicked()"), self.moveAttrUP)
        self.connect(self.buttonDOWNAttr, SIGNAL("clicked()"), self.moveAttrDOWN)

        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)

        self.connect(self.slider, SIGNAL("valueChanged(int)"), self.updateGraph)

        # ####################################
        # SETTINGS functionality
        # jittering options
        OWGUI.comboBoxWithCaption(self.SettingsTab, self, "jitterSize", 'Jittering size (% of size):  ', box = " Jittering options ", callback = self.setJitteringSize, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)

        # attribute axis distance
        OWGUI.comboBoxWithCaption(self.SettingsTab, self, "linesDistance", 'Minimum distance: ', box = " Attribute axis distance ", callback = self.updateGraph, items = self.linesDistanceNums, tooltip = "What is the minimum distance between two adjecent attribute axis", sendSelectedValue = 1, valueType = int)
        
        # ####
        # visual settings
        box = OWGUI.widgetBox(self.SettingsTab, " Visual settings ")
        OWGUI.checkBox(box, self, 'showDistributions', 'Show distributions', callback = self.updateValues, tooltip = "Show bars with distribution of class values")
        OWGUI.checkBox(box, self, 'showAttrValues', 'Show attribute values', callback = self.updateValues)
        OWGUI.checkBox(box, self, 'hidePureExamples', 'Hide pure examples', callback = self.updateValues, tooltip = "When one value of a discrete attribute has only examples from one class, \nstop drawing lines for this example. Figure must be interpreted from left to right.")
        OWGUI.checkBox(box, self, 'useSplines', 'Show splines', callback = self.updateValues, tooltip  = "Show lines using splines")
        OWGUI.checkBox(box, self, 'lineTracking', 'Line tracking', callback = self.updateValues, tooltip = "Show nearest example with a wider line. The rest of the lines \nwill be shown in lighter colors.")
        OWGUI.checkBox(box, self, 'showLegend', 'Show legend', callback = self.updateValues)
        OWGUI.checkBox(box, self, 'globalValueScaling', 'Global Value Scaling', callback = self.setGlobalValueScaling)
        
        
        box2 = OWGUI.widgetBox(self.SettingsTab, " Sending selection ")
        OWGUI.checkBox(box2, self, 'autoSendSelection', 'Auto send selected data', callback = self.setAutoSendSelection, tooltip = "Send signals with selected data whenever the selection changes.")
        OWGUI.checkBox(box2, self, 'sendShownAttributes', 'Send only shown attributes', callback = self.setAutoSendSelection, tooltip = "Send dataset with all attributes or just attributes that are currently shown.")

        OWGUI.comboBox(self.SettingsTab, self, "middleLabels", box = " Middle labels ", items = ["Off", "Correlations", "VizRank"], callback = self.updateGraph, tooltip = "What information do you wish to view on top in the middle of coordinate axes?", sendSelectedValue = 1, valueType = str)
        
        # continuous attribute ordering
        OWGUI.comboBox(self.SettingsTab, self, "attrContOrder", box = " Continuous attribute ordering ", items = self.attributeContOrder, callback = self.updateShownAttributeList, sendSelectedValue = 1, valueType = str)
        OWGUI.comboBox(self.SettingsTab, self, "attrDiscOrder", box = " Discrete attribute ordering ", items = self.attributeDiscOrder, callback = self.updateShownAttributeList, sendSelectedValue = 1, valueType = str)

        self.gSetCanvasColorB = QPushButton("Canvas Color", self.SettingsTab)
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)

        self.graph.autoSendSelectionCallback = self.setAutoSendSelection
        
        # add a settings dialog and initialize its values
        self.activateLoadedSettings()
        self.resize(900, 700)


    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.graph.updateSettings(enabledLegend = self.showLegend, useSplines = self.useSplines, lineTracking = self.lineTracking)
        self.graph.showDistributions = self.showDistributions
        self.graph.showAttrValues = self.showAttrValues
        self.graph.hidePureExamples = self.hidePureExamples
        self.graph.globalValueScaling = self.globalValueScaling
        self.graph.jitterSize = self.jitterSize
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        if not self.data:
            self.send("Selected Examples", None)
            self.send("Unselected Examples", None)
            self.send("Example Distribution", None)
            return
        
        (selected, unselected, merged) = self.graph.getSelectionsAsExampleTables()
        if not self.sendShownAttributes:
            self.send("Selected Examples", selected)
            self.send("Unselected Examples", unselected)
            self.send("Example Distribution", merged)
        else:
            attrs = self.getShownAttributeList()
            if self.data.domain.classVar: attrs += [self.data.domain.classVar.name]
            if selected:    self.send("Selected Examples", selected.select(attrs))
            else:           self.send("Selected Examples", None)
            if unselected:  self.send("Unselected Examples", unselected.select(attrs))
            else:           self.send("Unselected Examples", None)
            if merged:
                attrs += [merged.domain.classVar.name]
                self.send("Example Distribution", merged.select(attrs))
            else:           self.send("Example Distribution", None)

    def sendAttributeSelection(self, attrs):
        self.send("Attribute selection", attrs)


    # ####################
    # LIST BOX FUNCTIONS
    # ####################

    # move selected attribute in "Attribute Order" list one place up
    def moveAttrUP(self):
        for i in range(self.shownAttribsLB.count()):
            if self.shownAttribsLB.isSelected(i) and i != 0:
                text = self.shownAttribsLB.text(i)
                self.shownAttribsLB.removeItem(i)
                self.shownAttribsLB.insertItem(text, i-1)
                self.shownAttribsLB.setSelected(i-1, TRUE)
        self.updateGraph()

    # move selected attribute in "Attribute Order" list one place down  
    def moveAttrDOWN(self):
        count = self.shownAttribsLB.count()
        for i in range(count-2,-1,-1):
            if self.shownAttribsLB.isSelected(i):
                text = self.shownAttribsLB.text(i)
                self.shownAttribsLB.removeItem(i)
                self.shownAttribsLB.insertItem(text, i+1)
                self.shownAttribsLB.setSelected(i+1, TRUE)
        self.updateGraph()

    def addAttribute(self):
        count = self.hiddenAttribsLB.count()
        pos   = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.hiddenAttribsLB.isSelected(i):
                text = self.hiddenAttribsLB.text(i)
                self.hiddenAttribsLB.removeItem(i)
                self.shownAttribsLB.insertItem(text, pos)
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
        self.updateGraph()

    def removeAttribute(self):
        count = self.shownAttribsLB.count()
        pos   = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.shownAttribsLB.isSelected(i):
                text = self.shownAttribsLB.text(i)
                self.shownAttribsLB.removeItem(i)
                self.hiddenAttribsLB.insertItem(text, pos)
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
        self.updateGraph()

    # #####################

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

        targetVal = str(self.targetValueCombo.currentText())
        if targetVal == "(None)": targetVal = None
        self.graph.updateData(attrs[start:start+maxAttrs], targetVal, self.buildMidLabels(attrs[start:start+maxAttrs]))
        self.slider.repaint()
        self.graph.update()
        #self.graph.repaint()


    # build a list of strings that will be shown in the middle of the parallel axis
    def buildMidLabels(self, attrs):
        labels = []
        if self.middleLabels == "Off": return None
        elif self.middleLabels == "Correlations":
            for i in range(len(attrs)-1):
                corr = None
                if attrs[i] + "-" + attrs[i+1] in self.correlationDict.keys():   corr = self.correlationDict[attrs[i] + "-" + attrs[i+1]]
                elif attrs[i+1] + "-" + attrs[i] in self.correlationDict.keys(): corr = self.correlationDict[attrs[i+1] + "-" + attrs[i]]
                else:
                    corr = OWVisAttrSelection.computeCorrelation(self.data, attrs[i], attrs[i+1])
                    self.correlationDict[attrs[i] + "-" + attrs[i+1]] = corr
                if corr != None: labels.append("%2.3f" % (corr))
                else: labels.append("")
        elif self.middleLabels == "VizRank":
            for i in range(len(attrs)-1):
                val = self.optimizationDlg.getVizRankVal(attrs[i], attrs[i+1])
                if val: labels.append("%2.2f%%" % (val))
                else: labels.append("")
        return labels
                


    # ###### SHOWN ATTRIBUTE LIST ##############
    # set attribute list
    def setShownAttributeList(self, data):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if data == None: return
        
        shown, hidden, maxIndex = OWVisAttrSelection.selectAttributes(data, self.attrContOrder, self.attrDiscOrder, self.projections)
        self.setSliderIndex = maxIndex
        if data.domain.classVar and data.domain.classVar.name not in shown and data.domain.classVar.name not in hidden:
            self.shownAttribsLB.insertItem(data.domain.classVar.name)
        for attr in shown:
            self.shownAttribsLB.insertItem(attr)
        for attr in hidden:
            self.hiddenAttribsLB.insertItem(attr)
        
        
    def getShownAttributeList(self):
        list = []
        for i in range(self.shownAttribsLB.count()):
            list.append(str(self.shownAttribsLB.text(i)))
        return list

    # #############################
    # if user clicks new attribute list in optimization dialog, we update shown attributes
    def showSelectedAttributes(self):
        attrList = self.optimizationDlg.getSelectedAttributes()
        if not attrList: return

        attrs = [attr.name for attr in self.data.domain.attributes]
        for attr in attrList:
            if attr not in attrs:
                print "Attribute ", attr, " does not exist in the data set. unable to set attributes."
                return

        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        for attr in attrList:
            self.shownAttribsLB.insertItem(attr)
        for attr in self.data.domain.attributes:
            if attr.name not in attrList:
                self.hiddenAttribsLB.insertItem(attr.name)

        if self.optimizationDlg.optimizationMeasure == VIZRANK:
            self.middleLabels = "VizRank"
        else:
            self.middleLabels = "Correlations"
        self.updateGraph()
        if self.sendShownAttributes: self.sendSelections()  # if we send only shown attributes, we also have to send new dataset with possibly new attributes
    
    # #############################################

    # had to override standart show to call updateGraph. otherwise self.mainArea.width() gives incorrect value    
    def show(self):
        OWWidget.show(self)
        self.updateGraph()
    
    # ###### DATA ################################
    # receive new data and update all fields
    def data(self, data):
        self.projections = None
        self.correlationDict = {}
        
        exData = self.data
        self.data = None
        if data: self.data = orange.Preprocessor_dropMissingClasses(data)
        self.graph.setData(self.data)

        if not (data and exData and str(exData.domain.attributes) == str(data.domain.attributes)): # preserve attribute choice if the domain is the same
            self.shownAttribsLB.clear()
            self.hiddenAttribsLB.clear()

            self.targetValueCombo.clear()
            self.targetValueCombo.insertItem("(None)")

            # update target combo
            if self.data and self.data.domain.classVar and self.data.domain.classVar.varType == orange.VarTypes.Discrete:
                for val in self.data.domain.classVar.values:
                    self.targetValueCombo.insertItem(val)
                self.targetValueCombo.setCurrentItem(0)
            
            self.setShownAttributeList(self.data)

        self.updateGraph()
        self.sendSelections()
    #################################################

    
    # ###### SELECTION ################################
    # receive a list of attributes we wish to show
    def selection(self, list):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if self.data == None: return

        for attr in self.data.domain:
            if attr.name in list: self.shownAttribsLB.insertItem(attr.name)
            else:                 self.hiddenAttribsLB.insertItem(attr.name)
                
        self.updateGraph()
    #################################################

    def updateValues(self):
        self.isResizing = 0
        self.graph.updateSettings(showDistributions = self.showDistributions, useSplines = self.useSplines, enabledLegend = self.showLegend, lineTracking = self.lineTracking)
        self.graph.showAttrValues = self.showAttrValues
        self.graph.hidePureExamples = self.hidePureExamples
        self.updateGraph()

    def resizeEvent(self, e):
        self.isResizing = 1
        # self.updateGraph()  # had to comment, otherwise python throws an exception

    # jittering options
    def setJitteringSize(self):
        self.isResizing = 0
        self.graph.jitterSize = self.jitterSize
        self.graph.setData(self.data)
        self.updateGraph()

    def setGlobalValueScaling(self):
        self.isResizing = 0
        self.graph.globalValueScaling = self.globalValueScaling
        self.graph.setData(self.data)
        if self.globalValueScaling:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
        self.updateGraph()

    # update attribute ordering
    def updateShownAttributeList(self):
        self.isResizing = 0
        self.setShownAttributeList(self.data)
        self.updateGraph()

    def setAutoSendSelection(self):
        if self.autoSendSelection:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(0)
            self.sendSelections()
        else:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(1)
            

    def setGraphCanvasColor(self):
        newColor = QColorDialog.getColor(QColor(self.graphCanvasColor))
        if newColor.isValid():
            self.graphCanvasColor = str(newColor.name())
            self.graph.setCanvasColor(QColor(newColor))


CORRELATION = 0
VIZRANK = 1

class ParallelOptimization(OWBaseWidget):
    resultListList = [50, 100, 200, 500, 1000]
    settingsList = ["attributeCount", "fileBuffer", "lastSaveDirName", "optimizationMeasure", "numberOfAttributes"]
    qualityMeasure =  ["Classification accuracy", "Average correct", "Brier score"]
    testingMethod = ["Leave one out", "10-fold cross validation", "Test on learning set"]

    def __init__(self, parallelWidget, parent=None):
        OWBaseWidget.__init__(self, parent, "Parallel Optimization Dialog", "Find attribute subsets that are interesting to visualize using parallel coordinates", FALSE, FALSE, FALSE)

        self.setCaption("Qt Parallel Optimization Dialog")
        self.topLayout = QVBoxLayout( self, 10 ) 
        self.grid=QGridLayout(4,2)
        self.topLayout.addLayout( self.grid, 10 )
        self.parallelWidget = parallelWidget

        self.optimizationMeasure = 0
        self.attributeCount = 5
        self.numberOfAttributes = 6
        self.resultListLen = 1000
        self.fileName = ""
        self.lastSaveDirName = os.getcwd() + "/"
        self.fileBuffer = []
        self.projections = []
        self.allResults = []
        self.canOptimize = 0
        self.worstVal = -1  # used in heuristics to stop the search in uninteresting parts of the graph

        self.loadSettings()

        self.measureBox = OWGUI.radioButtonsInBox(self, self, "optimizationMeasure", ["Correlation", "VizRank"], box = " Select optimization measure ")
        self.vizrankSettingsBox = OWGUI.widgetBox(self, " VizRank settings ")
        self.optimizeBox = OWGUI.widgetBox(self, " Optimize ")
        self.manageBox = OWGUI.widgetBox(self, " Manage results ")
        self.resultsBox = OWGUI.widgetBox(self, " Results ")

        self.grid.addWidget(self.measureBox,0,0)
        self.grid.addWidget(self.vizrankSettingsBox,1,0)
        self.grid.addWidget(self.optimizeBox,2,0)
        self.grid.addWidget(self.manageBox,3,0)
        self.grid.addMultiCellWidget (self.resultsBox,0,3, 1, 1)
        self.grid.setColStretch(0, 0)
        self.grid.setColStretch(1, 100)
        self.grid.setRowStretch(0, 0)
        self.grid.setRowStretch(1, 0)
        self.grid.setRowStretch(2, 0)
        self.grid.setRowStretch(3, 100)
        self.vizrankSettingsBox.setMinimumWidth(200)

        self.resultList = QListBox(self.resultsBox)
        self.resultList.setMinimumSize(200,200)
              
        # remove non-existing files
        names = []
        for i in range(len(self.fileBuffer)-1, -1, -1):
            (short, longName) = self.fileBuffer[i]
            if not os.path.exists(longName):
                self.fileBuffer.remove((short, longName))
            else: names.append(short)
        if len(self.fileBuffer) > 0: self.fileName = self.fileBuffer[0][0]
                
        self.hbox1 = OWGUI.widgetBox(self.vizrankSettingsBox, " VizRank projections file ", orientation = "horizontal")
        self.vizrankFileCombo = OWGUI.comboBox(self.hbox1, self, "fileName", items = names, tooltip = "File that contains information about interestingness of scatterplots \ngenerated by VizRank method in scatterplot widget", callback = self.changeProjectionFile, sendSelectedValue = 1, valueType = str)
        self.browseButton = OWGUI.button(self.hbox1, self, "...", callback = self.loadProjections)
        self.browseButton.setMaximumWidth(20)
        
        
        self.resultsInfoBox = OWGUI.widgetBox(self.vizrankSettingsBox, " VizRank parameters ")
        self.kNeighborsLabel = OWGUI.widgetLabel(self.resultsInfoBox, "Number of neighbors (k):")
        self.percentDataUsedLabel = OWGUI.widgetLabel(self.resultsInfoBox, "Percent of data used:")
        self.testingMethodLabel = OWGUI.widgetLabel(self.resultsInfoBox, "Testing method used:")
        self.qualityMeasureLabel = OWGUI.widgetLabel(self.resultsInfoBox, "Quality measure used:")

        self.numberOfAttributesCombo = OWGUI.comboBoxWithCaption(self.optimizeBox, self, "numberOfAttributes", "Number of visualized attributes: ", tooltip = "Projections with this number of attributes will be evaluated", items = [x for x in range(3, 12)], sendSelectedValue = 1, valueType = int)
        self.startOptimizationButton = OWGUI.button(self.optimizeBox, self, " Start optimization ", callback = self.startOptimization)
        f = self.startOptimizationButton.font()
        f.setBold(1)
        self.startOptimizationButton.setFont(f)
        self.stopOptimizationButton = OWGUI.button(self.optimizeBox, self, "Stop evaluation", callback = self.stopOptimizationClick)
        self.stopOptimizationButton.setFont(f)
        self.stopOptimizationButton.hide()
        self.connect(self.stopOptimizationButton , SIGNAL("clicked()"), self.stopOptimizationClick)

        
        self.resultListCombo = OWGUI.comboBoxWithCaption(self.manageBox, self, "resultListLen", "Number of interesting projections: ", tooltip = "Maximum length of the list of interesting projections", items = self.resultListList, callback = self.updateShownProjections, sendSelectedValue = 1, valueType = int)
        self.clearButton = OWGUI.button(self.manageBox, self, "Clear results", self.clearResults)
        self.loadButton = OWGUI.button(self.manageBox, self, "Load", self.loadResults)
        self.saveButton = OWGUI.button(self.manageBox, self, "Save", self.saveResults)
        self.closeButton = OWGUI.button(self.manageBox, self, "Close dialog", self.hide)

        self.changeProjectionFile()


    # return list of selected attributes
    def getSelectedAttributes(self):
        if self.resultList.count() == 0: return None
        return self.allResults[self.resultList.currentItem()][1]
        
        
    # called when optimization is in progress
    def canContinueOptimization(self):
        return self.canOptimize

    def getWorstVal(self):
        return self.worstVal
        
    def stopOptimizationClick(self):
        self.canOptimize = 0

    def destroy(self, dw, dsw):
        self.saveSettings()

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
            name = str(QFileDialog.getOpenFileName( self.lastSaveDirName, "Interesting projections (*.proj)", self, "", "Open Projections"))
            if name == "": return

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        file = open(name, "rt")
        settings = eval(file.readline()[:-1])
        if settings.has_key("parentName") and settings["parentName"] != "ScatterPlot":
            QMessageBox.critical( None, "Optimization Dialog", 'Unable to load projection file. Only projection file generated by scatterplot is compatible. \nThis file was created using %s method'%(settings["parentName"]), QMessageBox.Ok)
            file.close()
            return
        
        line = file.readline()[:-1]; ind = 0

        try:
            (acc, lenTable, attrList, strList) = eval(line)
            if len(attrList) != 2:
                QMessageBox.information(self, "Incorrect file", "File should contain projections with 2 attributes!", QMessageBox.Ok)
                return
            
            while (line != ""):
                (acc, lenTable, attrList, strList) = eval(line)
                self.projections += [(acc, attrList)]
                line = file.readline()[:-1]
        except:
            self.projections = []
            file.close()
            QMessageBox.information(self, "Incorrect file", "Incorrect file format!", QMessageBox.Ok)
            return

        file.close()

        if (shortFileName, name) not in self.fileBuffer:
            self.fileBuffer.insert(0, (shortFileName, name))
            self.vizrankFileCombo.insertItem(shortFileName)
            if len(self.fileBuffer) > 10: self.fileBuffer.remove(self.fileBuffer[-1])

        self.kNeighborsLabel.setText("Number of neighbors (k): %s" % (str(settings["kValue"])))
        self.percentDataUsedLabel.setText("Percent of data used: %d %%" % (settings["percentDataUsed"]))
        self.testingMethodLabel.setText("Testing method used: %s" % (self.testingMethod[settings["testingMethod"]]))
        self.qualityMeasureLabel.setText("Quality measure used: %s" % (self.qualityMeasure[settings["qualityMeasure"]]))


    def addProjection(self, val, attrList):
        index = self.findTargetIndex(val, max)
        if index > self.resultListLen: return
        self.allResults.insert(index, (val, attrList))
        self.resultList.insertItem("%.3f - %s" % (val, str(attrList)), index)
        if len(self.allResults) > self.resultListLen:
            self.allResults.remove(self.allResults[-1])
            self.resultList.removeItem(self.resultList.count()-1)
            self.worstVal = self.allResults[-1][0]
        

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
        
        if self.optimizationMeasure == VIZRANK and self.fileName == "":
            QMessageBox.information(self, "No projection file", "If you wish to optimize using VizRank you first have to load a projection file \ncreated by VizRank using Scatterplot widget.", QMessageBox.Ok)
            return
        if self.parallelWidget.data == None:
            QMessageBox.information(self, "Missing data set", "A data set has to be loaded in order to perform optimization.", QMessageBox.Ok)
            return

        attrInfo = []
        if self.optimizationMeasure == CORRELATION:
            attrList = [attr.name for attr in self.parallelWidget.data.domain.attributes]
            attrInfo = OWVisAttrSelection.computeCorrelationBetweenAttributes(self.parallelWidget.data, attrList)
            #attrInfo = OWVisAttrSelection.computeCorrelationInsideClassesBetweenAttributes(self.parallelWidget.data, attrList)
        elif self.optimizationMeasure == VIZRANK:
            for (val, [a1, a2]) in self.projections:
                attrInfo.append((val, a1, a2))

        if len(attrInfo) == 0:
            print "len(attrInfo) == 0. No attribute pairs. Unable to optimize."; return

        self.worstVal = -1
        self.canOptimize = 1
        self.startOptimizationButton.hide()
        self.stopOptimizationButton.show()

        limit = getrecursionlimit()
        setrecursionlimit(max(limit, len(attrInfo)+1000))
        OWVisAttrSelection.optimizeAttributeOrder(attrInfo, [], 0.0, self.numberOfAttributes, self, qApp)
        setrecursionlimit(limit)

        self.stopOptimizationButton.hide()
        self.startOptimizationButton.show()
                    

    # ################################
    # MANAGE RESULTS
    def updateShownProjections(self, *args):
        self.resultList.clear()
        for i in range(min(len(self.allResults), self.resultListLen)):
            self.resultList.insertItem("%.2f - %s" % (self.allResults[i][0], str(self.allResults[i][1])), i)
        if self.resultList.count() > 0: self.resultList.setCurrentItem(0)  
    
    def clearResults(self):
        self.allResults = []
        self.resultList.clear()


    def saveResults(self, filename = None):
        if filename == None:
            name = str(QFileDialog.getSaveFileName( self.lastSaveDirName + "/" + "Parallel projections", "Parallel projections (*.papr)", self, "", "Save Parallel Projections"))
            if name == "": return
        else:
            name = filename

        # take care of extension
        if os.path.splitext(name)[1] != ".papr": name = name + ".papr"

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        # open, write and save file
        file = open(name, "wt")
        for val in self.allResults:
            file.write(str(val) + "\n")
        file.close()

    def loadResults(self):
        self.clearResults()
                
        name = str(QFileDialog.getOpenFileName( self.lastSaveDirName, "Parallel projections (*.papr)", self, "", "Open Parallel Projections"))
        if name == "": return

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        file = open(name, "rt")
        line = file.readline()[:-1]; ind = 0
        while (line != ""):
            (val, attrList) = eval(line)
            self.allResults.insert(ind, (val, attrList))
            self.resultList.insertItem("%.2f - %s" % (val, str(attrList)), ind)
            line = file.readline()[:-1]
            ind+=1
        file.close()

        
   

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWParallelCoordinates()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
