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

###########################################################################################
##### WIDGET : Parallel coordinates visualization
###########################################################################################
class OWParallelCoordinates(OWWidget):
    settingsList = ["attrContOrder", "attrDiscOrder", "graphCanvasColor", "jitterSize", "showDistributions", "showAttrValues", "hidePureExamples", "showCorrelations", "globalValueScaling", "linesDistance", "useSplines", "lineTracking", "showLegend", "autoSendSelection", "sendShownAttributes"]
    attributeContOrder = ["None","ReliefF","Correlation", "Fisher discriminant"]
    attributeDiscOrder = ["None","ReliefF","GainRatio", "Oblivious decision graphs"]
    jitterSizeNums = [0, 2,  5,  10, 15, 20, 30]
    linesDistanceNums = [20, 30, 40, 50, 60, 70, 80, 100, 120, 150]

    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Parallel Coordinates", "Show data using parallel coordinates visualization method", FALSE, TRUE)
        self.resize(700,700)

        self.inputs = [("Examples", ExampleTable, self.data, 1), ("Selection", list, self.selection, 1)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass), ("Attribute selection", list)]
    
        #set default settings
        self.data = None

        self.jitterSize = 10
        self.linesDistance = 40
        
        self.showDistributions = 1
        self.showAttrValues = 1
        self.hidePureExamples = 1
        self.showCorrelations = 1
        
        self.globalValueScaling = 0
        self.useSplines = 0
        self.lineTracking = 0
        self.showLegend = 1
        self.autoSendSelection = 0
        self.sendShownAttributes = 1
        self.attrDiscOrder = "None"
        self.attrContOrder = "None"
        self.graphCanvasColor = str(Qt.white.name())

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

        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph)
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
        OWGUI.checkBox(box, self, 'showDistributions', 'Show distributions', callback = self.setDistributions, tooltip = "Show bars with distribution of class values")
        OWGUI.checkBox(box, self, 'showAttrValues', 'Show attribute values', callback = self.setAttrValues)
        OWGUI.checkBox(box, self, 'hidePureExamples', 'Hide pure examples', callback = self.setHidePureExamples, tooltip = "When one value of a discrete attribute has only examples from one class, \nstop drawing lines for this example. Figure must be interpreted from left to right.")
        OWGUI.checkBox(box, self, 'showCorrelations', 'Show correlations', callback = self.setShowCorrelations, tooltip = "Show correlations between two neighboring attributes")
        OWGUI.checkBox(box, self, 'useSplines', 'Show splines', callback = self.setUseSplines, tooltip  = "Show lines using splines")
        OWGUI.checkBox(box, self, 'lineTracking', 'Line tracking', callback = self.setLineTracking, tooltip = "Show nearest example with a wide line")
        OWGUI.checkBox(box, self, 'showLegend', 'Show legend', callback = self.setLegend)
        OWGUI.checkBox(box, self, 'globalValueScaling', 'Global Value Scaling', callback = self.setGlobalValueScaling)
        
        
        box2 = OWGUI.widgetBox(self.SettingsTab, " Sending selection ")
        OWGUI.checkBox(box2, self, 'autoSendSelection', 'Auto send selected data', callback = self.setAutoSendSelection, tooltip = "Send signals with selected data whenever the selection changes.")
        OWGUI.checkBox(box2, self, 'sendShownAttributes', 'Send only shown attributes')
        
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
        self.graph.updateSettings(enabledLegend = self.showLegend)
        self.graph.updateSettings(useSplines = self.useSplines)
        self.graph.setShowDistributions(self.showDistributions)
        self.graph.setShowAttrValues(self.showAttrValues)
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setJitterSize(self.jitterSize)
        self.graph.updateSettings(lineTracking = self.lineTracking)
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        if not self.data: return
        (selected, unselected, merged) = self.graph.getSelectionsAsExampleTables()
        if not self.sendShownAttributes:
            self.send("Selected Examples",selected)
            self.send("Unselected Examples",unselected)
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
        else:
            self.slider.setRange(0,0)
            self.sliderRange = 0
            maxAttrs = len(attrs)

        start = min(self.slider.value(), len(attrs)-maxAttrs)
        targetVal = self.targetValueCombo.currentText()
        if targetVal == "(None)": targetVal = None
        self.graph.updateData(attrs[start:start+maxAttrs], targetVal)
        self.slider.repaint()
        self.graph.update()
        #self.graph.repaint()


    # ###### SHOWN ATTRIBUTE LIST ##############
    # set attribute list
    def setShownAttributeList(self, data):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if data == None: return
        
        shown, hidden = OWVisAttrSelection.selectAttributes(data, self.graph, self.attrContOrder, self.attrDiscOrder)
        if data.domain.classVar and data.domain.classVar.name not in shown and data.domain.classVar.name not in hidden:
            self.shownAttribsLB.insertItem(data.domain.classVar.name)
        for attr in shown:
            self.shownAttribsLB.insertItem(attr)
        for attr in hidden:
            self.hiddenAttribsLB.insertItem(attr)
        
        
    def getShownAttributeList (self):
        list = []
        for i in range(self.shownAttribsLB.count()):
            list.append(str(self.shownAttribsLB.text(i)))
        return list
    ##############################################

    # had to override standart show to call updateGraph. otherwise self.mainArea.width() gives incorrect value    
    def show(self):
        OWWidget.show(self)
        self.updateGraph()
    
    ####### DATA ################################
    # receive new data and update all fields
    def data(self, data):
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
    #################################################

    
    ####### SELECTION ################################
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


    def resizeEvent(self, e):
        self.isResizing = 1
        # self.updateGraph()  # had to comment, otherwise python throws an exception

    # jittering options
    def setJitteringSize(self):
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setData(self.data)
        self.updateGraph()

    def setDistributions(self):
        self.graph.updateSettings(showDistributions = self.showDistributions)
        self.updateGraph()

    def setAttrValues(self):
        self.graph.setShowAttrValues(self.showAttrValues)
        self.updateGraph()

    def setHidePureExamples(self):
        self.graph.setHidePureExamples(self.hidePureExamples)
        self.updateGraph()
        
    def setShowCorrelations(self):
        self.graph.setShowCorrelations(self.showCorrelations)
        self.updateGraph()

    def setUseSplines(self):
        self.graph.updateSettings(useSplines = self.useSplines)
        self.updateGraph()

    def setLegend(self):
        self.graph.updateSettings(enabledLegend = self.showLegend)
        self.updateGraph()

    def setLineTracking(self):
        self.graph.updateSettings(lineTracking = self.lineTracking)

    def setGlobalValueScaling(self):
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setData(self.data)
        if self.globalValueScaling:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
        self.updateGraph()

    # continuous attribute ordering
    def updateShownAttributeList(self):
        if self.data != None:
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

   

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWParallelCoordinates()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
