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
from random import betavariate 
from OWParallelGraph import *
import OWVisAttrSelection 
import OWToolbars    

###########################################################################################
##### WIDGET : Parallel coordinates visualization
###########################################################################################
class OWParallelCoordinates(OWWidget):
    settingsList = ["attrContOrder", "attrDiscOrder", "graphCanvasColor", "jitterSize", "showDistributions", "showAttrValues", "hidePureExamples", "showCorrelations", "globalValueScaling", "linesDistance", "useSplines", "lineTracking", "showLegend", "autoSendSelection", "sendShownAttributes"]
    #spreadType=["none","uniform","triangle","beta"]
    attributeContOrder = ["None","RelieF","Correlation"]
    attributeDiscOrder = ["None","RelieF","GainRatio", "Oblivious decision graphs"]
    jitterSizeList = ['0', '2','5','10', '15', '20', '30']
    jitterSizeNums = [0, 2,  5,  10, 15, 20, 30]
    linesDistanceList = ['20', '30', '40', '50', '60', '70', '80', '100']
    linesDistanceNums = [20, 30, 40, 50, 60, 70, 80, 100]

    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Parallel Coordinates", "Show data using parallel coordinates visualization method", FALSE, TRUE)
        self.resize(700,700)

        self.inputs = [("Examples", ExampleTable, self.data, 1), ("Selection", list, self.selection, 1)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass)]
    
        #set default settings
        self.attrDiscOrder = "None"
        self.attrContOrder = "None"
        #self.jitteringType = "uniform"
        self.GraphCanvasColor = str(Qt.white.name())
        self.showDistributions = 1
        self.showAttrValues = 1
        self.hidePureExamples = 1
        self.showCorrelations = 1
        self.GraphGridColor = str(Qt.black.name())
        self.data = None
        self.jitterSize = 10
        self.linesDistance = 40
        self.ShowVerticalGridlines = TRUE
        self.ShowHorizontalGridlines = TRUE
        self.globalValueScaling = 0
        self.useSplines = 0
        self.lineTracking = 0
        self.showLegend = 1
        self.autoSendSelection = 0
        self.sendShownAttributes = 1
        self.graphCanvasColor = str(Qt.white.name())

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = OWParallelCoordinatesOptions(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")

        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWParallelGraph(self.mainArea)
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

        self.connect(self.slider, SIGNAL("valueChanged(int)"), self.sliderValueChanged)

        # ####################################
        # SETTINGS functionality
        self.connect(self.SettingsTab.showDistributions, SIGNAL("toggled(bool)"), self.setDistributions)
        self.connect(self.SettingsTab.showAttrValues, SIGNAL("toggled(bool)"), self.setAttrValues)
        self.connect(self.SettingsTab.hidePureExamples, SIGNAL("toggled(bool)"), self.setHidePureExamples)
        self.connect(self.SettingsTab.showCorrelations, SIGNAL("toggled(bool)"), self.setShowCorrelations)
        self.connect(self.SettingsTab.showLegend, SIGNAL("toggled(bool)"), self.setLegend)
        self.connect(self.SettingsTab.useSplines, SIGNAL("toggled(bool)"), self.setUseSplines)
        self.connect(self.SettingsTab.lineTracking, SIGNAL("toggled(bool)"), self.setLineTracking)
        self.connect(self.SettingsTab.globalValueScaling, SIGNAL("toggled(bool)"), self.setGlobalValueScaling)
        self.connect(self.SettingsTab.jitterSize, SIGNAL("activated(int)"), self.setJitteringSize)
        self.connect(self.SettingsTab.linesDistance, SIGNAL("activated(int)"), self.setLinesDistance)
        self.connect(self.SettingsTab.attrContButtons, SIGNAL("clicked(int)"), self.setAttrContOrderType)
        self.connect(self.SettingsTab.attrDiscButtons, SIGNAL("clicked(int)"), self.setAttrDiscOrderType)
        #self.connect(self.SettingsTab.spreadButtons, SIGNAL("clicked(int)"), self.setSpreadType)
        self.connect(self.SettingsTab.autoSendSelection, SIGNAL("clicked()"), self.setAutoSendSelection)
        self.connect(self.SettingsTab.sendShownAttributes, SIGNAL("clicked()"), self.setSendShownAttributes)
        self.graph.autoSendSelectionCallback = self.setAutoSendSelection
        

        # add a settings dialog and initialize its values
        self.activateLoadedSettings()


    def resizeEvent(self, e):
        self.isResizing = 1
        #self.updateGraph() # had to comment this otherwise qt dll throws weird exception (since zooming and selection was added)

    def sliderValueChanged(self, val):
        self.updateGraph()

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        #self.SettingsTab.spreadButtons.setButton(self.spreadType.index(self.jitteringType))
        self.SettingsTab.attrContButtons.setButton(self.attributeContOrder.index(self.attrContOrder))
        self.SettingsTab.attrDiscButtons.setButton(self.attributeDiscOrder.index(self.attrDiscOrder))
        self.SettingsTab.showDistributions.setChecked(self.showDistributions)
        self.SettingsTab.showAttrValues.setChecked(self.showAttrValues)
        self.SettingsTab.hidePureExamples.setChecked(self.hidePureExamples)
        self.SettingsTab.showLegend.setChecked(self.showLegend)
        self.SettingsTab.showCorrelations.setChecked(self.showCorrelations)
        self.SettingsTab.useSplines.setChecked(self.useSplines)
        self.SettingsTab.globalValueScaling.setChecked(self.globalValueScaling)
        self.SettingsTab.lineTracking.setChecked(self.lineTracking)
        self.SettingsTab.autoSendSelection.setChecked(self.autoSendSelection)
        self.SettingsTab.sendShownAttributes.setChecked(self.sendShownAttributes)
        self.setAutoSendSelection() # update send button state
        
        self.SettingsTab.jitterSize.clear()
        for i in range(len(self.jitterSizeList)):
            self.SettingsTab.jitterSize.insertItem(self.jitterSizeList[i])
        self.SettingsTab.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))
        self.SettingsTab.linesDistance.clear()
        for i in range(len(self.linesDistanceList)):
            self.SettingsTab.linesDistance.insertItem(self.linesDistanceList[i])
        self.SettingsTab.linesDistance.setCurrentItem(self.linesDistanceNums.index(self.linesDistance))

        self.graph.updateSettings(enabledLegend = self.showLegend)
        self.graph.updateSettings(useSplines = self.useSplines)
        #self.graph.setJitteringOption(self.jitteringType)
        self.graph.setShowDistributions(self.showDistributions)
        self.graph.setShowAttrValues(self.showAttrValues)
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setJitterSize(self.jitterSize)
        self.graph.updateSettings(lineTracking = self.lineTracking)
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))

    """
    # jittering options
    def setSpreadType(self, n):
        self.jitteringType = self.spreadType[n]
        self.graph.setJitteringOption(self.spreadType[n])
        self.graph.setData(self.data)
        self.updateGraph()
    """

    # minimum lines distance
    def setLinesDistance(self, n):
        self.linesDistance = self.linesDistanceNums[n]
        self.updateGraph()

    # jittering options
    def setJitteringSize(self, n):
        self.jitterSize = self.jitterSizeNums[n]
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setData(self.data)
        self.updateGraph()

    def setDistributions(self, b):
        self.showDistributions = b
        self.graph.updateSettings(showDistributions = b)
        self.updateGraph()

    def setAttrValues(self, b):
        self.showAttrValues = b
        self.graph.setShowAttrValues(b)
        self.updateGraph()

    def setHidePureExamples(self, b):
        self.hidePureExamples = b
        self.graph.setHidePureExamples(b)
        self.updateGraph()
        
    def setShowCorrelations(self, b):
        self.showCorrelations = b
        self.graph.setShowCorrelations(b)
        self.updateGraph()

    def setUseSplines(self, b):
        self.useSplines = b
        self.graph.updateSettings(useSplines = b)
        self.updateGraph()

    def setLegend(self, b):
        self.showLegend = b
        self.graph.updateSettings(enabledLegend = self.showLegend)
        self.updateGraph()

    def setLineTracking(self, b):
        self.lineTracking = b
        self.graph.updateSettings(lineTracking = b)

    def setGlobalValueScaling(self, b):
        self.globalValueScaling = b
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setData(self.data)
        if self.globalValueScaling:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
        self.updateGraph()

    # continuous attribute ordering
    def setAttrContOrderType(self, n):
        self.attrContOrder = self.attributeContOrder[n]
        if self.data != None:
            self.setShownAttributeList(self.data)
        self.updateGraph()

    # discrete attribute ordering
    def setAttrDiscOrderType(self, n):
        self.attrDiscOrder = self.attributeDiscOrder[n]
        if self.data != None:
            self.setShownAttributeList(self.data)
        self.updateGraph()

    def setAutoSendSelection(self):
        self.autoSendSelection = self.SettingsTab.autoSendSelection.isChecked()
        if self.autoSendSelection:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(0)
            self.sendSelections()
        else:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(1)
            
    def setSendShownAttributes(self):
        self.sendShownAttributes = self.SettingsTab.sendShownAttributes.isChecked()

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
        #self.graph.replot()

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
        #self.graph.replot()

    # #####################

    def updateGraph(self, *args):
        graphWidth = self.width()-230
        attrs = self.getShownAttributeList()
        maxAttrs = graphWidth / self.linesDistance
        if len(attrs) > maxAttrs:
            rest = len(attrs) - maxAttrs
            if self.sliderRange != rest:
                self.slider.setRange(0, rest)
                self.sliderRange = rest
            elif self.isResizing:
                print "is resizing = 1"
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
        #self.repaint()


    # ###### SHOWN ATTRIBUTE LIST ##############
    # set attribute list
    def setShownAttributeList(self, data):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if data == None: return
        
        shown, hidden = OWVisAttrSelection.selectAttributes(data, self.attrContOrder, self.attrDiscOrder)
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

class OWParallelCoordinatesOptions(QVGroupBox):
    def __init__(self,parent = None, name = None):
        QVGroupBox.__init__(self, parent, name)
        self.parent = parent

        """
        # ####
        # jittering
        self.spreadButtons = QVButtonGroup("Jittering type", self)
        QToolTip.add(self.spreadButtons, "Selected the type of jittering for discrete variables")
        self.spreadButtons.setExclusive(TRUE)
        self.spreadNone = QRadioButton('none', self.spreadButtons)
        self.spreadUniform = QRadioButton('uniform', self.spreadButtons)
        self.spreadTriangle = QRadioButton('triangle', self.spreadButtons)
        self.spreadBeta = QRadioButton('beta', self.spreadButtons)
        """

        # #####
        # jittering options
        self.jitteringOptionsBG = QVButtonGroup("Jittering options", self)
        QToolTip.add(self.jitteringOptionsBG, "Percents of a discrete value to be jittered")
        self.hbox = QHBox(self.jitteringOptionsBG, "Jittering size")
        self.jitterLabel = QLabel('Jittering size (% of size)  ', self.hbox)
        self.jitterSize = QComboBox(self.hbox)

        # ####
        # attribute axis options
        self.linesDistanceOptionsBG = QVButtonGroup("Attribute axis distance", self)
        QToolTip.add(self.linesDistanceOptionsBG, "What is the minimum distance between two adjecent attribute axis")
        self.hbox2 = QHBox(self.linesDistanceOptionsBG, "Minimum distance")
        self.linesLabel = QLabel('Minimum distance (pixels)  ', self.hbox2)
        self.linesDistance = QComboBox(self.hbox2)        

        # ####
        # visual settings
        self.visualSettings = QVButtonGroup("Visual settings", self)
        self.showDistributions = QCheckBox("Show distributions", self.visualSettings)
        self.showAttrValues = QCheckBox("Show attribute values", self.visualSettings)
        self.hidePureExamples = QCheckBox("Hide pure examples", self.visualSettings)
        self.showCorrelations = QCheckBox("Show correlations between attributes", self.visualSettings)      # show correlations
        self.useSplines = QCheckBox("Show lines using splines", self.visualSettings)      # show correlations
        self.lineTracking = QCheckBox("Enable line tracking", self.visualSettings)      # show nearest line in bold
        self.showLegend = QCheckBox('Show legend', self.visualSettings)

        self.sendingSelectionsBG = QVButtonGroup("Sending selections", self)
        self.autoSendSelection = QCheckBox("Auto send selected data", self.sendingSelectionsBG)
        self.sendShownAttributes = QCheckBox("Send only shown attributes", self.sendingSelectionsBG)        

        # ####
        # attribute value scaling
        self.attrValueScalingButtons = QVButtonGroup("Attribute value scaling", self)
        self.globalValueScaling = QCheckBox("Global Value Scaling", self.attrValueScalingButtons)

        # ####        
        # continuous attribute ordering
        self.attrContButtons = QVButtonGroup("Continuous attribute ordering", self)
        QToolTip.add(self.attrContButtons, "Select the measure for continuous attribute ordering")
        self.attrContButtons.setExclusive(TRUE)
        
        self.attrContNone = QRadioButton('None', self.attrContButtons)
        self.attrContRelieF = QRadioButton('RelieF', self.attrContButtons)
        self.attrCorrelation = QRadioButton('Correlation', self.attrContButtons)

        # ####
        # discrete attribute ordering
        self.attrDiscButtons = QVButtonGroup("Discrete attribute ordering", self)
        QToolTip.add(self.attrDiscButtons, "Select the measure for discrete attribute ordering")
        self.attrDiscButtons.setExclusive(TRUE)

        self.attrDiscNone = QRadioButton('None', self.attrDiscButtons)
        self.attrDiscRelieF = QRadioButton('RelieF', self.attrDiscButtons)
        self.attrDiscGainRatio = QRadioButton('GainRatio', self.attrDiscButtons)
        #self.attrDiscGini = QRadioButton('Gini', self.attrDiscButtons)
        self.attrDiscFD   = QRadioButton('Oblivious decision graphs', self.attrDiscButtons)

        # ####
        self.gSetCanvasColorB = QPushButton("Canvas Color", self)
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)


    def setGraphCanvasColor(self):
        newColor = QColorDialog.getColor(QColor(self.parent.graphCanvasColor))
        if newColor.isValid():
            self.parent.graphCanvasColor = str(newColor.name())
            self.parent.graph.setCanvasColor(QColor(newColor))

   

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWParallelCoordinates()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
