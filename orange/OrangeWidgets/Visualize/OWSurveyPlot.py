"""
<name>Survey Plot</name>
<description>Shows data using survey plot visualization method</description>
<category>Visualization</category>
<icon>icons/SurveyPlot.png</icon>
<priority>3250</priority>
"""
# OWSurveyPlot.py
#
# Show data using survey plot visualization method
# 

from OWWidget import *
from OWSurveyPlotGraph import *
import OWVisAttrSelection

           
###########################################################################################
##### WIDGET : Survey plot visualization
###########################################################################################
class OWSurveyPlot(OWWidget):
    settingsList = ["attrDiscOrder", "attrContOrder", "globalValueScaling", "exampleTracking", "showLegend"]
    attributeContOrder = ["None","RelieF","Correlation"]
    attributeDiscOrder = ["None","RelieF","GainRatio","Gini", "Oblivious decision graphs"]

    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Survey Plot", "Show data using survey plot visualization method", FALSE, TRUE)

        self.inputs = [("Examples", ExampleTable, self.cdata, 1)]
        self.outputs = [("Selection", list)] 

        #set default settings
        self.attrDiscOrder = "RelieF"
        self.attrContOrder = "RelieF"
        self.GraphCanvasColor = str(Qt.white.name())
        self.data = None
        self.globalValueScaling = 0
        self.exampleTracking = 1
        self.showLegend = 1
        self.graphCanvasColor = str(Qt.white.name())

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = OWSurveyPlotOptions(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")

        #connect settingsbutton to show options
        self.connect(self.SettingsTab.globalValueScaling, SIGNAL("toggled(bool)"), self.setGlobalValueScaling)
        self.connect(self.SettingsTab.exampleTracking, SIGNAL("toggled(bool)"), self.setExampleTracking)
        self.connect(self.SettingsTab.attrContButtons, SIGNAL("clicked(int)"), self.setAttrContOrderType)
        self.connect(self.SettingsTab.attrDiscButtons, SIGNAL("clicked(int)"), self.setAttrDiscOrderType)
        self.connect(self.SettingsTab.showLegend, SIGNAL("toggled(bool)"), self.setLegend)
        self.connect(self.SettingsTab, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)
        
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWSurveyPlotGraph(self.mainArea)
        self.box.addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        
        #add controls to self.controlArea widget
        self.sortingAttrGB = QVGroupBox(self.GeneralTab)
        self.shownAttribsGroup = QVGroupBox(self.GeneralTab)
        self.addRemoveGroup = QHButtonGroup(self.GeneralTab)
        self.hiddenAttribsGroup = QVGroupBox(self.GeneralTab)
        self.sortingAttrGB.setTitle("Sorting")
        self.shownAttribsGroup.setTitle("Shown attributes")
        self.hiddenAttribsGroup.setTitle("Hidden attributes")

        self.primarySortCB = QCheckBox('Enable sorting primary', self.sortingAttrGB)
        self.primaryAttr = QComboBox(self.sortingAttrGB)
        self.connect(self.primarySortCB, SIGNAL("clicked()"), self.sortingClick)
        self.connect(self.primaryAttr, SIGNAL('activated ( const QString & )'), self.sortingClick)

        self.secondarySortCB = QCheckBox('Enable sorting secondary', self.sortingAttrGB)
        self.secondaryAttr = QComboBox(self.sortingAttrGB)
        self.connect(self.secondarySortCB, SIGNAL("clicked()"), self.sortingClick)
        self.connect(self.secondaryAttr, SIGNAL('activated ( const QString & )'), self.sortingClick)

        self.primarySortCB.setChecked(0)
        self.secondarySortCB.setChecked(0)

        self.shownAttribsLB = QListBox(self.shownAttribsGroup)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)
        
        self.hbox = QHBox(self.shownAttribsGroup)
        self.buttonUPAttr = QPushButton("Attr UP", self.hbox)
        self.buttonDOWNAttr = QPushButton("Attr DOWN", self.hbox)

        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        self.connect(self.buttonUPAttr, SIGNAL("clicked()"), self.moveAttrUP)
        self.connect(self.buttonDOWNAttr, SIGNAL("clicked()"), self.moveAttrDOWN)

        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)

        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
        
        # add a settings dialog and initialize its values
        self.activateLoadedSettings()
        self.resize(700,700)

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.SettingsTab.attrContButtons.setButton(self.attributeContOrder.index(self.attrContOrder))
        self.SettingsTab.attrDiscButtons.setButton(self.attributeDiscOrder.index(self.attrDiscOrder))
        self.SettingsTab.globalValueScaling.setChecked(self.globalValueScaling)
        self.SettingsTab.showLegend.setChecked(self.showLegend)
        self.SettingsTab.exampleTracking.setChecked(self.exampleTracking)

        self.graph.updateSettings(enabledLegend = self.showLegend)        
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.updateSettings(exampleTracking = self.exampleTracking)
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))

    # just tell the graph to hide the selected rectangle
    def enterEvent(self, e):
        self.graph.hideSelectedRectangle()
        self.graph.replot()

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

    def setGlobalValueScaling(self, b):
        self.globalValueScaling = b
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setData(self.data)

        # this is not optimal, because we do the rescaling twice (TO DO)
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())

        self.updateGraph()

    def setExampleTracking(self, b):
        self.exampleTracking = b
        self.graph.updateSettings(exampleTracking = b)
        
    def setCanvasColor(self, c):
        self.GraphCanvasColor = c
        self.graph.setCanvasColor(c)

    def setLegend(self, b):
        self.showLegend = b
        self.graph.updateSettings(enabledLegend = self.showLegend)
        self.updateGraph()
        
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
    def setSortCombo(self):
        self.primaryAttr.clear()
        self.secondaryAttr.clear()
        if not self.data: return
        for i in range(len(self.data.domain)):
            self.primaryAttr.insertItem(self.data.domain[i].name)
            self.secondaryAttr.insertItem(self.data.domain[i].name)
        self.primaryAttr.setCurrentItem(0)
        self.secondaryAttr.setCurrentItem(len(self.data.domain)>1)
    

    def sortData2(self, primaryAttr, secondaryAttr, data):
        if not data: return None
        data.sort([primaryAttr, secondaryAttr])
        return data

    def sortData1(self, primaryAttr, data):
        if not data: return None
        data.sort(primaryAttr)
        return data
        
    def updateGraph(self, *args):
        self.graph.updateData(self.getShownAttributeList(), self.statusBar)
        self.graph.update()
        self.repaint()

    # set combo box values with attributes that can be used for coloring the data
    def sortingClick(self, *args):
        primaryOn = self.primarySortCB.isOn()
        secondaryOn = self.secondarySortCB.isOn()

        primaryAttr = str(self.primaryAttr.currentText())
        secondaryAttr = str(self.secondaryAttr.currentText())

        if secondaryOn == 1 and secondaryAttr != "":
            primaryOn = 1
            self.primarySortCB.setChecked(1)
            
        if primaryOn == 1 and secondaryOn == 1 and primaryAttr != "" and secondaryAttr != "":
            self.data = self.sortData2(primaryAttr, secondaryAttr, self.data)
        elif primaryOn == 1 and primaryAttr != "":
            self.data = self.sortData1(primaryAttr, self.data)

        self.graph.setData(self.data)
        self.updateGraph()        
        
    # ###### SHOWN ATTRIBUTE LIST ##############
    # set attribute list
    def setShownAttributeList(self, data):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()
        if data == None: return
        
        shown, hidden = OWVisAttrSelection.selectAttributes(data, self.graph, self.attrContOrder, self.attrDiscOrder)
        if data.domain.classVar.name not in shown and data.domain.classVar.name not in hidden:
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
    
    
    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        exData = self.data
        self.data = None
        if data: self.data = orange.Preprocessor_dropMissingClasses(data)

        if self.data and exData and str(exData.domain.attributes) == str(self.data.domain.attributes): # preserve attribute choice if the domain is the same
            self.sortingClick()
            return  
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        self.setSortCombo()
        self.setShownAttributeList(self.data)
        self.sortingClick()
    #################################################

    ####### SELECTION signal ################################
    # receive info about which attributes to show
    def selection(self, list):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if self.data == None: return

        for attr in self.data.domain:
            if attr.name in list: self.shownAttribsLB.insertItem(attr.name)
            else:                 self.hiddenAttribsLB.insertItem(attr.name)

        self.updateGraph()
    #################################################

class OWSurveyPlotOptions(QVGroupBox):
    def __init__(self,parent=None,name=None):
        QVGroupBox.__init__(self, parent, name)
        self.parent = parent

        # ####
        # attribute value scaling
        self.attrValueScalingButtons = QVButtonGroup("Attribute value scaling", self)
        self.globalValueScaling = QCheckBox("Global Value Scaling", self.attrValueScalingButtons)

        # ####
        # visual settings
        self.visualSettingsButtons = QVButtonGroup("Visual settings", self)
        self.exampleTracking = QCheckBox("Enable example tracking", self.visualSettingsButtons)
        self.showLegend = QCheckBox('show legend', self.visualSettingsButtons)
        

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
        self.attrDiscGini = QRadioButton('Gini', self.attrDiscButtons)
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
    ow=OWSurveyPlot()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
