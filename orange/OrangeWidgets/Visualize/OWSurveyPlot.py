"""
<name>Survey Plot</name>
<description>Shows data using survey plot visualization method</description>
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
import OWGUI
           
###########################################################################################
##### WIDGET : Survey plot visualization
###########################################################################################
class OWSurveyPlot(OWWidget):
    settingsList = ["attrDiscOrder", "attrContOrder", "globalValueScaling", "exampleTracking", "showLegend", "tooltipKind"]
    attributeContOrder = ["None","ReliefF", "Fisher discriminant"]
    attributeDiscOrder = ["None","ReliefF","GainRatio", "Oblivious decision graphs"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Survey Plot", TRUE)

        self.inputs = [("Examples", ExampleTable, self.cdata)]
        self.outputs = [("Selection", list)] 

        #set default settings
        self.data = None
        self.globalValueScaling = 0
        self.exampleTracking = 1
        self.showLegend = 1
        self.attrDiscOrder = "None"
        self.attrContOrder = "None"
        self.tooltipKind = 1
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

        # ##################################
        # survey plot settings
        # ####
        box = OWGUI.widgetBox(self.SettingsTab, " Visual settings ")
        OWGUI.checkBox(box, self, "globalValueScaling", "Global Value Scaling", callback = self.setGlobalValueScaling)
        OWGUI.checkBox(box, self, "exampleTracking", "Enable example tracking", callback = self.updateValues)
        OWGUI.checkBox(box, self, "showLegend", "Show legend", callback = self.updateValues)

        OWGUI.comboBox(self.SettingsTab, self, "attrContOrder", box = " Continuous attribute ordering ", items = self.attributeContOrder, callback = self.updateShownAttributeList, sendSelectedValue = 1, valueType = str)
        OWGUI.comboBox(self.SettingsTab, self, "attrDiscOrder", box = " Discrete attribute ordering ", items = self.attributeDiscOrder, callback = self.updateShownAttributeList, sendSelectedValue = 1, valueType = str)

        box = OWGUI.widgetBox(self.SettingsTab, " Tooltips settings ")
        OWGUI.comboBox(box, self, "tooltipKind", items = ["Don't show tooltips", "Show visible attributes", "Show all attributes"], callback = self.updateValues)


        self.gSetCanvasColorB = QPushButton("Canvas Color", self.SettingsTab)
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)

        # add a settings dialog and initialize its values
        self.activateLoadedSettings()
        self.resize(700,700)

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.graph.exampleTracking = self.exampleTracking
        self.graph.enabledLegend = self.showLegend
        self.graph.globalValueScaling = self.globalValueScaling
        self.graph.tooltipKind = self.tooltipKind
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        

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
        self.graph.updateData(self.getShownAttributeList())
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
        
        shown, hidden, maxIndex = OWVisAttrSelection.selectAttributes(data, self.attrContOrder, self.attrDiscOrder)
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

    def updateValues(self):
        self.graph.exampleTracking = self.exampleTracking
        self.graph.enabledLegend = self.showLegend
        self.graph.tooltipKind = self.tooltipKind
        self.updateGraph()

    # update attribute ordering
    def updateShownAttributeList(self):
        self.setShownAttributeList(self.data)
        self.updateGraph()

    # just tell the graph to hide the selected rectangle
    def enterEvent(self, e):
        self.graph.hideSelectedRectangle()
        self.graph.replot()

    def setGlobalValueScaling(self):
        self.graph.globalValueScaling = self.globalValueScaling
        self.graph.setData(self.data)

        # this is not optimal, because we do the rescaling twice (TO DO)
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())

        self.updateGraph()

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
