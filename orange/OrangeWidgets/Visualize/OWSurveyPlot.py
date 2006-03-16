"""
<name>Survey Plot</name>
<description>Survey plot (multiattribute) visualization.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/SurveyPlot.png</icon>
<priority>3250</priority>
"""
# OWSurveyPlot.py
#
# Show data using survey plot visualization method
# 

from OWWidget import *
from OWSurveyPlotGraph import *
import orngVisFuncts
import OWGUI
           
###########################################################################################
##### WIDGET : Survey plot visualization
###########################################################################################
class OWSurveyPlot(OWWidget):
    settingsList = ["attrDiscOrder", "attrContOrder", "graph.globalValueScaling", "graph.exampleTracking", "graph.enabledLegend", "graph.tooltipKind", "showAllAttributes"]
    attributeContOrder = ["None","ReliefF", "Fisher discriminant"]
    attributeDiscOrder = ["None","ReliefF","GainRatio", "Oblivious decision graphs"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Survey Plot", TRUE)

        self.inputs = [("Examples", ExampleTable, self.cdata, Default), ("Attribute Selection List", AttributeList, self.attributeSelection)]
        self.outputs = [("Attribute Selection List", AttributeList)]

        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWSurveyPlotGraph(self.mainArea)
        self.box.addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)


        #set default settings
        self.data = None
        self.showAllAttributes = 0
        self.graph.globalValueScaling = 0
        self.graph.exampleTracking = 0
        self.graph.enabledLegend = 1
        self.graph.tooltipKind = 1
        self.attrDiscOrder = "None"
        self.attrContOrder = "None"
        self.attributeSelectionList = None
        self.graphCanvasColor = str(Qt.white.name())

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")

        #add controls to self.controlArea widget
        self.sortingAttrGB = QVGroupBox(self.GeneralTab)
        self.shownAttribsGroup = QVGroupBox(self.GeneralTab)
        hbox = OWGUI.widgetBox(self.shownAttribsGroup, orientation = 'horizontal')
        self.addRemoveGroup = QHButtonGroup(self.GeneralTab)
        self.hiddenAttribsGroup = QVGroupBox(self.GeneralTab)
        self.sortingAttrGB.setTitle("Sorting")
        self.shownAttribsGroup.setTitle("Shown attributes")
        self.hiddenAttribsGroup.setTitle("Hidden attributes")

        self.primarySortCB = QCheckBox('Enable primary sorting by:', self.sortingAttrGB)
        self.primaryAttr = QComboBox(self.sortingAttrGB)
        self.connect(self.primarySortCB, SIGNAL("clicked()"), self.sortingClick)
        self.connect(self.primaryAttr, SIGNAL('activated ( const QString & )'), self.sortingClick)

        self.secondarySortCB = QCheckBox('Enable secondary sorting by:', self.sortingAttrGB)
        self.secondaryAttr = QComboBox(self.sortingAttrGB)
        self.connect(self.secondarySortCB, SIGNAL("clicked()"), self.sortingClick)
        self.connect(self.secondaryAttr, SIGNAL('activated ( const QString & )'), self.sortingClick)

        self.primarySortCB.setChecked(0)
        self.secondarySortCB.setChecked(0)

        self.shownAttribsLB = QListBox(hbox)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)
        
        vbox = OWGUI.widgetBox(hbox, orientation = 'vertical')
        self.buttonUPAttr   = OWGUI.button(vbox, self, "", callback = self.moveAttrUP, tooltip="Move selected attributes up")
        self.buttonDOWNAttr = OWGUI.button(vbox, self, "", callback = self.moveAttrDOWN, tooltip="Move selected attributes down")
        self.buttonUPAttr.setPixmap(QPixmap(os.path.join(self.widgetDir, r"icons\Dlg_up1.png")))
        self.buttonUPAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Expanding))
        self.buttonUPAttr.setMaximumWidth(20)
        self.buttonDOWNAttr.setPixmap(QPixmap(os.path.join(self.widgetDir, r"icons\Dlg_down1.png")))
        self.buttonDOWNAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Expanding))
        self.buttonDOWNAttr.setMaximumWidth(20)
        self.buttonUPAttr.setMaximumWidth(20)

        self.attrAddButton =    OWGUI.button(self.addRemoveGroup, self, "", callback = self.addAttribute, tooltip="Add (show) selected attributes")
        self.attrAddButton.setPixmap(QPixmap(os.path.join(self.widgetDir, r"icons\Dlg_up2.png")))
        self.attrRemoveButton = OWGUI.button(self.addRemoveGroup, self, "", callback = self.removeAttribute, tooltip="Remove (hide) selected attributes")
        self.attrRemoveButton.setPixmap(QPixmap(os.path.join(self.widgetDir, r"icons\Dlg_down2.png")))
        OWGUI.checkBox(self.addRemoveGroup, self, "showAllAttributes", "Show all", callback = self.cbShowAllAttributes) 

        # ##################################
        # survey plot settings
        # ####
        box = OWGUI.widgetBox(self.SettingsTab, " Visual settings ")
        OWGUI.checkBox(box, self, "graph.globalValueScaling", "Global Value Scaling", callback = self.setGlobalValueScaling)
        OWGUI.checkBox(box, self, "graph.exampleTracking", "Enable example tracking", callback = self.updateValues)
        OWGUI.checkBox(box, self, "graph.enabledLegend", "Show legend", callback = self.updateValues)

        OWGUI.comboBox(self.SettingsTab, self, "attrContOrder", box = " Continuous attribute ordering ", items = self.attributeContOrder, callback = self.updateShownAttributeList, sendSelectedValue = 1, valueType = str)
        OWGUI.comboBox(self.SettingsTab, self, "attrDiscOrder", box = " Discrete attribute ordering ", items = self.attributeDiscOrder, callback = self.updateShownAttributeList, sendSelectedValue = 1, valueType = str)

        box = OWGUI.widgetBox(self.SettingsTab, " Tooltips settings ")
        OWGUI.comboBox(box, self, "graph.tooltipKind", items = ["Don't show tooltips", "Show visible attributes", "Show all attributes"], callback = self.updateValues)

        OWGUI.button(self.SettingsTab, self, "Canvas Color", callback = self.setGraphCanvasColor, tooltip = "Set color for canvas background", debuggingEnabled = 0)

        self.icons = self.createAttributeIconDict()

        # add a settings dialog and initialize its values        
        self.activateLoadedSettings()
        self.resize(700,700)

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.graph.setCanvasColor(QColor(self.graphCanvasColor))
        #self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        self.cbShowAllAttributes()

    # ####################
    # LIST BOX FUNCTIONS
    # ####################

    # move selected attribute in "Attribute Order" list one place up
    def moveAttrUP(self):
        for i in range(1, self.shownAttribsLB.count()):
            if self.shownAttribsLB.isSelected(i):
                self.shownAttribsLB.insertItem(self.shownAttribsLB.pixmap(i), self.shownAttribsLB.text(i), i-1)
                self.shownAttribsLB.removeItem(i+1)
                self.shownAttribsLB.setSelected(i-1, TRUE)
        self.updateGraph()

    # move selected attribute in "Attribute Order" list one place down  
    def moveAttrDOWN(self):
        count = self.shownAttribsLB.count()
        for i in range(count-2,-1,-1):
            if self.shownAttribsLB.isSelected(i):
                self.shownAttribsLB.insertItem(self.shownAttribsLB.pixmap(i), self.shownAttribsLB.text(i), i+2)
                self.shownAttribsLB.removeItem(i)
                self.shownAttribsLB.setSelected(i+1, TRUE)
        self.updateGraph()

    def cbShowAllAttributes(self):
        if self.showAllAttributes:
            self.addAttribute(True)
        self.attrRemoveButton.setDisabled(self.showAllAttributes)
        self.attrAddButton.setDisabled(self.showAllAttributes)

    def addAttribute(self, addAll = False):
        count = self.hiddenAttribsLB.count()
        pos   = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):
            if addAll or self.hiddenAttribsLB.isSelected(i):
                self.shownAttribsLB.insertItem(self.hiddenAttribsLB.pixmap(i), self.hiddenAttribsLB.text(i), pos)
                self.hiddenAttribsLB.removeItem(i)
                
        if self.graph.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())

        self.sendShownAttributes()
        self.updateGraph()
        self.graph.removeAllSelections()
        #self.graph.replot()

    def removeAttribute(self):
        count = self.shownAttribsLB.count()
        pos   = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.shownAttribsLB.isSelected(i):
                self.hiddenAttribsLB.insertItem(self.shownAttribsLB.pixmap(i), self.shownAttribsLB.text(i), pos)
                self.shownAttribsLB.removeItem(i)
                
        if self.graph.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
        self.updateGraph()
        self.sendShownAttributes()
        #self.graph.replot()

    # #####################
    def setSortCombo(self):
        self.primaryAttr.clear()
        self.secondaryAttr.clear()
        if not self.data: return
        for attr in self.data.domain:
            self.primaryAttr.insertItem(self.icons[attr.varType], attr.name)
            self.secondaryAttr.insertItem(self.icons[attr.varType], attr.name)
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
    def setShownAttributeList(self, data, shownAttributes = None):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if data == None: return

        if shownAttributes:
            for attr in shownAttributes:
                self.shownAttribsLB.insertItem(self.icons[self.data.domain[self.graph.attributeNameIndex[attr]].varType], attr)
                
            for attr in data.domain:
                if attr.name not in shownAttributes:
                    self.hiddenAttribsLB.insertItem(self.icons[attr.varType], attr.name)
        else:
            shown, hidden, maxIndex = orngVisFuncts.selectAttributes(data, self.attrContOrder, self.attrDiscOrder)
            if data.domain.classVar.name not in shown and data.domain.classVar.name not in hidden:
                self.shownAttribsLB.insertItem(self.icons[data.domain.classVar.varType], data.domain.classVar.name)
            for attr in shown[:10]:
                self.shownAttribsLB.insertItem(self.icons[data.domain[attr].varType], attr)
            for attr in shown[10:] + hidden:
                self.hiddenAttribsLB.insertItem(self.icons[data.domain[attr].varType], attr)    
        self.sendShownAttributes()
        
    def getShownAttributeList (self):
        return [str(self.shownAttribsLB.text(i)) for i in range(self.shownAttribsLB.count())]

    def sendShownAttributes(self):
        self.send("Attribute Selection List", self.getShownAttributeList())
    
    ##############################################
    
    
    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        if data:
            name = ""
            if hasattr(data, "name"): name = data.name
            data = orange.Preprocessor_dropMissingClasses(data)
            data.name = name
        if self.data != None and data != None and self.data.checksum() == data.checksum(): return    # check if the new data set is the same as the old one
        exData = self.data
        self.data = data
        
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
    def attributeSelection(self, attributeSelectionList):
        self.attributeSelectionList = attributeSelectionList
        if self.data and self.attributeSelectionList:
            for attr in self.attributeSelectionList:
                if not self.graph.attributeNameIndex.has_key(attr):  # this attribute list belongs to a new dataset that has not come yet
                    return
            self.setShownAttributeList(self.data, self.attributeSelectionList)
        self.updateGraph()
    #################################################

    def updateValues(self):
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
        self.graph.setData(self.data)

        # this is not optimal, because we do the rescaling twice (TO DO)
        if self.graph.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())

        self.updateGraph()

    def setGraphCanvasColor(self):
        newColor = QColorDialog.getColor(QColor(self.graphCanvasColor))
        if newColor.isValid():
            self.graphCanvasColor = str(newColor.name())
            self.graph.setCanvasColor(QColor(newColor))



#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSurveyPlot()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
