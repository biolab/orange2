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

from OWVisWidget import *
from OWSurveyPlotGraph import *
import orngVisFuncts
import OWGUI
           
###########################################################################################
##### WIDGET : Survey plot visualization
###########################################################################################
class OWSurveyPlot(OWVisWidget):
    settingsList = ["attrDiscOrder", "attrContOrder", "graph.globalValueScaling", "graph.exampleTracking", "graph.enabledLegend", "graph.tooltipKind", "showAllAttributes"]
    attributeContOrder = ["None","ReliefF", "Fisher discriminant"]
    attributeDiscOrder = ["None","ReliefF","GainRatio"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Survey Plot", TRUE)

        self.inputs = [("Examples", ExampleTable, self.setData, Default), ("Attribute Selection List", AttributeList, self.setAttributeSelection)]
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
        self.primaryAttribute = "(None)"
        self.secondaryAttribute = "(None)"

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "Main")
        self.tabs.insertTab(self.SettingsTab, "Settings")

        #add controls to self.controlArea widget
        self.sortingAttrGB = OWGUI.widgetBox(self.GeneralTab, "Sorting")
        self.primaryAttrCombo = OWGUI.comboBoxWithCaption(self.sortingAttrGB, self, "primaryAttribute", label = '1st:', items = ["(None)"], sendSelectedValue = 1, valueType = str, callback = self.sortingClick, labelWidth = 25)
        self.secondaryAttrCombo = OWGUI.comboBoxWithCaption(self.sortingAttrGB, self, "secondaryAttribute", label = '2nd:', items = ["(None)"], sendSelectedValue = 1, valueType = str, callback = self.sortingClick, labelWidth = 25)

        self.createShowHiddenLists(self.GeneralTab, callback = self.updateGraph)

        # ##################################
        # survey plot settings
        box = OWGUI.widgetBox(self.SettingsTab, "Visual settings")
        OWGUI.checkBox(box, self, "graph.globalValueScaling", "Global Value Scaling", callback = self.setGlobalValueScaling)
        OWGUI.checkBox(box, self, "graph.exampleTracking", "Enable example tracking", callback = self.updateGraph)
        OWGUI.checkBox(box, self, "graph.enabledLegend", "Show legend", callback = self.updateGraph)

        OWGUI.comboBox(self.SettingsTab, self, "attrContOrder", box = "Continuous attribute ordering", items = self.attributeContOrder, callback = self.updateShownAttributeList, sendSelectedValue = 1, valueType = str)
        OWGUI.comboBox(self.SettingsTab, self, "attrDiscOrder", box = "Discrete attribute ordering", items = self.attributeDiscOrder, callback = self.updateShownAttributeList, sendSelectedValue = 1, valueType = str)

        box = OWGUI.widgetBox(self.SettingsTab, "Tooltips settings")
        OWGUI.comboBox(box, self, "graph.tooltipKind", items = ["Don't show tooltips", "Show visible attributes", "Show all attributes"], callback = self.updateGraph)

        OWGUI.button(self.SettingsTab, self, "Canvas Color", callback = self.setGraphCanvasColor, tooltip = "Set color for canvas background", debuggingEnabled = 0)

        self.icons = self.createAttributeIconDict()

        # add a settings dialog and initialize its values        
        self.activateLoadedSettings()
        self.resize(700,700)

        # this is needed so that the tabs are wide enough! 
        qApp.processEvents()
        self.tabs.updateGeometry()

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.graph.setCanvasColor(QColor(self.graphCanvasColor))
        #self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        self.cbShowAllAttributes()

    # #####################
    def setSortCombo(self):
        self.primaryAttrCombo.clear()
        self.secondaryAttrCombo.clear()
        self.primaryAttrCombo.insertItem("(None)")
        self.secondaryAttrCombo.insertItem("(None)")
        if not self.data: return
        for attr in self.data.domain:
            self.primaryAttrCombo.insertItem(self.icons[attr.varType], attr.name)
            self.secondaryAttrCombo.insertItem(self.icons[attr.varType], attr.name)
        #self.primaryAttrCombo.setCurrentItem(0)
        #self.secondaryAttrCombo.setCurrentItem(0)
        self.primaryAttribute = "(None)"
        self.secondaryAttribute = "(None)"
    
    def updateGraph(self, *args):
        self.graph.updateData(self.getShownAttributeList())
        self.graph.update()
        self.repaint()

    # set combo box values with attributes that can be used for coloring the data
    def sortingClick(self):
        attrs = [self.primaryAttribute, self.secondaryAttribute]
        while "(None)" in attrs: attrs.remove("(None)")
        if attrs and self.data:
            self.data.sort(attrs)

        self.graph.setData(self.data, sortValuesForDiscreteAttrs = 0)
        self.updateGraph()        
        
   
    # receive new data and update all fields
    def setData(self, data):
        if data:
            name = getattr(data, "name", "")
            data = data.filterref(orange.Filter_hasClassValue())
            data.name = name
        if self.data != None and data != None and self.data.checksum() == data.checksum():
            return    # check if the new data set is the same as the old one

        exData = self.data
        self.data = data
        
        sameDomain = self.data and exData and exData.domain.checksum() == self.data.domain.checksum() # preserve attribute choice if the domain is the same
        if not sameDomain:
            self.resetAttrManipulation()
            self.setSortCombo()
            self.setShownAttributeList(self.data)
        self.sortingClick()


    ####### SELECTION signal ################################
    # receive info about which attributes to show
    def setAttributeSelection(self, attributeSelectionList):
        self.attributeSelectionList = attributeSelectionList
        if self.data and self.attributeSelectionList:
            for attr in self.attributeSelectionList:
                if not self.graph.attributeNameIndex.has_key(attr):  # this attribute list belongs to a new dataset that has not come yet
                    return
            self.setShownAttributeList(self.data, self.attributeSelectionList)
        self.updateGraph()

    # update attribute ordering
    def updateShownAttributeList(self):
        self.setShownAttributeList(self.data)
        self.updateGraph()

    def sendShownAttributes(self):
        self.send("Attribute Selection List", [a[0] for a in self.shownAttributes])

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
