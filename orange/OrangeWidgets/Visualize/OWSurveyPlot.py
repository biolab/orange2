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
import OWColorPalette
import orngVisFuncts
import OWGUI

###########################################################################################
##### WIDGET : Survey plot visualization
###########################################################################################
class OWSurveyPlot(OWVisWidget):
    settingsList = ["attrDiscOrder", "attrContOrder", "graph.globalValueScaling", "graph.exampleTracking", "graph.enabledLegend",
                    "graph.tooltipKind", "showAllAttributes", "colorSettings", "selectedSchemaIndex"]
    attributeContOrder = ["Unordered","ReliefF", "Fisher discriminant"]
    attributeDiscOrder = ["Unordered","ReliefF","GainRatio"]
    contextHandlers = {"": DomainContextHandler("", [ContextField("shownAttributes", DomainContextHandler.RequiredList, selected="selectedShown", reservoir="hiddenAttributes")])}

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Survey Plot", TRUE)

        self.inputs = [("Examples", ExampleTable, self.setData, Default), ("Attribute Selection List", AttributeList, self.setShownAttributes)]
        self.outputs = [("Attribute Selection List", AttributeList)]

        #add a graph widget
        self.graph = OWSurveyPlotGraph(self.mainArea)
        self.mainArea.layout().addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        #set default settings
        self.data = None
        self.showAllAttributes = 0
        self.graph.globalValueScaling = 0
        self.graph.exampleTracking = 0
        self.graph.enabledLegend = 1
        self.graph.tooltipKind = 1
        self.attrDiscOrder = "Unordered"
        self.attrContOrder = "Unordered"
        self.attributeSelectionList = None
        self.graphCanvasColor = str(QColor(Qt.white).name())
        self.primaryAttribute = "(None)"
        self.secondaryAttribute = "(None)"
        self.colorSettings = None
        self.selectedSchemaIndex = 0

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.GeneralTab = OWGUI.createTabPage(self.tabs, "Main")
        self.SettingsTab = OWGUI.createTabPage(self.tabs, "Settings")


        #add controls to self.controlArea widget
        self.sortingAttrGB = OWGUI.widgetBox(self.GeneralTab, "Sorting")
        self.primaryAttrCombo = OWGUI.comboBoxWithCaption(self.sortingAttrGB, self, "primaryAttribute", label = '1st:', items = ["(None)"], sendSelectedValue = 1, valueType = str, callback = self.sortingClick, labelWidth = 25)
        self.secondaryAttrCombo = OWGUI.comboBoxWithCaption(self.sortingAttrGB, self, "secondaryAttribute", label = '2nd:', items = ["(None)"], sendSelectedValue = 1, valueType = str, callback = self.sortingClick, labelWidth = 25)

        self.createShowHiddenLists(self.GeneralTab, callback = self.updateGraph)

        # ##################################
        # survey plot settings
        box = OWGUI.widgetBox(self.SettingsTab, "Visual settings")
        OWGUI.checkBox(box, self, "graph.globalValueScaling", "Global value scaling", callback = self.setGlobalValueScaling)
        OWGUI.checkBox(box, self, "graph.exampleTracking", "Example tracking", callback = self.updateGraph)
        OWGUI.checkBox(box, self, "graph.enabledLegend", "Show legend", callback = self.updateGraph)

        OWGUI.comboBox(self.SettingsTab, self, "attrContOrder", box = "Continuous attribute ordering", items = self.attributeContOrder, callback = self.updateShownAttributeList, sendSelectedValue = 1, valueType = str)
        OWGUI.comboBox(self.SettingsTab, self, "attrDiscOrder", box = "Discrete attribute ordering", items = self.attributeDiscOrder, callback = self.updateShownAttributeList, sendSelectedValue = 1, valueType = str)

        box = OWGUI.widgetBox(self.SettingsTab, "Tooltips settings")
        OWGUI.comboBox(box, self, "graph.tooltipKind", items = ["Don't Show Tooltips", "Show Visible Attributes", "Show All Attributes"], callback = self.updateGraph)

        self.colorButtonsBox = OWGUI.widgetBox(self.SettingsTab, "Colors", orientation = "horizontal")
        OWGUI.button(self.colorButtonsBox, self, "Set Colors", self.setColors, tooltip = "Set the canvas background color, grid color and color palette for coloring continuous variables", debuggingEnabled = 0)
        self.SettingsTab.layout().addStretch(100)

        self.icons = self.createAttributeIconDict()

        dlg = self.createColorDialog()
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette("discPalette")
        self.graph.setCanvasBackground(dlg.getColor("Canvas"))
        self.graph.gridCurve.setPen(QPen(dlg.getColor("Grid")))

        #self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        self.cbShowAllAttributes()
        self.resize(750,700)

    # #####################
    def setSortCombo(self):
        items = [str(self.primaryAttrCombo.itemText(i)) for i in range(1, self.primaryAttrCombo.count())]
        attrs = self.graph.dataDomain and [attr.name for attr in self.graph.dataDomain]
        if items == attrs:
            return

        self.primaryAttrCombo.clear()
        self.secondaryAttrCombo.clear()
        self.primaryAttrCombo.addItem("(None)")
        self.secondaryAttrCombo.addItem("(None)")
        if not self.graph.haveData: return
        for attr in self.graph.dataDomain:
            self.primaryAttrCombo.addItem(self.icons[attr.varType], attr.name)
            self.secondaryAttrCombo.addItem(self.icons[attr.varType], attr.name)
        self.primaryAttribute = "(None)"
        self.secondaryAttribute = "(None)"

    def updateGraph(self, *args):
        self.graph.updateData(self.getShownAttributeList())


    # set combo box values with attributes that can be used for coloring the data
    def sortingClick(self):
        attrs = [self.primaryAttribute, self.secondaryAttribute]
        while "(None)" in attrs: attrs.remove("(None)")
        if attrs and self.data:
            self.data.sort(attrs)

        self.graph.setData(self.data, sortValuesForDiscreteAttrs = 0, skipIfSame = 0)
        self.updateGraph()


    # receive new data and update all fields
    def setData(self, data):
        if data:
            name = getattr(data, "name", "")
            data = data.filterref(orange.Filter_hasClassValue())
            data.name = name
            if len(data) == 0 or len(data.domain) == 0:        # if we don't have any examples or attributes then this is not a valid data set
                data = None
        if self.data and data and self.data.checksum() == data.checksum():
            return    # check if the new data set is the same as the old one

        self.closeContext()
        sameDomain = self.data and data and data.domain.checksum() == self.data.domain.checksum() # preserve attribute choice if the domain is the same
        self.data = data

        self.graph.setData(self.data)
        if self.graph.dataHasDiscreteClass:
            self.graph.discPalette.setNumberOfColors(len(self.graph.dataDomain.classVar.values))
        if not sameDomain:
            self.setShownAttributes(self.attributeSelectionList)
            self.resetAttrManipulation()
            self.setSortCombo()
        self.openContext("", self.data)
        self.resetAttrManipulation()
        self.updateGraph()

    ####### SELECTION signal ################################
    # receive info about which attributes to show
    def setShownAttributes(self, attributeSelectionList):
        self.attributeSelectionList = attributeSelectionList
        if self.graph.haveData and self.attributeSelectionList and 0 not in [self.graph.attributeNameIndex.has_key(attr) for attr in self.attributeSelectionList]:
            self.setShownAttributeList(self.attributeSelectionList)
        else:
            self.setShownAttributeList()

    # update attribute ordering
    def updateShownAttributeList(self):
        self.setShownAttributeList()
        self.updateGraph()

    def sendShownAttributes(self):
        self.send("Attribute Selection List", [a[0] for a in self.shownAttributes])

    # just tell the graph to hide the selected rectangle
    def enterEvent(self, e):
        if self.graph.selectedRectangle:
            self.graph.selectedRectangle.detach()
            self.graph.selectedRectangle = None
            self.graph.replot()

    def setGlobalValueScaling(self):
        self.graph.setData(self.data)
        self.updateGraph()

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.graph.contPalette = dlg.getContinuousPalette("contPalette")
            self.graph.discPalette = dlg.getDiscretePalette("discPalette")
            self.graph.setCanvasBackground(dlg.getColor("Canvas"))
            self.graph.gridCurve.setPen(QPen(dlg.getColor("Grid")))
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
        c.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return c

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSurveyPlot()
    ow.show()
    data = orange.ExampleTable(r"e:\Development\Orange Datasets\UCI\wine.tab")
    ow.setData(data)
    a.exec_()