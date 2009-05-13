"""
<name>Select Attributes</name>
<description>Manual selection of attributes.</description>
<icon>icons/SelectAttributes.png</icon>
<priority>1100</priority>
<contact>Peter Juvan (peter.juvan@fri.uni-lj.si)</contact>
"""
from OWWidget import *
from OWGraph import *
import OWGUI, OWGUIEx

class OWDataDomain(OWWidget):
    contextHandlers = {"": DomainContextHandler("", [ContextField("chosenAttributes", 
                                                                  DomainContextHandler.RequiredList + DomainContextHandler.IncludeMetaAttributes, 
                                                                  selected="selectedChosen", reservoir="inputAttributes"), 
                                                     ContextField("classAttribute", 
                                                                  DomainContextHandler.RequiredList + DomainContextHandler.IncludeMetaAttributes, 
                                                                  selected="selectedClass", reservoir="inputAttributes"), 
                                                     ContextField("metaAttributes", 
                                                                  DomainContextHandler.RequiredList + DomainContextHandler.IncludeMetaAttributes, 
                                                                  selected="selectedMeta", reservoir="inputAttributes")
                                                     ], syncWithGlobal = False)}


    def __init__(self, parent = None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Data Domain", wantMainArea = 0) 

        self.inputs = [("Examples", ExampleTable, self.setData), ("Attribute Subset", AttributeList, self.setAttributeList)]
        self.outputs = [("Examples", ExampleTable)]

        buttonWidth = 50
        applyButtonWidth = 101

        self.data = None
        self.receivedAttrList = None

        self.selectedInput = []
        self.inputAttributes = []
        self.selectedChosen = []
        self.chosenAttributes = []
        self.selectedClass = []
        self.classAttribute = []
        self.metaAttributes = []
        self.selectedMeta = []

        self.loadSettings()

        import sip
        sip.delete(self.controlArea.layout())
        grid = QGridLayout()
        self.controlArea.setLayout(grid)
        grid.setMargin(0)
                
        boxAvail = OWGUI.widgetBox(self, 'Available Attributes', addToLayout = 0)
        grid.addWidget(boxAvail, 0, 0, 3, 1)

        self.filterInputAttrs = OWGUIEx.lineEditFilter(boxAvail, self, None, useRE = 1, emptyText = "filter attributes...", callback = self.setInputAttributes, caseSensitive = 0)
        self.inputAttributesList = OWGUI.listBox(boxAvail, self, "selectedInput", "inputAttributes", callback = self.onSelectionChange, selectionMode = QListWidget.ExtendedSelection, enableDragDrop = 1, dragDropCallback = self.updateInterfaceAndApplyButton)
        self.filterInputAttrs.listbox = self.inputAttributesList 

        vbAttr = OWGUI.widgetBox(self, addToLayout = 0)
        grid.addWidget(vbAttr, 0, 1)
        self.attributesButtonUp = OWGUI.button(vbAttr, self, "Up", self.onAttributesButtonUpClick)
        self.attributesButtonUp.setMaximumWidth(buttonWidth)
        self.attributesButton = OWGUI.button(vbAttr, self, ">", self.onAttributesButtonClicked)
        self.attributesButton.setMaximumWidth(buttonWidth)
        self.attributesButtonDown = OWGUI.button(vbAttr, self, "Down", self.onAttributesButtonDownClick)
        self.attributesButtonDown.setMaximumWidth(buttonWidth)

        boxAttr = OWGUI.widgetBox(self, 'Attributes', addToLayout = 0)
        grid.addWidget(boxAttr, 0, 2)
        self.attributesList = OWGUI.listBox(boxAttr, self, "selectedChosen", "chosenAttributes", callback = self.onSelectionChange, selectionMode = QListWidget.ExtendedSelection, enableDragDrop = 1, dragDropCallback = self.updateInterfaceAndApplyButton)

        self.classButton = OWGUI.button(self, self, ">", self.onClassButtonClicked, addToLayout = 0)
        self.classButton.setMaximumWidth(buttonWidth)
        grid.addWidget(self.classButton, 1, 1)
        boxClass = OWGUI.widgetBox(self, 'Class', addToLayout = 0)
        boxClass.setFixedHeight(55)
        grid.addWidget(boxClass, 1, 2)
        self.classList = OWGUI.listBox(boxClass, self, "selectedClass", "classAttribute", callback = self.onSelectionChange, selectionMode = QListWidget.ExtendedSelection, enableDragDrop = 1, dragDropCallback = self.updateInterfaceAndApplyButton, dataValidityCallback = self.dataValidityCallback)

        vbMeta = OWGUI.widgetBox(self, addToLayout = 0)
        grid.addWidget(vbMeta, 2, 1)
        self.metaButtonUp = OWGUI.button(vbMeta, self, "Up", self.onMetaButtonUpClick)
        self.metaButtonUp.setMaximumWidth(buttonWidth)
        self.metaButton = OWGUI.button(vbMeta, self, ">", self.onMetaButtonClicked)
        self.metaButton.setMaximumWidth(buttonWidth)
        self.metaButtonDown = OWGUI.button(vbMeta, self, "Down", self.onMetaButtonDownClick)
        self.metaButtonDown.setMaximumWidth(buttonWidth)
        boxMeta = OWGUI.widgetBox(self, 'Meta Attributes', addToLayout = 0)
        grid.addWidget(boxMeta, 2, 2)
        self.metaList = OWGUI.listBox(boxMeta, self, "selectedMeta", "metaAttributes", callback = self.onSelectionChange, selectionMode = QListWidget.ExtendedSelection, enableDragDrop = 1, dragDropCallback = self.updateInterfaceAndApplyButton)

        boxApply = OWGUI.widgetBox(self, addToLayout = 0, orientation = "horizontal", addSpace = 1)
        grid.addWidget(boxApply, 3, 0, 3, 3)
        self.applyButton = OWGUI.button(boxApply, self, "Apply", callback = self.setOutput)
        self.applyButton.setEnabled(False)
        self.applyButton.setMaximumWidth(applyButtonWidth)
        self.resetButton = OWGUI.button(boxApply, self, "Reset", callback = self.reset)
        self.resetButton.setMaximumWidth(applyButtonWidth)
        
        grid.setRowStretch(0, 4)
        grid.setRowStretch(1, 0)
        grid.setRowStretch(2, 2)

        self.icons = self.createAttributeIconDict()

        self.inChange = False
        self.resize(400, 480)


    def setAttributeList(self, attrList):
        self.receivedAttrList = attrList
        self.setData(self.data)

    def onSelectionChange(self):
        if not self.inChange:
            self.inChange = True
            for lb, co in [(self.inputAttributesList, "selectedInput"), 
                       (self.attributesList, "selectedChosen"), 
                       (self.classList, "selectedClass"), 
                       (self.metaList, "selectedMeta")]:
                if not lb.hasFocus():
                    setattr(self, co, [])
            self.inChange = False

        self.updateInterfaceState()


    def setData(self, data):
        if self.data and data and self.data.checksum() == data.checksum():
            return   # we received the same dataset again

        self.closeContext()

        self.data = data
        self.attributes = {}
        self.filterInputAttrs.setText("")

        if data:
            domain = data.domain

            if domain.classVar:
                self.classAttribute = [(domain.classVar.name, domain.classVar.varType)]
            else:
                self.classAttribute = []
            self.metaAttributes = [(a.name, a.varType) for a in domain.getmetas().values()]

            if self.receivedAttrList:
                self.chosenAttributes = [(a.name, a.varType) for a in self.receivedAttrList]
                cas = set(chosenAttributes)
                self.inputAttributes = [(a.name, a.varType) for a in domain.attributes if (a.name, a.varType) not in cas]
            else:
                self.chosenAttributes = [(a.name, a.varType) for a in domain.attributes]
                self.inputAttributes = []

            metaIds = domain.getmetas().keys()
            metaIds.sort()
            self.allAttributes = [(attr.name, attr.varType) for attr in domain] + [(domain[i].name, domain[i].varType) for i in metaIds]
        else:
            self.inputAttributes = []
            self.chosenAttributes = []
            self.classAttribute = []
            self.metaAttributes = []
            self.allAttributes = []

        self.openContext("", data)

        self.usedAttributes = set(self.chosenAttributes + self.classAttribute + self.metaAttributes)
        self.setInputAttributes()

        self.setOutput()
        self.updateInterfaceState()


    def setOutput(self):
        if self.data:
            self.applyButton.setEnabled(False)
            
            attributes = [self.data.domain[x[0]] for x in self.chosenAttributes]
            classVar = self.classAttribute and self.data.domain[self.classAttribute[0][0]] or None
            domain = orange.Domain(attributes, classVar)
            for meta in self.metaAttributes:
                domain.addmeta(orange.newmetaid(), self.data.domain[meta[0]])

            newdata = orange.ExampleTable(domain, self.data, filterMetas=1)
            newdata.name = self.data.name
            self.outdataReport = self.prepareDataReport(newdata)
            self.send("Examples", newdata)
        else:
            self.send("Examples", None)

    def sendReport(self):
        self.reportData(self.data, "Input data")
        self.reportData(self.outdataReport, "Output data")
        if len(self.allAttributes) != len(self.usedAttributes):
            removed = set.difference(set(self.allAttributes), self.usedAttributes)
            self.reportSettings("", [("Removed", "%i (%s)" % (len(removed), ", ".join(x[0] for x in removed)))])

    def reset(self):
        data = self.data
        self.data = None
        self.setData(data)


    def disableButtons(self, *arg):
        for b in arg:
            b.setEnabled(False)

    def setButton(self, button, dir):
        button.setText(dir)
        button.setEnabled(True)

    # this callback is called when the user is dragging some listbox item(s) over class listbox
    # and we need to check if the data contains a single item or more. if more, reject the data
    def dataValidityCallback(self, ev):
        ev.ignore()     # by default we will not accept items
        if ev.mimeData().hasText() and self.classAttribute == []:
            try:
                selectedItemIndices = eval(str(ev.mimeData().text()))
                if type(selectedItemIndices) == list and len(selectedItemIndices) == 1:
                    ev.accept()
                else:
                    ev.ignore()
            except:
                pass
            
    # this is called when we have dropped some items into a listbox using drag and drop and we have to update interface
    def updateInterfaceAndApplyButton(self):
        self.usedAttributes = set(self.chosenAttributes + self.classAttribute + self.metaAttributes)    # we used drag and drop so we have to compute which attributes are now used
        self.setInputAttributes()
        self.updateInterfaceState()
        self.applyButton.setEnabled(True)
        

    def updateInterfaceState(self):
        if self.selectedInput:
            self.setButton(self.attributesButton, ">")
            self.setButton(self.metaButton, ">")
            self.disableButtons(self.attributesButtonUp, self.attributesButtonDown, self.metaButtonUp, self.metaButtonDown)

            if len(self.selectedInput) == 1 and self.inputAttributes[self.selectedInput[0]][1] in [orange.VarTypes.Discrete, orange.VarTypes.Continuous]:
                self.setButton(self.classButton, ">")
            else:
                self.classButton.setEnabled(False)

        elif self.selectedChosen:
            self.setButton(self.attributesButton, "<")
            self.disableButtons(self.classButton, self.metaButton, self.metaButtonUp, self.metaButtonDown)

            mini, maxi = min(self.selectedChosen), max(self.selectedChosen)
            cons = maxi - mini == len(self.selectedChosen) - 1
            self.attributesButtonUp.setEnabled(cons and mini)
            self.attributesButtonDown.setEnabled(cons and maxi < len(self.chosenAttributes)-1)

        elif self.selectedClass:
            self.setButton(self.classButton, "<")
            self.disableButtons(self.attributesButtonUp, self.attributesButtonDown, self.metaButtonUp, self.metaButtonDown, 
                                self.attributesButton, self.metaButton)

        elif self.selectedMeta:
            self.setButton(self.metaButton, "<")
            self.disableButtons(self.attributesButton, self.classButton, self.attributesButtonDown, self.attributesButtonUp)

            mini, maxi, leni = min(self.selectedMeta), max(self.selectedMeta), len(self.selectedMeta)
            cons = maxi - mini == leni - 1
            self.metaButtonUp.setEnabled(cons and mini)
            self.metaButtonDown.setEnabled(cons and maxi < len(self.metaAttributes)-1)

        else:
            self.disableButtons(self.attributesButtonUp, self.attributesButtonDown, self.metaButtonUp, self.metaButtonDown, 
                                self.attributesButton, self.metaButton, self.classButton)


    def splitSelection(self, alist, selected):
        selected.sort()

        i, sele = 0, selected[0]
        selList, restList = [], []
        for j, attr in enumerate(alist):
            if j == sele:
                selList.append(attr)
                i += 1
                sele = i<len(selected) and selected[i] or None
            else:
                restList.append(attr)
        return selList, restList

    def setInputAttributes(self):
        self.selectedInput = []
        if self.data:
            self.inputAttributes = filter(lambda x:x not in self.usedAttributes, self.allAttributes)
        else:
            self.inputAttributes = []
        #print "before: ", self.inputAttributes
        self.filterInputAttrs.setAllListItems()
        self.filterInputAttrs.updateListBoxItems(callCallback = 0)
        if self.data and self.inputAttributesList.count() != len(self.inputAttributes):       # the user has entered a filter - we have to remove some inputAttributes
            itemsText = [str(self.inputAttributesList.item(i).text()) for i in range(self.inputAttributesList.count())]
            self.inputAttributes = [ (item, self.data.domain[item].varType) for item in itemsText]
        #print "after: ", self.inputAttributes
        self.updateInterfaceState()

    def removeFromUsed(self, attributes):
        for attr in attributes:
            self.usedAttributes.remove(attr)
        self.setInputAttributes()

    def addToUsed(self, attributes):
        self.usedAttributes.update(attributes)
        self.setInputAttributes()


    def onAttributesButtonClicked(self):
        if self.selectedInput:
            selList, restList = self.splitSelection(self.inputAttributes, self.selectedInput)
            self.chosenAttributes = self.chosenAttributes + selList
            self.addToUsed(selList)
        else:
            selList, restList = self.splitSelection(self.chosenAttributes, self.selectedChosen)
            self.chosenAttributes = restList
            self.removeFromUsed(selList)

        self.updateInterfaceState()
        self.applyButton.setEnabled(True)


    def onClassButtonClicked(self):
        if self.selectedInput:
            selected = self.inputAttributes[self.selectedInput[0]]
            if self.classAttribute:
                self.removeFromUsed(self.classAttribute)
            self.addToUsed([selected])
            self.classAttribute = [selected]
        else:
            self.removeFromUsed(self.classAttribute)
            self.selectedClass = []
            self.classAttribute = []

        self.updateInterfaceState()
        self.applyButton.setEnabled(True)


    def onMetaButtonClicked(self):
        if self.selectedInput:
            selList, restList = self.splitSelection(self.inputAttributes, self.selectedInput)
            self.metaAttributes = self.metaAttributes + selList
            self.addToUsed(selList)
        else:
            selList, restList = self.splitSelection(self.metaAttributes, self.selectedMeta)
            self.metaAttributes = restList
            self.removeFromUsed(selList)

        self.updateInterfaceState()
        self.applyButton.setEnabled(True)


    def moveSelection(self, labels, selection, dir):
        labs = getattr(self, labels)
        sel = getattr(self, selection)
        mini, maxi = min(sel), max(sel)+1
        if dir == -1:
            setattr(self, labels, labs[:mini-1] + labs[mini:maxi] + [labs[mini-1]] + labs[maxi:])
        else:
            setattr(self, labels, labs[:mini] + [labs[maxi]] + labs[mini:maxi] + labs[maxi+1:])
        setattr(self, selection, map(lambda x:x+dir, sel))
        self.updateInterfaceState()
        self.applyButton.setEnabled(True)

    def onMetaButtonUpClick(self):
        self.moveSelection("metaAttributes", "selectedMeta", -1)

    def onMetaButtonDownClick(self):
        self.moveSelection("metaAttributes", "selectedMeta", 1)

    def onAttributesButtonUpClick(self):
        self.moveSelection("chosenAttributes", "selectedChosen", -1)

    def onAttributesButtonDownClick(self):
        self.moveSelection("chosenAttributes", "selectedChosen", 1)


if __name__=="__main__":
    import sys
    #data = orange.ExampleTable(r'../../doc/datasets/iris.tab')
    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\iris.tab")
    # add meta attribute
    data.domain.addmeta(orange.newmetaid(), orange.StringVariable("name"))
    for ex in data:
        ex["name"] = str(ex.getclass())

    a=QApplication(sys.argv)
    ow=OWDataDomain()
    ow.show()
    ow.setData(data)
    a.exec_()
    ow.saveSettings()
