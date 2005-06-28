"""
<name>Select Attributes</name>
<description>Manual selection of attributes.</description>
<icon>icons/SelectAttributes.png</icon>
<priority>1100</priority>
"""

from OWTools import *
from OWWidget import *
from OWGraph import *
import OWGUI

import sys


AVAILABLE_ATTR = 0
ATTRIBUTE  = 1
CLASS_ATTR = 2
META_ATTR  = 3

class OWDataDomain(OWWidget):

    ############################################################################################################################################################
    ## Class initialization ####################################################################################################################################
    ############################################################################################################################################################
    def __init__(self,parent = None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Data Domain") #initialize base class

        buttonWidth = 50
        applyButtonWidth = 101

        # set member variables
        self.data = None
        self.attributes = {}
        self.internalSelectionUpdateFlag = 0
        self.attributesButtonLeft = False
        self.classButtonLeft = False
        self.metaButtonLeft = False        
        self.receivedAttrList = None
        
        # set channels
        self.inputs = [("Examples", ExampleTable, self.onDataInput), ("Attribute Subset", AttributeList, self.onAttributeList)]
        self.outputs = [("Examples", ExampleTable), ("Classified Examples", ExampleTableWithClass)]

        # GUI
        self.mainArea.setFixedWidth(0)
        #self.buttonBackground.hide()
        ca=QFrame(self.controlArea)
        ca.adjustSize()
        gl=QGridLayout(ca,4,3,5)

        # available attributes
        boxAvail = QVGroupBox(ca)
        boxAvail.setTitle('Available Attributes')
        gl.addMultiCellWidget(boxAvail, 0,2,0,0)
        self.inputAttributesList = QListBox(boxAvail,'InputAttributes')
        self.inputAttributesList.setSelectionMode(QListBox.Extended)
        self.connect(self.inputAttributesList, SIGNAL('selectionChanged()'), self.onInputAttributesSelectionChange)
        self.connect(self.inputAttributesList, SIGNAL('currentChanged(QListBoxItem*)'), self.onInputAttributesCurrentChange)

        # attributes
        vbAttr = QVBox(ca)
        gl.addWidget(vbAttr, 0,1)
        self.attributesButtonUp = OWGUI.button(vbAttr, self, "Up", self.onAttributesButtonUpClick)
        self.attributesButtonUp.setMaximumWidth(buttonWidth)
        self.attributesButton = OWGUI.button(vbAttr, self, ">",self.onAttributesButtonClicked)        
        self.attributesButton.setMaximumWidth(buttonWidth)
        self.attributesButtonDown = OWGUI.button(vbAttr, self, "Down", self.onAttributesButtonDownClick)
        self.attributesButtonDown.setMaximumWidth(buttonWidth)
        boxAttr = QVGroupBox(ca)
        boxAttr.setTitle('Attributes')
        gl.addWidget(boxAttr, 0,2)
        self.attributesList = QListBox(boxAttr,'Attributes')
        self.attributesList.setSelectionMode(QListBox.Extended)
        self.connect(self.attributesList, SIGNAL('selectionChanged()'), self.onAttributesSelectionChange)
        self.connect(self.attributesList, SIGNAL('currentChanged(QListBoxItem*)'), self.onAttributesCurrentChange)
        self.connect(self.attributesList, SIGNAL('doubleClicked(QListBoxItem*)'), self.onAttributesDoubleClick)

        # class        
        self.classButton = OWGUI.button(ca, self, ">", self.onClassButtonClicked)
        self.classButton.setMaximumWidth(buttonWidth)
        gl.addWidget(self.classButton, 1,1)
        boxClass = QVGroupBox(ca)
        boxClass.setTitle('Class')
        boxClass.setFixedHeight(46)
        gl.addWidget(boxClass, 1,2)
        self.classList = QListBox(boxClass,'ClassAttribute')
        self.classList.setSelectionMode(QListBox.Extended)
        self.connect(self.classList, SIGNAL('selectionChanged()'), self.onClassSelectionChange)
        self.connect(self.classList, SIGNAL('currentChanged(QListBoxItem*)'), self.onClassCurrentChange)
        self.connect(self.classList, SIGNAL('doubleClicked(QListBoxItem*)'), self.onClassDoubleClick)
        
        # meta
        vbMeta = QVBox(ca)
        gl.addWidget(vbMeta, 2,1)
        self.metaButtonUp = OWGUI.button(vbMeta, self, "Up", self.onMetaButtonUpClick)
        self.metaButtonUp.setMaximumWidth(buttonWidth)
        self.metaButton = OWGUI.button(vbMeta, self, ">",self.onMetaButtonClicked)
        self.metaButton.setMaximumWidth(buttonWidth)
        self.metaButtonDown = OWGUI.button(vbMeta, self, "Down", self.onMetaButtonDownClick)
        self.metaButtonDown.setMaximumWidth(buttonWidth)
        boxMeta = QVGroupBox(ca)
        boxMeta.setTitle('Meta Attributes')
        gl.addWidget(boxMeta, 2,2)
        self.metaList = QListBox(boxMeta,'MetaAttributes')
        self.metaList.setSelectionMode(QListBox.Extended)
        self.connect(self.metaList, SIGNAL('selectionChanged()'), self.onMetaSelectionChange)
        self.connect(self.metaList, SIGNAL('currentChanged(QListBoxItem*)'), self.onMetaCurrentChange)         
        self.connect(self.metaList, SIGNAL('doubleClicked(QListBoxItem*)'), self.onMetaDoubleClick)
        
        # apply/reset buttons
        boxApply = QHBox(ca)
        gl.addMultiCellWidget(boxApply, 3,3,0,2)
        self.applyButton = OWGUI.button(boxApply, self, "Apply", callback = self.setOutput)
        self.applyButton.setEnabled(False)
        self.applyButton.setMaximumWidth(applyButtonWidth)
        self.resetButton = OWGUI.button(boxApply, self, "Reset", callback = self.reset)
        self.resetButton.setMaximumWidth(applyButtonWidth)

        # icons
        self.icons = self.createAttributeIconDict()

        self.resize(400,480)       


    ############################################################################################################################################################
    ## Data input and output management ########################################################################################################################
    ############################################################################################################################################################
    def onAttributeList(self, attrList):
        self.receivedAttrList = attrList
        self.onDataInput(self.data)
        

    def onDataInput(self, data):
        if self.data and data and self.data.checksum() == data.checksum(): return   # we received the same dataset again
        
        self.data = data
        self.attributes = {}
        
        if self.receivedAttrList:
            self.onAttributeList(self.receivedAttrList)
            return

        self.inputAttributesList.clear()
        self.attributesList.clear()
        self.classList.clear()
        self.metaList.clear()        

        if data and self.receivedAttrList:
            for attr in self.data.domain.attributes:
                self.attributes[attr.name] = (attr.name in self.receivedAttrList)
                if attr.name in self.receivedAttrList:      self.attributesList.insertItem(self.icons[attr.varType], attr.name)
                else:                                       self.inputAttributesList.insertItem(self.icons[attr.varType], attr.name)

            #set up class variable
            if self.data.domain.classVar and self.data.domain.classVar.name not in self.receivedAttrList:
                self.attributes[self.data.domain.classVar.name] = CLASS_ATTR
                self.classList.insertItem(self.icons[self.data.domain.classVar.varType], self.data.domain.classVar.name)

            #set up meta attributes
            for attr in self.data.domain.getmetas().values():
                self.attributes[attr.name] = META_ATTR
                if attr.name not in self.receivedAttrList:
                    self.metaList.insertItem(self.icons[attr.varType], attr.name)
                    
        elif data:
            #set up normal attributes
            for attr in data.domain.attributes:
                self.attributes[attr.name] = ATTRIBUTE
                self.attributesList.insertItem(self.icons[attr.varType], attr.name)

            #set up class variable
            if data and data.domain.classVar:
                self.attributes[data.domain.classVar.name] = CLASS_ATTR
                self.classList.insertItem(self.icons[data.domain.classVar.varType], data.domain.classVar.name)

            #set up meta attriutes
            for attr in data.domain.getmetas().values():
                self.attributes[attr.name] = META_ATTR
                self.metaList.insertItem(self.icons[attr.varType], attr.name)

            self.setInputAttributesListElements()

        self.setOutput()
        self.updateInterfaceState()
        
    def setOutput(self):
        if self.data:
            self.applyButton.setEnabled(False)
            attributes = [];
            for i in range(0, self.attributesList.count()):
                attributes.append(self.data.domain[str(self.attributesList.text(i))])
                
            #create domain without class attribute
            if self.classList.count()>0:
                domain = orange.Domain(attributes, self.data.domain[str(self.classList.text(0))],self.data.domain)
            else:
                domain = orange.Domain(attributes, None,self.data.domain)
            for i in range(0,self.metaList.count()):
                domain.addmeta(orange.newmetaid(), self.data.domain[str(self.metaList.text(i))])
            newdata = orange.ExampleTable(domain, self.data)
            newdata.name = self.data.name
            self.send("Examples", newdata)

            #create domaing with  class atribute
            if self.classList.count()>0:
                self.send("Classified Examples", newdata)
            else:
                self.send("Classified Examples", None)
        else:
            self.send("Examples", None)
            self.send("Classified Examples", None)
        
    def reset(self):
        data = self.data
        self.data = None
        self.onDataInput(data)

        
    ############################################################################################################################################################
    ## List box selection change handlers ######################################################################################################################
    ############################################################################################################################################################
        
    def onInputAttributesSelectionChange(self):
        self.handleListSelectionChange(self.inputAttributesList)
        
    def onInputAttributesCurrentChange(self, item):
        self.handleListSelectionChange(self.inputAttributesList)
            
    def onAttributesSelectionChange(self):
        self.handleListSelectionChange(self.attributesList)

    def onAttributesCurrentChange(self, item):
        self.handleListSelectionChange(self.attributesList)
        
    def onClassSelectionChange(self):
        self.handleListSelectionChange(self.classList)
        
    def onClassCurrentChange(self, item):
        self.handleListSelectionChange(self.classList)        
            
    def onMetaSelectionChange(self):
        self.handleListSelectionChange(self.metaList)

    def onMetaCurrentChange(self, item):
        self.handleListSelectionChange(self.metaList)


    def onAttributesDoubleClick(self, item):
        self.onAttributesButtonClicked()

    def onClassDoubleClick(self, item):
        self.onClassButtonClicked()

    def onMetaDoubleClick(self, item):
        self.onMetaButtonClicked()

    ############################################################################################################################################################
    ## Interface state management - updates interface elements based on selection in list boxes ################################################################
    ############################################################################################################################################################
            
    def updateInterfaceState(self):

        #set buttons for adding or removing attributes in lists
        self.attributesButton.setEnabled(False)
        self.metaButton.setEnabled(False)
        self.classButton.setEnabled(False)
        if (self.computeSelectionCount(self.inputAttributesList)>0):
            self.attributesButton.setText(">")
            self.attributesButton.setEnabled(True)
            self.attributesButtonLeft = False

            self.metaButton.setText(">")
            self.metaButton.setEnabled(True)
            self.metaButtonLeft = False
         
        if (self.computeSelectionCount(self.inputAttributesList)==1):
            text = self.getSelectedItemText(self.inputAttributesList)

            if (self.data.domain[text].varType<>orange.VarTypes.String):
                self.classButton.setText(">")
                self.classButton.setEnabled(True)
                self.classButtonLeft = False

        if (self.computeSelectionCount(self.attributesList)>0):
            self.attributesButton.setText("<")
            self.attributesButton.setEnabled(True)
            self.attributesButtonLeft = True

        if (self.computeSelectionCount(self.metaList)>0):
            self.metaButton.setText("<")
            self.metaButton.setEnabled(True)
            self.metaButtonLeft = True
            
        if (self.computeSelectionCount(self.classList)>0):
            self.classButton.setText("<")
            self.classButton.setEnabled(True)
            self.classButtonLeft = True

        #set buttons for moving selection up or down in list
        self.attributesButtonUp.setEnabled(False)
        self.attributesButtonDown.setEnabled(False)
        if (self.computeSelectionCount(self.attributesList)>0 and self.isConsecutiveSelection(self.attributesList)):
            if (not self.attributesList.isSelected(0)):
                self.attributesButtonUp.setEnabled(True)
            if (not self.attributesList.isSelected(self.attributesList.count()-1)):
                self.attributesButtonDown.setEnabled(True)

        self.metaButtonUp.setEnabled(False)
        self.metaButtonDown.setEnabled(False)  
        if (self.computeSelectionCount(self.metaList)>0 and self.isConsecutiveSelection(self.metaList)):
            if (not self.metaList.isSelected(0)):
                self.metaButtonUp.setEnabled(True)
            if (not self.metaList.isSelected(self.metaList.count()-1)):                
                self.metaButtonDown.setEnabled(True)
          

    ############################################################################################################################################################
    ## Button click handlers ###################################################################################################################################
    ############################################################################################################################################################
            
    def onAttributesButtonClicked(self):
        self.internalSelectionUpdateFlag = self.internalSelectionUpdateFlag + 1
        if self.attributesButtonLeft:
            self.removeSelectedItems(self.attributesList)
        else:
            for i in range(0, self.inputAttributesList.count()):
                if (self.inputAttributesList.isSelected(i)):
                    itemText = str(self.inputAttributesList.text(i))
                    itemType = self.data.domain[itemText].varType
                    self.attributesList.insertItem(self.icons[itemType],itemText)
                    self.attributes[itemText] = ATTRIBUTE

        self.inputAttributesList.clearSelection()                                
        self.attributesList.clearSelection()
        self.classList.clearSelection()
        self.metaList.clearSelection()
        
        self.internalSelectionUpdateFlag = self.internalSelectionUpdateFlag - 1
        self.setInputAttributesListElements()
        self.updateInterfaceState()

        self.applyButton.setEnabled(True)        

    def onClassButtonClicked(self):
        self.internalSelectionUpdateFlag = self.internalSelectionUpdateFlag + 1
        if self.classList.count() > 0: self.attributes[str(self.classList.text(0))] = AVAILABLE_ATTR
        self.classList.clear()
        if not self.classButtonLeft:
            for i in range(0, self.inputAttributesList.count()):
                if (self.inputAttributesList.isSelected(i)):
                    itemText = str(self.inputAttributesList.text(i))
                    itemType = self.data.domain[itemText].varType
                    self.classList.insertItem(self.icons[itemType],itemText)
                    self.attributes[itemText] = CLASS_ATTR

        self.inputAttributesList.clearSelection()                                
        self.attributesList.clearSelection()
        self.classList.clearSelection()
        self.metaList.clearSelection()
        
        self.internalSelectionUpdateFlag = self.internalSelectionUpdateFlag - 1
        self.setInputAttributesListElements()
        self.updateInterfaceState()

        self.applyButton.setEnabled(True)        

    def onMetaButtonClicked(self):
        self.internalSelectionUpdateFlag = self.internalSelectionUpdateFlag + 1
        if self.metaButtonLeft:
            self.removeSelectedItems(self.metaList)
        else:
            for i in range(0, self.inputAttributesList.count()):
                if (self.inputAttributesList.isSelected(i)):
                    itemText = str(self.inputAttributesList.text(i))
                    itemType = self.data.domain[itemText].varType
                    self.metaList.insertItem(self.icons[itemType],itemText)
                    self.attributes[itemText] = META_ATTR

        self.inputAttributesList.clearSelection()                                
        self.attributesList.clearSelection()
        self.classList.clearSelection()
        self.metaList.clearSelection()
        
        self.internalSelectionUpdateFlag = self.internalSelectionUpdateFlag - 1
        self.setInputAttributesListElements()        
        self.updateInterfaceState()

        self.applyButton.setEnabled(True)        

    def onMetaButtonUpClick(self):
        self.moveSelectionUp(self.metaList)
        self.applyButton.setEnabled(True)

    def onMetaButtonDownClick(self):
        self.moveSelectionDown(self.metaList)
        self.applyButton.setEnabled(True)
        
    def onAttributesButtonUpClick(self):
        self.moveSelectionUp(self.attributesList)
        self.applyButton.setEnabled(True)
        
    def onAttributesButtonDownClick(self):
        self.moveSelectionDown(self.attributesList)
        self.applyButton.setEnabled(True)
        
    ############################################################################################################################################################
    ## Utility functions #######################################################################################################################################
    ############################################################################################################################################################

    def computeSelectionCount(self, listBox):
        result = 0
        for i in range(0, listBox.count()):
            if (listBox.isSelected(i)):
                result = result + 1

        return result

    def getSelectedItemText(self, listBox):
        result = 0
        for i in range(0, listBox.count()):
            if (listBox.isSelected(i)):
                result = str(listBox.text(i))
                break
        return result

    def removeSelectedItems(self, listBox):
        for i in range(listBox.count(), -1, -1):
            if (listBox.isSelected(i)):
                self.attributes[str(listBox.text(i))] = AVAILABLE_ATTR
                listBox.removeItem(i)

    def isConsecutiveSelection(self, listBox):
        next = -1
        result = True
        for i in range(0, listBox.count()):
            if (listBox.isSelected(i)):
                if (i<>next and next<>-1):
                    result = False
                    break
                else:
                    next = i + 1
        return result

    def moveSelectionUp(self, listBox):
        selection = []
        for i in range(0, listBox.count()):
            if (listBox.isSelected(i)):
                selection.append(i)
        for i in selection:
            txt1 = str(listBox.text(i-1))
            txt2 = str(listBox.text(i))
            listBox.changeItem(self.icons[self.data.domain[txt1].varType], txt1, i)
            listBox.changeItem(self.icons[self.data.domain[txt2].varType], txt2, i-1)
        listBox.clearSelection()
        for i in selection:
            listBox.setSelected(i-1, True)
            
        
    def moveSelectionDown(self, listBox):
        selection = []
        for i in range(0, listBox.count()):
            if (listBox.isSelected(i)):
                selection.append(i)

        selection.reverse()        
        for i in selection:
            txt1 = str(listBox.text(i))
            txt2 = str(listBox.text(i+1))
            listBox.changeItem(self.icons[self.data.domain[txt1].varType], txt1, i+1)
            listBox.changeItem(self.icons[self.data.domain[txt2].varType], txt2, i)
        listBox.clearSelection()
        for i in selection:
            listBox.setSelected(i+1, True)

    def setInputAttributesListElements(self):
        self.internalSelectionUpdateFlag = self.internalSelectionUpdateFlag + 1
        self.inputAttributesList.clear()
        if self.data:
            #set up normal attributes + class
            for attr in self.data.domain:
                if self.attributes[attr.name] == AVAILABLE_ATTR: self.inputAttributesList.insertItem(self.icons[attr.varType], attr.name)

            #set up meta attributes
            for attr in self.data.domain.getmetas().values():
                if self.attributes[attr.name] == AVAILABLE_ATTR: self.inputAttributesList.insertItem(self.icons[attr.varType], attr.name)

        self.internalSelectionUpdateFlag = self.internalSelectionUpdateFlag - 1
        
    def handleListSelectionChange(self, listBox):
        if (self.internalSelectionUpdateFlag==0):
            self.internalSelectionUpdateFlag = self.internalSelectionUpdateFlag + 1
            if (self.inputAttributesList<>listBox):
                self.inputAttributesList.clearSelection()
            if (self.attributesList<>listBox):
                self.attributesList.clearSelection()
            if (self.classList<>listBox):
                self.classList.clearSelection()
                
            if (self.metaList<>listBox):
                self.metaList.clearSelection()

            self.updateInterfaceState()
            self.internalSelectionUpdateFlag = self.internalSelectionUpdateFlag - 1
            
if __name__=="__main__":
    data = orange.ExampleTable(r'..\..\doc\datasets\iris.tab')
    # add meta attribute
    data.domain.addmeta(orange.newmetaid(), orange.StringVariable("name"))
    for ex in data:
        ex["name"] = str(ex.getclass())

    a=QApplication(sys.argv)
    ow=OWDataDomain()
    a.setMainWidget(ow)
    ow.show()
    ow.onDataInput(data)
    a.exec_loop()
        
