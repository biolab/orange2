"""
<name>Data Domain</name>
<description>This widget obtains the data set on the input, and allows the user to redefine the domain
(select which attributes to use,select if and which attribute will be used as a class variable,
and which attributes will be used as meta attributes), and output a new data set with that domain.</description>
<icon>icons/SelectAttributes.png</icon>
<priority>20</priority>
"""

from OWTools import *
from OWWidget import *
from OWGraph import *
import OWGUI

import sys

class OWDataDomain(OWWidget):

    ############################################################################################################################################################
    ## Class initialization ####################################################################################################################################
    ############################################################################################################################################################
    def __init__(self,parent = None):
        OWWidget.__init__(self, parent, "Data Domain") #initialize base class

        buttonSize = QSize(40, 30)
        upDownButtonSize = QSize(37,30)
        
        # set member variables
        self.data = None
        self.internalSelectionUpdateFlag = 0
        self.attributesButtonLeft = False
        self.classButtonLeft = False
        self.metaButtonLeft = False        
        
        # set channels
        self.inputs = [("InputData", ExampleTable, self.onDataInput)]
        self.outputs = [("OutputData", ExampleTable), ("OutputDataWithClass", ExampleTableWithClass)]

        self.space.setMinimumSize(QSize(650,500))
        self.hbox = QHBox(self.space)
        self.hbox.setSpacing(10)

        self.vbox1 = QVBox(self.hbox)
        self.vbox1.setSpacing(10)
        
        self.vframe2 = QFrame(self.hbox)
        self.vframe2Layout = QGridLayout(self.vframe2, 9, 1, 0, 0)
        self.vframe3 = QFrame(self.hbox)
        self.vframe3Layout = QBoxLayout(self.vframe3,QBoxLayout.TopToBottom,10,5)

        #set up leftmost column
        self.inputAttributesNameBox = QVGroupBox(self.vbox1)
        self.inputAttributesNameBox.setTitle('Available Attributes')
        self.inputAttributesNameBox.setMinimumSize(QSize(200,100))
        self.inputAttributesNameBox.setMaximumSize(QSize(100,1000))        
        self.inputAttributesList = QListBox(self.inputAttributesNameBox,'InputAttributes')
        self.inputAttributesList.setSelectionMode(QListBox.Extended)
        self.connect(self.inputAttributesList, SIGNAL('selectionChanged()'), self.onInputAttributesSelectionChange)
        self.connect(self.inputAttributesList, SIGNAL('currentChanged(QListBoxItem*)'), self.onInputAttributesCurrentChange)


        #set up middle column
        self.horizontalAttributesFrame = QFrame(self.vframe2)
        self.vframe2Layout.addMultiCellWidget(self.horizontalAttributesFrame,0,3, 0,0,0)
        self.horizontalAttributesFrameLayout = QHBoxLayout(self.horizontalAttributesFrame, 5, 5)
        
        self.attributesButton = OWGUI.button(self.horizontalAttributesFrame, self, ">",self.onAttributesButtonClicked)        
        self.attributesButton.setMaximumSize(buttonSize)
        self.horizontalAttributesFrameLayout.addWidget(self.attributesButton, 0, Qt.AlignLeft)
        self.attributesNameBox = QVGroupBox(self.horizontalAttributesFrame)
        self.attributesNameBox.setTitle('Attributes')
        self.attributesNameBox.setMinimumSize(QSize(200,200))        
        self.horizontalAttributesFrameLayout.addWidget(self.attributesNameBox, 0, Qt.AlignLeft)        
        self.attributesList = QListBox(self.attributesNameBox,'Attributes')
        self.attributesList.setSelectionMode(QListBox.Extended)
        self.connect(self.attributesList, SIGNAL('selectionChanged()'), self.onAttributesSelectionChange)
        self.connect(self.attributesList, SIGNAL('currentChanged(QListBoxItem*)'), self.onAttributesCurrentChange)                

        self.verticalAttributeFrame = QFrame(self.horizontalAttributesFrame)
        self.horizontalAttributesFrameLayout.addWidget(self.verticalAttributeFrame, 0, Qt.AlignLeft)        
        self.verticalAttributeFrameLayout = QVBoxLayout(self.verticalAttributeFrame,5,5)
        self.attributesButtonUp = OWGUI.button(self.verticalAttributeFrame, self, "Up", self.onAttributesButtonUpClick)
        self.attributesButtonUp.setMaximumSize(upDownButtonSize)        
        self.verticalAttributeFrameLayout.addWidget(self.attributesButtonUp, 0, Qt.AlignTop)
        self.attributesButtonDown = OWGUI.button(self.verticalAttributeFrame, self, "Down", self.onAttributesButtonDownClick)
        self.attributesButtonDown.setMaximumSize(upDownButtonSize)
        self.verticalAttributeFrameLayout.addWidget(self.attributesButtonDown, 0, Qt.AlignTop)
        self.verticalAttributeFrameLayout.addStretch(100)

        self.horizontalAttributesFrameLayout.addStretch(100)             
        
        self.horizontalClassFrame = QFrame(self.vframe2)
        self.vframe2Layout.addMultiCellWidget(self.horizontalClassFrame,4,5,0,0,0)
        self.horizontalClassFrameLayout = QHBoxLayout(self.horizontalClassFrame, 5, 5)  
        self.classButton = OWGUI.button(self.horizontalClassFrame, self, ">", self.onClassButtonClicked)
        self.classButton.setMaximumSize(buttonSize)
        self.horizontalClassFrameLayout.addWidget(self.classButton, 0, Qt.AlignLeft)                
        self.classNameBox = QVGroupBox(self.horizontalClassFrame)
        self.classNameBox.setTitle('Class')
        self.classNameBox.setMinimumSize(QSize(200,50))        
        self.classNameBox.setMaximumSize(QSize(200,50))
        self.horizontalClassFrameLayout.addWidget(self.classNameBox, 0, Qt.AlignLeft)           
        self.classList = QListBox(self.classNameBox,'ClassAttribute')  
        self.classList.setSelectionMode(QListBox.Extended)
        self.connect(self.classList, SIGNAL('selectionChanged()'), self.onClassSelectionChange)
        self.connect(self.classList, SIGNAL('currentChanged(QListBoxItem*)'), self.onClassCurrentChange)

        self.horizontalClassFrameLayout.addStretch(200)

        self.horizontalMetaFrame = QFrame(self.vframe2)
        self.vframe2Layout.addMultiCellWidget(self.horizontalMetaFrame,6,9,0,0,0)          
        self.horizontalMetaFrameLayout = QHBoxLayout(self.horizontalMetaFrame, 5, 5)
        
        self.metaButton = OWGUI.button(self.horizontalMetaFrame, self, ">",self.onMetaButtonClicked)
        self.metaButton.setMaximumSize(buttonSize)
        self.horizontalMetaFrameLayout.addWidget(self.metaButton, 0, Qt.AlignLeft)        
        self.metaNameBox = QVGroupBox(self.horizontalMetaFrame)
        self.metaNameBox.setTitle('Meta Attributes')
        self.metaNameBox.setMinimumSize(QSize(200,200))
        self.horizontalMetaFrameLayout.addWidget(self.metaNameBox, 0, Qt.AlignLeft)          
        self.metaList = QListBox(self.metaNameBox,'MetaAttributes')
        self.metaList.setSelectionMode(QListBox.Extended)
        self.connect(self.metaList, SIGNAL('selectionChanged()'), self.onMetaSelectionChange)
        self.connect(self.metaList, SIGNAL('currentChanged(QListBoxItem*)'), self.onMetaCurrentChange)         
        
        self.verticalMetaFrame = QFrame(self.horizontalMetaFrame)
        self.horizontalMetaFrameLayout.addWidget(self.verticalMetaFrame, 0, Qt.AlignLeft)        
        self.verticalMetaFrameLayout = QVBoxLayout(self.verticalMetaFrame,5,5)
        self.metaButtonUp = OWGUI.button(self.verticalMetaFrame, self, "Up", self.onMetaButtonUpClick)
        self.metaButtonUp.setMaximumSize(upDownButtonSize)        
        self.verticalMetaFrameLayout.addWidget(self.metaButtonUp, 0, Qt.AlignTop)
        self.metaButtonDown = OWGUI.button(self.verticalMetaFrame, self, "Down", self.onMetaButtonDownClick)
        self.metaButtonDown.setMaximumSize(upDownButtonSize)      
        self.verticalMetaFrameLayout.addWidget(self.metaButtonDown, 0, Qt.AlignTop)
        self.verticalMetaFrameLayout.addStretch(50)
        self.horizontalMetaFrameLayout.addStretch(200)        
        
        #set up rightmost column
        self.applyButton = OWGUI.button(self.vframe3, self, "Apply", callback = self.setOutput)
        self.applyButton.setEnabled(False)        
        self.vframe3Layout.addWidget(self.applyButton,0,Qt.AlignTop)
        self.resetButton = OWGUI.button(self.vframe3, self, "Reset", callback = self.reset)
        self.vframe3Layout.addWidget(self.resetButton,0,Qt.AlignTop)

        self.vframe3Layout.addStretch(200)
        
    ############################################################################################################################################################
    ## Data input and output management ########################################################################################################################
    ############################################################################################################################################################

    def onDataInput(self, data):
        self.data = data

        self.inputAttributesList.clear()
        self.attributesList.clear()
        self.classList.clear()
        self.metaList.clear()        

        if data:
            #set up normal attributes
            for attr in data.domain.attributes:
                self.attributesList.insertItem(self.createListItem(attr.varType, attr.name))

            #set up class variable
            if data and data.domain.classVar:
                self.classList.insertItem(self.createListItem(data.domain.classVar.varType, data.domain.classVar.name))

            #set up meta attriutes
            for attr in data.domain.getmetas().values():
                self.metaList.insertItem(self.createListItem(attr.varType, attr.name))

            self.setInputAttributesListElements()
            self.setOutput()

        self.updateInterfaceState()
        
    def setOutput(self):
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
        self.send("OutputData", newdata)

        #create domaing with  class atribute
        if self.classList.count()>0:
            self.send("OutputDataWithClass", newdata)
        else:
            self.send("OutputDataWithClass", None)
        
    def reset(self):
        self.onDataInput(self.data)

        
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
                    self.attributesList.insertItem(self.createListItem(itemType,itemText))

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
        self.classList.clear()
        if not self.classButtonLeft:
            for i in range(0, self.inputAttributesList.count()):
                if (self.inputAttributesList.isSelected(i)):
                    itemText = str(self.inputAttributesList.text(i))
                    itemType = self.data.domain[itemText].varType
                    self.classList.insertItem(self.createListItem(itemType,itemText))

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
                    self.metaList.insertItem(self.createListItem(itemType,itemText))

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
        
    def createListItem(self, varType, text):
            marks = {}
            marks[orange.VarTypes.Continuous] = 'C'
            marks[orange.VarTypes.Discrete] = 'D'
            marks[orange.VarTypes.String] = 'S'


            pixmap = QPixmap()
            pixmap.resize(13,13)

            painter = QPainter()
            painter.begin(pixmap)


            painter.setPen( Qt.black );
            painter.setBrush( Qt.white );
            painter.drawRect( 0, 0, 13, 13 );
            painter.drawText(3, 11, marks[varType])

            painter.end()
            
            listItem= QListBoxPixmap(pixmap)
            listItem.setText(text)

            return listItem

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
            item1 = self.createListItem(self.data.domain[str(listBox.text(i-1))].varType, str(listBox.text(i-1)))
            item2 = self.createListItem(self.data.domain[str(listBox.text(i))].varType, str(listBox.text(i)))
            listBox.changeItem(item1, i)
            listBox.changeItem(item2, i-1)
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
            item1 = self.createListItem(self.data.domain[str(listBox.text(i))].varType, str(listBox.text(i)))
            item2 = self.createListItem(self.data.domain[str(listBox.text(i+1))].varType, str(listBox.text(i+1)))
            listBox.changeItem(item1, i+1)
            listBox.changeItem(item2, i)
        listBox.clearSelection()
        for i in selection:
            listBox.setSelected(i+1, True)

    def setInputAttributesListElements(self):
        self.internalSelectionUpdateFlag = self.internalSelectionUpdateFlag + 1
        self.inputAttributesList.clear()
        if self.data:
            #set up normal attributes
            for attr in self.data.domain.attributes:
                self.inputAttributesList.insertItem(self.createListItem(attr.varType, attr.name))

            #set up class variable
            if self.data.domain.classVar:
                self.inputAttributesList.insertItem(self.createListItem(self.data.domain.classVar.varType, self.data.domain.classVar.name))

            #set up meta attriutes
            for attr in self.data.domain.getmetas().values():
                self.inputAttributesList.insertItem(self.createListItem(attr.varType, attr.name))

        for i in range(self.inputAttributesList.count(), -1, -1):
            item = self.attributesList.findItem(str(self.inputAttributesList.text(i)))
            if item:
                self.inputAttributesList.removeItem(i)

        for i in range(self.inputAttributesList.count(), -1, -1):
            item = self.classList.findItem(str(self.inputAttributesList.text(i)))
            if item:
                self.inputAttributesList.removeItem(i)

        for i in range(self.inputAttributesList.count(), -1, -1):
            item = self.metaList.findItem(str(self.inputAttributesList.text(i)))
            if item:
                self.inputAttributesList.removeItem(i)

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
    data = orange.ExampleTable('iris.tab')
    
    a=QApplication(sys.argv)
    ow=OWDataDomain()
    a.setMainWidget(ow)
    ow.show()
    ow.onDataInput(data)
    a.exec_loop()
        
