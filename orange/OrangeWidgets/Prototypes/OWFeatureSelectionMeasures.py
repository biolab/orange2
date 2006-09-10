"""
<name>Feature Selection Measures</name>
<description>Manual selection of attributes.</description>
<icon>icons/SelectAttributes.png</icon>
<priority>1100</priority>
"""

from OWTools import *
from OWWidget import *
from OWGraph import *
import OWGUI
from orngTextCorpus import FeatureSelection, checkFromText

class OWFeatureSelectionMeasures(OWWidget):
##    contextHandlers = {"": DomainContextHandler("", [ContextField("chosenAttributes",
##                                                                  DomainContextHandler.RequiredList,
##                                                                  selected="selectedChosen", reservoir="inputAttributes")
##                                                     ])}


    def __init__(self,parent = None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Data Domain") #initialize base class

        self.inputs = [("Examples", ExampleTable, self.onDataInput)]
        self.outputs = [("Examples", ExampleTable)]

        buttonWidth = 50
        applyButtonWidth = 101

        self.data = None

        self.selectedInput = []
        self.inputAttributes = []        
        self.selectedChosen = []
        self.chosenAttributes = []

        self.loadSettings()
        
        self.mainArea.setFixedWidth(0)
        ca = QFrame(self.controlArea)
        ca.adjustSize()
        gl=QGridLayout(ca,2,3,5)

        boxAvail = QVGroupBox(ca)
        boxAvail.setTitle('Available Measures')
        gl.addWidget(boxAvail, 0,0)

        self.inputAttributesList = OWGUI.listBox(boxAvail, self, "selectedInput", "inputAttributes", callback = self.onSelectionChange, selectionMode = QListBox.Extended)

        vbAttr = QVBox(ca)
        gl.addWidget(vbAttr, 0,1)
        self.attributesButtonUp = OWGUI.button(vbAttr, self, "Up", self.onAttributesButtonUpClick)
        self.attributesButtonUp.setMaximumWidth(buttonWidth)
        self.attributesButton = OWGUI.button(vbAttr, self, ">",self.onAttributesButtonClicked)        
        self.attributesButton.setMaximumWidth(buttonWidth)
        self.attributesButtonDown = OWGUI.button(vbAttr, self, "Down", self.onAttributesButtonDownClick)
        self.attributesButtonDown.setMaximumWidth(buttonWidth)
        
        boxAttr = QVGroupBox(ca)
        boxAttr.setTitle('Measures')
        gl.addWidget(boxAttr, 0,2)
        self.attributesList = OWGUI.listBox(boxAttr, self, "selectedChosen", "chosenAttributes", callback = self.onSelectionChange, selectionMode = QListBox.Extended)
       
        boxApply = QHBox(ca)
        gl.addMultiCellWidget(boxApply, 1,1,0,2)
        self.applyButton = OWGUI.button(boxApply, self, "Apply", callback = self.setOutput)
        self.applyButton.setEnabled(False)
        self.applyButton.setMaximumWidth(applyButtonWidth)
        self.resetButton = OWGUI.button(boxApply, self, "Reset", callback = self.reset)
        self.resetButton.setMaximumWidth(applyButtonWidth)

        self.icons = self.createAttributeIconDict()

        self.inChange = False
        self.resize(400,480)       



    def onSelectionChange(self):
        if not self.inChange:
            self.inChange = True
            for lb, co in [(self.inputAttributesList, "selectedInput"),
                       (self.attributesList, "selectedChosen")]:
                if not lb.hasFocus():
                    setattr(self, co, [])
            self.inChange = False

        self.updateInterfaceState()            


    def onDataInput(self, data):
        if self.data and data and self.data.checksum() == data.checksum():
            return   # we received the same dataset again

##        self.closeContext()
        
        self.data = data
        self.attributes = {}
        
        if data:
            if not checkFromText(data):
                self.onDataInput(None)
                return

            self.chosenAttributes = []
            self.inputAttributes = FeatureSelection.measures.keys()[:]
                
            self.allAttributes = FeatureSelection.measures.keys()[:]
        else:
            self.inputAttributes = []
            self.chosenAttributes = []
            self.allAttributes = []

##        self.openContext("", data)

        self.usedAttributes = dict.fromkeys(self.chosenAttributes, 1)
        self.setInputAttributes()

        self.setOutput()
        self.updateInterfaceState()

        
    def setOutput(self):
        if self.data:
            self.applyButton.setEnabled(False)

            fs = FeatureSelection(self.data, self.chosenAttributes)
            self.send("Examples", fs.data)
            
        else:
            self.send("Examples", None)

        
    def reset(self):
        data = self.data
        self.data = None
        self.onDataInput(data)

        
    def disableButtons(self, *arg):
        for b in arg:
            b.setEnabled(False)

    def setButton(self, button, dir):
        button.setText(dir)
        button.setEnabled(True)

        
    def updateInterfaceState(self):
        if self.selectedInput:
            self.setButton(self.attributesButton, ">")
            self.disableButtons(self.attributesButtonUp, self.attributesButtonDown)            
        elif self.selectedChosen:
            self.setButton(self.attributesButton, "<")

            mini, maxi = min(self.selectedChosen), max(self.selectedChosen)
            cons = maxi - mini == len(self.selectedChosen) - 1
            self.attributesButtonUp.setEnabled(cons and mini)
            self.attributesButtonDown.setEnabled(cons and maxi < len(self.chosenAttributes)-1)

        else:
            self.disableButtons(self.attributesButtonUp, self.attributesButtonDown,
                                self.attributesButton)


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
            self.inputAttributes = filter(lambda x:not self.usedAttributes.has_key(x), self.allAttributes)
        else:
            self.inputAttributes = []

    def removeFromUsed(self, attributes):
        for attr in attributes:
            del self.usedAttributes[attr]
        self.setInputAttributes()

    def addToUsed(self, attributes):
        self.usedAttributes.update(dict.fromkeys(attributes))
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
        if (len(self.chosenAttributes)):
            self.applyButton.setEnabled(True)
        else:
            self.applyButton.setEnabled(False)


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

        
    def onAttributesButtonUpClick(self):
        self.moveSelection("chosenAttributes", "selectedChosen", -1)
        
    def onAttributesButtonDownClick(self):
        self.moveSelection("chosenAttributes", "selectedChosen", 1)
        

if __name__=="__main__":
    import sys
    from orngTextCorpus import *
    hrdict = '../../OrangeWidgets/TextData/hrvatski_rjecnik.fsa'
    engdict = '../../OrangeWidgets/TextData/engleski_rjecnik.fsa'
    hrstop = '../../OrangeWidgets/TextData/hrvatski_stoprijeci.txt'
    engstop = '../../OrangeWidgets/TextData/engleski_stoprijeci.txt'    
    
    lem = lemmatizer.FSALemmatization(engdict)
    for word in loadWordSet(engstop):
        lem.stopwords.append(word)       

    fName = '/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/Other/reuters-exchanges-small.xml'
    #fName = '/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/Other/test.xml'
    #fName = '/home/mkolar/Docs/Diplomski/repository/orange/HR-learn-norm.xml'

    b = TextCorpusLoader(fName
            , lem = lem
##            , wordsPerDocRange = (50, -1)
##            , doNotParse = ['small', 'a']
            , tags = {"content":"cont"}
            )

    a=QApplication(sys.argv)
    ow=OWFeatureSelectionMeasures()
    a.setMainWidget(ow)
    ow.show()
    ow.onDataInput(b.data)
    a.exec_loop()
    ow.saveSettings()
