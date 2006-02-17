"""
<name>Merge Data</name>
<description>Merge datasets based on values of selected attributes.</description>
<icon>icons/MergeData.png</icon>
<priority>1110</priority>
<contact>Peter Juvan (peter.juvan@fri.uni-lj.si)</contact>
"""

import orange
from OWWidget import *
from qttable import *
import OWGUI


class OWMergeData(OWWidget):

##    settingsList = ["memberName"]

    def __init__(self, parent = None, signalManager = None, name = "Merge data"):
        OWWidget.__init__(self, parent, signalManager, name)  #initialize base class

        # set channels
        self.inputs = [("Examples A", ExampleTable, self.onDataAInput), ("Examples B", ExampleTable, self.onDataBInput)]
        self.outputs = [("Merged Examples A+B", ExampleTable), ("Merged Examples B+A", ExampleTable)]

        # data
        self.dataA = None
        self.dataB = None
        self.varListA = []
        self.varListB = []
        self.varA = None
        self.varB = None

        # load settings
        self.loadSettings()
        
        # GUI
        self.mainArea.setFixedWidth(0)
        ca=QFrame(self.controlArea)
        gl=QGridLayout(ca,2,2,5)
        
        # attribute A
        boxAttrA = QVGroupBox('Attribute A', ca)
        gl.addWidget(boxAttrA, 0,0)
        self.lbAttrA = QListBox(boxAttrA)
        self.connect(self.lbAttrA, SIGNAL('selectionChanged()'), self.lbAttrAChange)

        # attribute B
        boxAttrB = QVGroupBox('Attribute B', ca)
        gl.addWidget(boxAttrB, 0,1)
        self.lbAttrB = QListBox(boxAttrB)
        self.connect(self.lbAttrB, SIGNAL('selectionChanged()'), self.lbAttrBChange)

        # info A
        boxDataA = QVGroupBox('Data A', ca)
        gl.addWidget(boxDataA, 1,0)
        self.lblDataAExamples = OWGUI.widgetLabel(boxDataA, "num examples")
        self.lblDataAAttributes = OWGUI.widgetLabel(boxDataA, "num attributes")

        # info B
        boxDataB = QVGroupBox('Data B', ca)
        gl.addWidget(boxDataB, 1,1)
        self.lblDataBExamples = OWGUI.widgetLabel(boxDataB, "num examples")
        self.lblDataBAttributes = OWGUI.widgetLabel(boxDataB, "num attributes")

##        # settings
##        boxSettings = QVGroupBox('Settings', ca)
##        gl.addMultiCellWidget(boxSettings, 2,2,0,1)
##        OWGUI.button(boxSettings, self, "Clear selection", callback=self.onClearSelectionClick)

        # icons
        self.icons = self.createAttributeIconDict()

        # resize        
        self.resize(500,500)


    ############################################################################################################################################################
    ## Data input and output management
    ############################################################################################################################################################

    def onDataAInput(self, data):
        # update self.dataA, self.varListA and self.varA
        self.dataA = data
        if data:
            self.varListA = data.domain.variables.native() + data.domain.getmetas().values()
        else:
            self.varListA = []
        if not self.varA in self.varListA:
            self.varA = None
        # update info
        self.updateInfoA()
        # update attribute A listbox
        self.lbAttrA.clear()
        for var in self.varListA:
            self.lbAttrA.insertItem(self.icons[var.varType], var.name)
        self.sendData()


    def onDataBInput(self, data):
        # update self.dataB, self.varListB and self.varB
        self.dataB = data
        if data:
            self.varListB = data.domain.variables.native() + data.domain.getmetas().values()
        else:
            self.varListB = []
        if not self.varB in self.varListB:
            self.varB = None
        # update info
        self.updateInfoB()
        # update attribute B listbox
        self.lbAttrB.clear()
        for var in self.varListB:
            self.lbAttrB.insertItem(self.icons[var.varType], var.name)
        self.sendData()


    def updateInfoA(self):
        """Updates data A info box.
        """
        if self.dataA:
            self.lblDataAExamples.setText("%s example%s" % self._sp(self.dataA))
            self.lblDataAAttributes.setText("%s attribute%s" % self._sp(self.varListA))
        else:
            self.lblDataAExamples.setText("No data on input A.")
            self.lblDataAAttributes.setText("")
        

    def updateInfoB(self):
        """Updates data B info box.
        """
        if self.dataB:
            self.lblDataBExamples.setText("%s example%s" % self._sp(self.dataB))
            self.lblDataBAttributes.setText("%s attribute%s" % self._sp(self.varListB))
        else:
            self.lblDataBExamples.setText("No data on input B.")
            self.lblDataBAttributes.setText("")


    def sendData(self):
        """Sends out data.
        """
        if self.varA and self.varB:
            # create dictionaries: attribute values -> example index
            val2idxDictA = {}
            if self.varA.varType == orange.VarTypes.String:
                # odstrani, ko Janez popravi bug #60
                for eIdx, e in enumerate(self.dataA):
                    val2idxDictA[str(e[self.varA])] = eIdx
            else:
                for eIdx, e in enumerate(self.dataA):
                    val2idxDictA[e[self.varA].native()] = eIdx
            val2idxDictB = {}
            if self.varB.varType == orange.VarTypes.String:
                # odstrani, ko Janez popravi bug #60
                for eIdx, e in enumerate(self.dataB):
                    val2idxDictB[str(e[self.varB])] = eIdx
            else:
                for eIdx, e in enumerate(self.dataB):
                    val2idxDictB[e[self.varB].native()] = eIdx
##            print val2idxDictA
##            print val2idxDictB
            # example table names
            nameA = self.dataA.name
            if not nameA: nameA = "Examples A"
            nameB = self.dataB.name
            if not nameB: nameB = "Examples B"
            # create example B with all values unknown
            exBDK = orange.Example(self.dataB[0])
            for var in self.varListB:
##                exBDK[var] = orange.Value(var.varType, orange.ValueTypes.DK)
                exBDK[var] = "?"
            # build example table to append to the right of A
            vlBreduced = list(self.varListB)
            vlBreduced.remove(self.varB)
            domBreduced = orange.Domain(vlBreduced, None)
            etBreduced = orange.ExampleTable(domBreduced)
            if self.varA.varType == orange.VarTypes.String:
                # odstrani, ko Janez popravi bug #60
                for e in self.dataA:
                    dataBidx = val2idxDictB.get(str(e[self.varA]), None)
                    if dataBidx <> None:
                        etBreduced.append(self.dataB[dataBidx])
                    else:
                        etBreduced.append(orange.Example(domBreduced, exBDK))
            else:
                for e in self.dataA:
                    dataBidx = val2idxDictB.get(e[self.varA].native(), None)
                    if dataBidx <> None:
                        etBreduced.append(self.dataB[dataBidx])
                    else:
                        etBreduced.append(orange.Example(domBreduced, exBDK))
            etAB = orange.ExampleTable([self.dataA, etBreduced])
            etAB.name = nameA + " (merged with %s)" % nameB
            self.send("Merged Examples A+B", etAB)
            
            # create example A with all values unknown
            exADK = orange.Example(self.dataA[0])
            for var in self.varListA:
##                exADK[var] = orange.Value(var.varType, orange.ValueTypes.DK)
                exADK[var] = "?"
            # build example table to append to the right of B
            vlAreduced = list(self.varListA)
            vlAreduced.remove(self.varA)
            domAreduced = orange.Domain(vlAreduced, None)
            etAreduced = orange.ExampleTable(domAreduced)
            if self.varB.varType == orange.VarTypes.String:
                # odstrani, ko Janez popravi bug #60
                for e in self.dataB:
                    dataAidx = val2idxDictA.get(str(e[self.varB]), None)
                    if dataAidx <> None:
                        etAreduced.append(self.dataA[dataAidx])
                    else:
                        etAreduced.append(orange.Example(domAreduced, exADK))
            else:
                for e in self.dataB:
                    dataAidx = val2idxDictA.get(e[self.varB].native(), None)
                    if dataAidx <> None:
                        etAreduced.append(self.dataA[dataAidx])
                    else:
                        etAreduced.append(orange.Example(domAreduced, exADK))
            etBA = orange.ExampleTable([self.dataB, etAreduced])
            etBA.name = nameB + " (merged with %s)" % nameA
            self.send("Merged Examples B+A", etBA)
        else:
            self.send("Merged Examples A+B", None)
            self.send("Merged Examples B+A", None)


    ############################################################################################################################################################
    ## Event handlers
    ############################################################################################################################################################

    def lbAttrAChange(self):
        if self.dataA:
##            print "self.lbAttrA.currentItem()", self.lbAttrA.currentItem()
            currItem = self.lbAttrA.currentItem()
            if currItem >= 0 and self.lbAttrA.isSelected(currItem):
                self.varA = self.varListA[self.lbAttrA.currentItem()]
            else:
                self.varA = None
        else:
            self.varA = None
        self.sendData()


    def lbAttrBChange(self):
        if self.dataB:
            currItem = self.lbAttrB.currentItem()
            if currItem >= 0 and self.lbAttrB.isSelected(currItem):
                self.varB = self.varListB[self.lbAttrB.currentItem()]
            else:
                self.varB = None
        else:
            self.varB = None
        self.sendData()


##    def onClearSelectionClick(self):
##        self.lbAttrA.setCurrentItem(-1)
##        self.lbAttrA.setSelected(self.lbAttrA.currentItem(), False)
##        self.lbAttrB.setCurrentItem(-1)
##        self.lbAttrA.clearSelection()
##        self.lbAttrB.clearSelection()
##        print "currentItem:", self.lbAttrA.currentItem(), "isSelected:", self.lbAttrA.isSelected(self.lbAttrA.currentItem()), "self.varA", self.varA
##        print "currentItem:", self.lbAttrB.currentItem(), "isSelected:", self.lbAttrB.isSelected(self.lbAttrB.currentItem()), "self.varB", self.varB

    ############################################################################################################################################################
    ## Utility functions 
    ############################################################################################################################################################

    def _sp(self, l, capitalize=True):
        """Input: list; returns tupple (str(len(l)), "s"/"")
        """
        n = len(l)
        if n == 0:
            if capitalize:                    
                return "No", "s"
            else:
                return "no", "s"
        elif n == 1:
            return str(n), ''
        else:
            return str(n), 's'



if __name__=="__main__":
    import sys
    import OWDataTable, orngSignalManager
    signalManager = orngSignalManager.SignalManager(0)
    #data = orange.ExampleTable('dicty_800_genes_from_table07.tab')
##    data = orange.ExampleTable(r'..\..\doc\datasets\adult_sample.tab')
    dataA = orange.ExampleTable(r'c:\Documents and Settings\peterjuv\My Documents\STEROLTALK\Sterolgene v.0 mouse\sterolgene v.0 mouse controlGeneRatios.tab')
    dataB = orange.ExampleTable(r'c:\Documents and Settings\peterjuv\My Documents\STEROLTALK\Sterolgene v.0 mouse\sterolgene v.0 mouse controlGeneRatios 2 ecoli.tab')
    a=QApplication(sys.argv)
    ow=OWMergeData()
    a.setMainWidget(ow)
    ow.show()
    ow.onDataAInput(dataA)
    ow.onDataBInput(dataB)
    # data table
    dt = OWDataTable.OWDataTable(signalManager = signalManager)
    signalManager.addWidget(ow)
    signalManager.addWidget(dt)
    signalManager.setFreeze(1)
    signalManager.addLink(ow, dt, 'Merged Examples A+B', 'Examples', 1)
    signalManager.addLink(ow, dt, 'Merged Examples B+A', 'Examples', 1)
    signalManager.setFreeze(0)
    dt.show()
    a.exec_loop()

