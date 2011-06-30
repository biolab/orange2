"""
<name>Merge Data</name>
<description>Merge datasets based on values of selected attributes.</description>
<icon>icons/MergeData.png</icon>
<priority>1110</priority>
<contact>Peter Juvan (peter.juvan@fri.uni-lj.si)</contact>
"""
import orange
from OWWidget import *
import OWGUI

class OWMergeData(OWWidget):

    contextHandlers = {"A": DomainContextHandler("A", [ContextField("varA")], syncWithGlobal=False, contextDataVersion=2),
                       "B": DomainContextHandler("B", [ContextField("varB")], syncWithGlobal=False, contextDataVersion=2)}                                            

    def __init__(self, parent = None, signalManager = None, name = "Merge data"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0)  #initialize base class

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
        self.lbAttrAItems = []
        self.lbAttrBItems = []

        # load settings
        self.loadSettings()

        # GUI
        w = QWidget(self)
        self.controlArea.layout().addWidget(w)
        grid = QGridLayout()
        grid.setMargin(0)
        w.setLayout(grid)

        # attribute A
        boxAttrA = OWGUI.widgetBox(self, 'Attribute A', orientation = "vertical", addToLayout=0)
        grid.addWidget(boxAttrA, 0,0)
        self.lbAttrA = OWGUI.listBox(boxAttrA, self, "lbAttrAItems", callback = self.lbAttrAChange)

        # attribute  B
        boxAttrB = OWGUI.widgetBox(self, 'Attribute B', orientation = "vertical", addToLayout=0)
        grid.addWidget(boxAttrB, 0,1)
        self.lbAttrB = OWGUI.listBox(boxAttrB, self, "lbAttrBItems", callback = self.lbAttrBChange)

        # info A
        boxDataA = OWGUI.widgetBox(self, 'Data A', orientation = "vertical", addToLayout=0)
        grid.addWidget(boxDataA, 1,0)
        self.lblDataAExamples = OWGUI.widgetLabel(boxDataA, "num examples")
        self.lblDataAAttributes = OWGUI.widgetLabel(boxDataA, "num attributes")

        # info B
        boxDataB = OWGUI.widgetBox(self, 'Data B', orientation = "vertical", addToLayout=0)
        grid.addWidget(boxDataB, 1,1)
        self.lblDataBExamples = OWGUI.widgetLabel(boxDataB, "num examples")
        self.lblDataBAttributes = OWGUI.widgetLabel(boxDataB, "num attributes")

        # icons
        self.icons = self.createAttributeIconDict()

        # resize
        self.resize(400,500)


    ############################################################################################################################################################
    ## Data input and output management
    ############################################################################################################################################################
        
    def inVarList(self, varList, var):
        varList = [(v.name, v.varType) for v in varList]
        if var in varList:
            return True, varList.index(var)
        else:
            return False, -1
        
    def onDataAInput(self, data):
        self.closeContext("A")
        self.dataA = data
        # update self.varListA and self.varA
        if self.dataA:
            self.varListA = list(self.dataA.domain.variables) + self.dataA.domain.getmetas().values()
        else:
            self.varListA = []
            
        # update info
        self.updateInfoA()
        # update attribute A listbox
        self.lbAttrA.clear()
        for var in self.varListA:
            self.lbAttrA.addItem(QListWidgetItem(self.icons[var.varType], var.name))
        if self.dataA:
            self.openContext("A", self.dataA)
        match, index = self.inVarList(self.varListA, self.varA)
        if match:
            var = self.varListA[index]
            self.varA = (var.name, var.varType)
            self.lbAttrA.setCurrentItem(self.lbAttrA.item(index))
            
        self.sendData()

    def onDataBInput(self, data):
        self.closeContext("B")
        self.dataB = data
        # update self.varListB and self.varB
        if self.dataB:
            self.varListB = list(self.dataB.domain.variables) + self.dataB.domain.getmetas().values()
        else:
            self.varListB = []
        
        # update info
        self.updateInfoB()
        # update attribute B listbox
        self.lbAttrB.clear()
        for var in self.varListB:
            self.lbAttrB.addItem(QListWidgetItem(self.icons[var.varType], var.name))
        
        if self.dataB:
            self.openContext("B", self.dataB)
        match, index = self.inVarList(self.varListB, self.varB)
        if match:
            var = self.varListB[index]
            self.varB = (var.name, var.varType)
            self.lbAttrB.setCurrentItem(self.lbAttrB.item(index))
            
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
        if self.dataA and self.dataB and self.varA and self.varB:
            self.send("Merged Examples A+B", self.merge(self.dataA, self.dataB, self.varA[0], self.varB[0]))
            self.send("Merged Examples B+A", self.merge(self.dataB, self.dataA, self.varB[0], self.varA[0]))
        else:
            self.send("Merged Examples A+B", None)
            self.send("Merged Examples B+A", None)

    ############################################################################################################################################################
    ## Event handlers
    ############################################################################################################################################################

    def lbAttrAChange(self):
        if self.dataA:
            if self.lbAttrA.selectedItems() != []:
                ind = self.lbAttrA.row(self.lbAttrA.selectedItems()[0])
                var = self.varListA[ind]
                self.varA = (var.name, var.varType)
            else:
                self.varA = None
        else:
            self.varA = None
        self.sendData()


    def lbAttrBChange(self):
        if self.dataB:
            if self.lbAttrB.selectedItems() != []:
                ind = self.lbAttrB.row(self.lbAttrB.selectedItems()[0])
                var = self.varListB[ind]
                self.varB = (var.name, var.varType)
            else:
                self.varB = None
        else:
            self.varB = None
        self.sendData()


    ############################################################################################################################################################
    ## Utility functions
    ############################################################################################################################################################

    def _sp(self, l, capitalize=True):
        """Input: list; returns tuple (str(len(l)), "s"/"")
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

    def merge(self, dataA, dataB, varA, varB):
        """ Merge two tables
        """
        
        val2idx = dict([(e[varB].native(), i) for i, e in reversed(list(enumerate(dataB)))])
        
        for key in ["?", "~", ""]:
            if key in val2idx:
                val2idx.pop(key)
                 
        metasA = dataA.domain.getmetas().items()
        metasB = dataB.domain.getmetas().items()
        
        includedAttsB = [attrB for attrB in dataB.domain if attrB not in dataA.domain]
        includedMetaB = [(mid, meta) for mid, meta in metasB if (mid, meta) not in metasA]
        includedClassVarB = dataB.domain.classVar and dataB.domain.classVar not in dataA.domain
        
        reducedDomainB = orange.Domain(includedAttsB, includedClassVarB)
        reducedDomainB.addmetas(dict(includedMetaB))
        
        
        mergingB = orange.ExampleTable(reducedDomainB)
        
        for ex in dataA:
            ind = val2idx.get(ex[varA].native(), None)
            if ind is not None:
                mergingB.append(orange.Example(reducedDomainB, dataB[ind]))
                
            else:
                mergingB.append(orange.Example(reducedDomainB, ["?"] * len(reducedDomainB)))
                
        return orange.ExampleTable([dataA, mergingB])
    
if __name__=="__main__":
    """
    import sys
    import OWDataTable, orngSignalManager
    signalManager = orngSignalManager.SignalManager(0)
    #data = orange.ExampleTable('dicty_800_genes_from_table07.tab')
##    data = orange.ExampleTable(r'..\..\doc\datasets\adult_sample.tab')
##    dataA = orange.ExampleTable(r'c:\Documents and Settings\peterjuv\My Documents\STEROLTALK\Sterolgene v.0 mouse\sterolgene v.0 mouse probeRatios.tab')
##    dataA = orange.ExampleTable(r'c:\Documents and Settings\peterjuv\My Documents\STEROLTALK\Sterolgene v.0 mouse\Copy of sterolgene v.0 mouse probeRatios.tab')
##    dataB = orange.ExampleTable(r'c:\Documents and Settings\peterjuv\My Documents\STEROLTALK\Sterolgene v.0 mouse\sterolgene v.0 mouse probeRatios.tab')
    dataA = orange.ExampleTable(r'c:\Documents and Settings\peterjuv\My Documents\et1.tab')
    dataB = orange.ExampleTable(r'c:\Documents and Settings\peterjuv\My Documents\et2.tab')
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
    a.exec_()
    """
    import sys
    a=QApplication(sys.argv)
    ow=OWMergeData()
    ow.show()
    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\iris.tab")
    ow.onDataAInput(data)
    a.exec_()
