"""
<name>Sieve multigram</name>
<description>Shows sieve multigram</description>
<category>Visualization</category>
<icon>icons/SieveMultigram.png</icon>
<priority>4150</priority>
"""
# Polyviz.py
#
# Show data using Polyviz visualization method
# 

from OWWidget import *
from random import betavariate 
from OWSieveMultigramGraph import *
import OWVisAttrSelection
from orngCI import FeatureByCartesianProduct


###########################################################################################
##### WIDGET : Polyviz visualization
###########################################################################################
class OWSieveMultigram(OWWidget):
    settingsList = ["maxLineWidth", "pearsonMinRes", "pearsonMaxRes"]
            
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Sieve Multigram", "Show sieve multigram", FALSE, TRUE, icon = "SieveMultigram.png")

        self.inputs = [("Examples", ExampleTable, self.data), ("Selection", list, self.selection)]
        self.outputs = [] 

        #set default settings
        self.graphCanvasColor = str(Qt.white.name())
        self.data = None
        self.maxLineWidth = 3
        self.pearsonMinRes = 2
        self.pearsonMaxRes = 10
        
        # add a settings dialog and initialize its values
        self.loadSettings()

        #GUI
        # add a settings dialog and initialize its values
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        #self.GeneralTab.setFrameShape(QFrame.NoFrame)
        self.SettingsTab = OWSieveMultigramOptions(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")
              
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWSieveMultigramGraph(self.mainArea)
        self.box.addWidget(self.graph)
        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
        self.statusBar.message("")
                
        #add controls to self.controlArea widget
        self.shownAttribsGroup = QVGroupBox(self.GeneralTab)
        self.addRemoveGroup = QHButtonGroup(self.GeneralTab)
        self.hiddenAttribsGroup = QVGroupBox(self.GeneralTab)
        self.shownAttribsGroup.setTitle("Shown attributes")
        self.hiddenAttribsGroup.setTitle("Hidden attributes")

        self.shownAttribsLB = QListBox(self.shownAttribsGroup)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)
        
        self.attrButtonGroup = QHButtonGroup(self.shownAttribsGroup)
        self.buttonUPAttr = QPushButton("Attr UP", self.attrButtonGroup)
        self.buttonDOWNAttr = QPushButton("Attr DOWN", self.attrButtonGroup)

        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        self.interestingButton =QPushButton("Find interesting attr.", self.GeneralTab)
        self.connect(self.interestingButton, SIGNAL("clicked()"),self.interestingSubsetSelection) 

        #connect controls to appropriate functions
        self.connect(self.SettingsTab.lineCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)
        self.connect(self.SettingsTab.pearsonMaxResCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)
        self.connect(self.SettingsTab.applyButton, SIGNAL("clicked()"), self.updateGraph)

        self.connect(self.buttonUPAttr, SIGNAL("clicked()"), self.moveAttrUP)
        self.connect(self.buttonDOWNAttr, SIGNAL("clicked()"), self.moveAttrDOWN)

        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # add a settings dialog and initialize its values
        self.activateLoadedSettings()

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        # set loaded options settings
        self.SettingsTab.lineCombo.setCurrentItem(self.maxLineWidth-1)        
        index = self.SettingsTab.pearsonMaxNums.index(self.pearsonMaxRes)
        self.SettingsTab.pearsonMaxResCombo.setCurrentItem(index)
        self.SettingsTab.minResidualEdit.setText(str(self.pearsonMinRes))

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
        self.updateGraph()
        #self.graph.replot()

    # ###### SHOWN ATTRIBUTE LIST ##############
    # set attribute list
    def setShownAttributeList(self, data, exData):
        if self.data and exData and str(exData.domain) == str(self.data.domain): return  # preserve attribute choice if the domain is the same

        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()
        if data == None: return

        for attr in data.domain:
            if attr.varType == orange.VarTypes.Discrete:
                self.shownAttribsLB.insertItem(attr.name)
        
    def getShownAttributeList (self):
        list = []
        for i in range(self.shownAttribsLB.count()):
            list.append(str(self.shownAttribsLB.text(i)))
        return list
    ##############################################
    
    
    ####### DATA ################################
    # receive new data and update all fields
    def data(self, data):
        exData = self.data
        self.data = None
        if data: self.data = orange.Preprocessor_dropMissing(data)
        self.computeProbabilities()        

        self.setShownAttributeList(self.data, exData)
        self.updateGraph()
        
    #################################################

    def updateGraph(self, *args):
        self.maxLineWidth = int(str(self.SettingsTab.lineCombo.currentText()))
        self.pearsonMaxRes = int(str(self.SettingsTab.pearsonMaxResCombo.currentText()))
        self.pearsonMinRes = float(str(self.SettingsTab.minResidualEdit.text()))
        self.graph.setSettings(self.maxLineWidth, self.pearsonMinRes, self.pearsonMaxRes)
        
        self.graph.updateData(self.data, self.getShownAttributeList(), self.probabilities, self.statusBar)
        self.graph.update()

    def interestingSubsetSelection(self):
        labels = self.getShownAttributeList()
        interestingList = []
        data = self.data

        # create a list of interesting attributes        
        for attrXindex in range(len(labels)):
            attrXName = labels[attrXindex]

            for attrYindex in range(attrXindex+1, len(labels)):
                attrYName = labels[attrYindex]

                for valXindex in range(len(data.domain[attrXName].values)):
                    valX = data.domain[attrXName].values[valXindex]

                    for valYindex in range(len(data.domain[attrYName].values)):
                        valY = data.domain[attrYName].values[valYindex]

                        ((nameX, countX),(nameY, countY), actual, sum) = self.probabilities['%s+%s:%s+%s' %(attrXName, valX, attrYName, valY)]
                        expected = float(countX*countY)/float(sum)
                        if actual == expected == 0: continue
                        elif expected == 0: pearson = actual/sqrt(actual)
                        else:               pearson = (actual - expected) / sqrt(expected)
                        if abs(pearson) > self.pearsonMinRes and attrXName not in interestingList: interestingList.append(attrXName)
                        if abs(pearson) > self.pearsonMinRes and attrYName not in interestingList: interestingList.append(attrYName)                     

        # remove attributes that are not in interestingList from visible attribute list
        for attr in labels:
            if attr not in interestingList:
                index = self.shownAttribsLB.index(self.shownAttribsLB.findItem(attr))
                self.shownAttribsLB.removeItem(index)
                self.hiddenAttribsLB.insertItem(attr)
        self.updateGraph()

    def computeProbabilities(self):
        self.probabilities = {}
        if self.data == None: return

        self.statusBar.message("Please wait. Computing...")
        total = len(self.data)
        conts = {}
        dc = []
        for i in range(len(self.data.domain)):
            dc.append(orange.ContingencyAttrAttr(self.data.domain[i], self.data.domain[i], self.data))
            
        for i in range(len(self.data.domain)):
            if self.data.domain[i].varType == orange.VarTypes.Continuous: continue      # we can only check discrete attributes
            
            cont = dc[i]   # distribution of X attribute
            vals = []
            # compute contingency of x attribute
            for key in cont.keys():
                sum = 0
                try:
                    for val in cont[key]: sum += val
                except: pass
                vals.append(sum)
            conts[self.data.domain[i].name] = (cont, vals)

        for attrX in range(len(self.data.domain)):
            if self.data.domain[attrX].varType == orange.VarTypes.Continuous: continue      # we can only check discrete attributes

            for attrY in range(attrX, len(self.data.domain)):
                if self.data.domain[attrY].varType == orange.VarTypes.Continuous: continue  # we can only check discrete attributes

                (contX, valsX) = conts[self.data.domain[attrX].name]
                (contY, valsY) = conts[self.data.domain[attrY].name]

                # create cartesian product of selected attributes and compute contingency 
                (cart, profit) = FeatureByCartesianProduct(self.data, [self.data.domain[attrX], self.data.domain[attrY]])
                tempData = self.data.select(list(self.data.domain) + [cart])
                contXY = orange.ContingencyAttrClass(cart, tempData)   # distribution of X attribute

                # compute probabilities
                for i in range(len(valsX)):
                    valx = valsX[i]
                    for j in range(len(valsY)):
                        valy = valsY[j]

                        actualCount = 0
                        try:
                            for val in contXY['%s-%s' %(contX.keys()[i], contY.keys()[j])]: actualCount += val
                        except: pass
                        self.probabilities['%s+%s:%s+%s' %(self.data.domain[attrX].name, contX.keys()[i], self.data.domain[attrY].name, contY.keys()[j])] = ((contX.keys()[i], valx), (contY.keys()[j], valy), actualCount, total)
                        self.probabilities['%s+%s:%s+%s' %(self.data.domain[attrY].name, contY.keys()[j], self.data.domain[attrX].name, contX.keys()[i])] = ((contY.keys()[j], valy), (contX.keys()[i], valx), actualCount, total)
        self.statusBar.message("")

    ####### SELECTION signal ################################
    # receive info about which attributes to show
    def selection(self, list):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if self.data == None: return

        if self.data.domain.classVar.name not in list:
            self.hiddenAttribsLB.insertItem(self.data.domain.classVar.name)
            
        for attr in list:
            self.shownAttribsLB.insertItem(attr)

        for attr in self.data.domain:
            if attr.name not in list:
                self.hiddenAttribsLB.insertItem(attr.name)

        self.updateGraph()
    #################################################

class OWSieveMultigramOptions(QVGroupBox):
    pearsonMaxList = ['4','6','8','10','12']
    pearsonMaxNums = [ 4,  6,  8,  10,  12]
    
    def __init__(self,parent=None,name=None):
        QVGroupBox.__init__(self, parent, name)
        self.parent = parent

        self.lineGroup = QVGroupBox(self)
        self.lineGroup.setTitle("Max line width")
        self.lineCombo = QComboBox(self.lineGroup)

        self.pearsonGroup = QVGroupBox(self)
        self.pearsonGroup.setTitle("Attribute independence (Pearson residuals)")

        self.hbox2 = QHBox(self.pearsonGroup, "residual")
        self.residualLabel = QLabel('Max residual', self.hbox2)
        self.pearsonMaxResCombo = QComboBox(self.hbox2)
        QToolTip.add(self.hbox2, "What is maximum expected Pearson standardized residual. Greater the maximum, brighter the colors.")

        self.hbox3 = QHBox(self.pearsonGroup, "minimum")
        self.residualLabel2 = QLabel('Min residual   ', self.hbox3)
        self.minResidualEdit = QLineEdit(self.hbox3)
        QToolTip.add(self.hbox3, "What is minimal absolute residual value that will be shown in graph.")

        self.applyButton = QPushButton("Apply changes", self)

        self.initSettings()        

    def initSettings(self):
        # line width combo values
        for i in range(1,10): self.lineCombo.insertItem(str(i))

        # max residual combo values
        for item in self.pearsonMaxList: self.pearsonMaxResCombo.insertItem(item)     


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSieveMultigram()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
