"""
<name>Sieve multigram</name>
<description>Shows sieve multigram</description>
<category>Classification</category>
<icon>icons/SieveMultigram.png</icon>
<priority>3150</priority>
"""
# Polyviz.py
#
# Show data using Polyviz visualization method
# 

from OWWidget import *
from random import betavariate 
from OData import *
from OWSieveMultigramGraph import *
from OWSieveMultigramOptions import *
import OWVisAttrSelection
from orngCI import FeatureByCartesianProduct


###########################################################################################
##### WIDGET : Polyviz visualization
###########################################################################################
class OWSieveMultigram(OWWidget):
    settingsList = ["independenceKvoc", "maxLineWidth", "pearsonMinRes", "pearsonMaxRes"]
        
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Sieve Multigram", "Show sieve multigram", TRUE, TRUE)

        #set default settings
        
        self.graphCanvasColor = str(Qt.white.name())
        self.data = None

        #load settings
        self.maxLineWidth = 3
        self.independenceKvoc = 3
        self.pearsonMinRes = 2
        self.pearsonMaxRes = 10
        
        # add a settings dialog and initialize its values
        self.options = OWSieveMultigramOptions()
        self.loadSettings()
        self.connect(self.options.lineCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)
        self.connect(self.options.kvocCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)
        self.connect(self.options.pearsonMaxResCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)
        self.connect(self.options.applyButton, SIGNAL("clicked()"), self.updateGraph)

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWSieveMultigramGraph(self.mainArea)
        self.box.addWidget(self.graph)
        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)

        self.shownCriteriaGroup = QVGroupBox(self.controlArea)
        self.shownCriteriaGroup.setTitle("Shown criteria")
        self.criteriaCombo = QComboBox(self.shownCriteriaGroup)
        self.connect(self.criteriaCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)
        
        self.statusBar.message("")
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)
        
        # graph main tmp variables
        self.addInput("cdata")
        self.addInput("selection")

        #add controls to self.controlArea widget
        self.shownAttribsGroup = QVGroupBox(self.space)
        self.addRemoveGroup = QHButtonGroup(self.space)
        self.hiddenAttribsGroup = QVGroupBox(self.space)
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

        #connect controls to appropriate functions
        self.connect(self.buttonUPAttr, SIGNAL("clicked()"), self.moveAttrUP)
        self.connect(self.buttonDOWNAttr, SIGNAL("clicked()"), self.moveAttrDOWN)

        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)

        # add a settings dialog and initialize its values
        self.activateLoadedSettings()

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        # criteria combo values
        self.criteriaCombo.insertItem("Attribute independence")
        self.criteriaCombo.insertItem("Attribute independence (Pearson residuals)")
        self.criteriaCombo.insertItem("Attribute interactions")
        self.criteriaCombo.setCurrentItem(1)

        # set loaded options settings
        index = self.options.kvocNums.index(self.independenceKvoc)
        self.options.kvocCombo.setCurrentItem(index)
        self.options.lineCombo.setCurrentItem(self.maxLineWidth-1)        
        index = self.options.pearsonMaxNums.index(self.pearsonMaxRes)
        self.options.pearsonMaxResCombo.setCurrentItem(index)
        self.options.minResidualEdit.setText(str(self.pearsonMinRes))

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
        self.graph.replot()

    def removeAttribute(self):
        count = self.shownAttribsLB.count()
        pos   = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.shownAttribsLB.isSelected(i):
                text = self.shownAttribsLB.text(i)
                self.shownAttribsLB.removeItem(i)
                self.hiddenAttribsLB.insertItem(text, pos)
        self.updateGraph()
        self.graph.replot()

    # ###### SHOWN ATTRIBUTE LIST ##############
    # set attribute list
    def setShownAttributeList(self, data):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()
        if data == None: return

        for attr in data.domain.attributes:
            if attr.varType == orange.VarTypes.Discrete:
                self.shownAttribsLB.insertItem(attr.name)
        
    def getShownAttributeList (self):
        list = []
        for i in range(self.shownAttribsLB.count()):
            list.append(str(self.shownAttribsLB.text(i)))
        return list
    ##############################################
    
    
    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        self.data = orange.Preprocessor_dropMissing(data.data)
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        self.computeProbabilities()        

        if self.data == None: return
        
        self.setShownAttributeList(self.data)
        self.updateGraph()
        
    #################################################

    def updateGraph(self):
        self.maxLineWidth = int(str(self.options.lineCombo.currentText()))
        self.independenceKvoc = float(str(self.options.kvocCombo.currentText()))
        self.pearsonMaxRes = int(str(self.options.pearsonMaxResCombo.currentText()))
        self.pearsonMinRes = float(str(self.options.minResidualEdit.text()))
        self.graph.setSettings(self.maxLineWidth, self.independenceKvoc, self.pearsonMinRes, self.pearsonMaxRes)
        
        self.graph.updateData(self.data, self.getShownAttributeList(), self.probabilities, self.criteriaCombo.currentText(), self.statusBar)
        self.graph.update()

    def computeProbabilities(self):
        self.probabilities = {}
        if self.data == None: return

        self.statusBar.message("Please wait. Computing...")
        total = len(self.data)
        conts = {}
        dc = orange.DomainContingency(self.data)
        for i in range(len(self.data.domain.attributes)):
            if self.data.domain[i].varType == orange.VarTypes.Continuous: continue      # we can only check discrete attributes
            
            cont = dc[i]   # distribution of X attribute
            vals = []
            # compute contingency of x attribute
            for key in cont.keys():
                sum = 0
                for val in cont[key]: sum += val
                vals.append(sum)
            conts[self.data.domain[i].name] = (cont, vals)

        for attrX in range(len(self.data.domain.attributes)):
            if self.data.domain[attrX].varType == orange.VarTypes.Continuous: continue      # we can only check discrete attributes

            for attrY in range(attrX, len(self.data.domain.attributes)):
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
                        for val in contXY['%s-%s' %(contX.keys()[i], contY.keys()[j])]: actualCount += val
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

        for attr in self.data.domain.attributes:
            if attr.name not in list:
                self.hiddenAttribsLB.insertItem(attr.name)

        self.updateGraph()
    #################################################

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSieveMultigram()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
