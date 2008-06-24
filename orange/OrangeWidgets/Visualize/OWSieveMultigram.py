"""
<name>Sieve multigram</name>
<description>Sieve multigram.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/SieveMultigram.png</icon>
<priority>4300</priority>
"""

from OWVisWidget import *
from OWSieveMultigramGraph import *
import orngVisFuncts
from orngCI import FeatureByCartesianProduct
import OWGUI

###########################################################################################
##### WIDGET : Polyviz visualization
###########################################################################################
class OWSieveMultigram(OWVisWidget):
    settingsList = ["maxLineWidth", "pearsonMinRes", "pearsonMaxRes", "showAllAttributes"]
    contextHandlers = {"": DomainContextHandler("", [ContextField("shownAttributes", DomainContextHandler.RequiredList, selected="selectedShown", reservoir="hiddenAttributes")])}


    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Sieve Multigram", TRUE)

        self.inputs = [("Examples", ExampleTable, self.setData), ("Attribute Selection List", AttributeList, self.setShownAttributes)]
        self.outputs = []

        #set default settings
        self.graphCanvasColor = str(QColor(Qt.white).name())
        self.data = None
        self.maxLineWidth = 3
        self.pearsonMinRes = 2
        self.pearsonMaxRes = 10
        self.showAllAttributes = 0

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
        self.createShowHiddenLists(self.GeneralTab, callback = self.interestingSubsetSelection)
        
        self.interestingButton = QPushButton("Find Interesting Attr.", self.GeneralTab)
        self.connect(self.interestingButton, SIGNAL("clicked()"),self.interestingSubsetSelection) 

        #connect controls to appropriate functions
        self.connect(self.SettingsTab.lineCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)
        self.connect(self.SettingsTab.pearsonMaxResCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)
        self.connect(self.SettingsTab.applyButton, SIGNAL("clicked()"), self.updateGraph)

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # add a settings dialog and initialize its values
        self.activateLoadedSettings()

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        # set loaded options settings
        self.SettingsTab.lineCombo.setCurrentIndex(self.maxLineWidth-1)
        index = self.SettingsTab.pearsonMaxNums.index(self.pearsonMaxRes)
        self.SettingsTab.pearsonMaxResCombo.setCurrentIndex(index)
        self.SettingsTab.minResidualEdit.setText(str(self.pearsonMinRes))
        self.cbShowAllAttributes()


    # receive new data and update all fields
    def setData(self, data):
        self.closeContext()
        self.data = None
        if data: self.data = orange.Preprocessor_dropMissing(data)
        self.computeProbabilities()

        self.setShownAttributeList()
        self.openContext("", self.data)
        self.resetAttrManipulation()
        self.updateGraph()


    def sendShownAttributes(self):
        pass

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
                index = self.shownAttribsLB.index(self.shownAttribsLB.findItems(attr)[0])
                self.shownAttribsLB.takeItem(index)
                self.hiddenAttribsLB.addItem(attr)
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

    # receive info about which attributes to show
    def setShownAttributes(self, list):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if self.data == None: return

        if self.data.domain.classVar.name not in list:
            self.hiddenAttribsLB.addItem(self.data.domain.classVar.name)

        self.shownAttribsLB.addItems(list)

        for attr in self.data.domain:
            if attr.name not in list:
                self.hiddenAttribsLB.addItem(attr.name)

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
        QToolTip.add(self.hbox2, "The maximum expected Pearson standardized residual. Greater the maximum, brighter the colors.")

        self.hbox3 = QHBox(self.pearsonGroup, "minimum")
        self.residualLabel2 = QLabel('Min residual   ', self.hbox3)
        self.minResidualEdit = QLineEdit(self.hbox3)
        QToolTip.add(self.hbox3, "The minimal absolute residual value that will be shown in graph.")

        self.applyButton = QPushButton("&Apply", self)

        self.initSettings()

    def initSettings(self):
        # line width combo values
        self.lineCombo.addItems([str(i) for i in range(1,10)])

        # max residual combo values
        self.pearsonMaxResCombo.addItems(self.pearsonMaxList)


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSieveMultigram()
    a.setMainWidget(ow)
    ow.show()
    a.exec_()

    #save settings
    ow.saveSettings()
