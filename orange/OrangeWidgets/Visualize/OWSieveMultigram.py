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
from math import sqrt
from orngScaleData import discretizeDomain

###########################################################################################
##### WIDGET : Polyviz visualization
###########################################################################################
class OWSieveMultigram(OWVisWidget):
    settingsList = ["graph.lineWidth", "graph.maxPearson", "graph.minPearson", "showAllAttributes"]
    contextHandlers = {"": DomainContextHandler("", [ContextField("shownAttributes", DomainContextHandler.RequiredList, selected="selectedShown", reservoir="hiddenAttributes")])}


    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Sieve Multigram", wantGraph = True, wantStatusBar = True)

        self.inputs = [("Examples", ExampleTable, self.setData), ("Attribute Selection List", AttributeList, self.setShownAttributes)]
        self.outputs = []

        #add a graph widget
        self.graph = OWSieveMultigramGraph(self.mainArea)
        self.graph.useAntialiasing = 1
        self.mainArea.layout().addWidget(self.graph)

        #set default settings
        self.graphCanvasColor = str(QColor(Qt.white).name())
        self.data = None
        self.graph.lineWidth = 3
        self.graph.minPearson = 2
        self.graph.maxPearson = 10
        self.showAllAttributes = 0

        # add a settings dialog and initialize its values
        self.loadSettings()

        #GUI
        # add a settings dialog and initialize its values
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.GeneralTab = OWGUI.createTabPage(self.tabs, "Main")
        self.SettingsTab = OWGUI.createTabPage(self.tabs, "Settings")

        #add controls to self.controlArea widget
        self.createShowHiddenLists(self.GeneralTab, callback = self.updateGraph)

        OWGUI.button(self.GeneralTab, self, "Find Interesting Attr.", callback = self.interestingSubsetSelection, debuggingEnabled = 0)

        OWGUI.hSlider(self.SettingsTab, self, 'graph.lineWidth', box = 1, label = "Max line width:", minValue=1, maxValue=10, step=1, callback = self.updateGraph)
        residualBox = OWGUI.widgetBox(self.SettingsTab, "Attribute independence (Pearson residuals)")
        OWGUI.hSlider(residualBox, self, 'graph.maxPearson', label = "Max residual:", minValue=4, maxValue=12, step=1, callback = self.updateGraph)
        OWGUI.hSlider(residualBox, self, 'graph.minPearson', label = "Min residual:", minValue=0, maxValue=4, step=1, callback = self.updateGraph, tooltip = "The minimal absolute residual value that will be shown in graph")
        self.SettingsTab.layout().addStretch(100)

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        self.resize(800, 600)

    def sendReport(self):
        self.reportImage(self.graph.saveToFileDirect, QSize(500,500))

    # receive new data and update all fields
    def setData(self, data):
        self.closeContext()
        self.information()
        self.data = None
        if data: 
            data = orange.Preprocessor_dropMissing(data)
        if data and data.domain.hasContinuousAttributes():
            data = discretizeDomain(data, 1)
            self.information("Continuous attributes were discretized using entropy discretization.")

        self.data = data
        self.computeProbabilities()

        self.setShownAttributeList()
        self.openContext("", self.data)
        self.resetAttrManipulation()
        self.updateGraph()


    def sendShownAttributes(self):
        pass

    def updateGraph(self, *args):
        self.graph.updateData(self.data, self.getShownAttributeList(), self.probabilities)
        
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
                        if abs(pearson) > self.graph.minPearson and attrXName not in interestingList: interestingList.append(attrXName)
                        if abs(pearson) > self.graph.minPearson and attrYName not in interestingList: interestingList.append(attrYName)

        # remove attributes that are not in interestingList from visible attribute list
        self.setShownAttributeList(interestingList)
        self.updateGraph()

    def computeProbabilities(self):
        self.probabilities = {}
        if self.data == None: return

        self.setStatusBarText("Please wait. Computing...")
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
        self.setStatusBarText("")

    # receive info about which attributes to show
    def setShownAttributes(self, list):
        self.setShownAttributeList(list)
        self.updateGraph()
    

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSieveMultigram()
    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\zoo.tab")
    ow.setData(data)
    ow.handleNewSignals()
    ow.show()
    a.exec_()

    #save settings
    ow.saveSettings()
