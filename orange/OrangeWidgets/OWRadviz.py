"""
<name>Radviz</name>
<description>Shows data using radviz visualization method</description>
<category>Visualization</category>
<icon>icons/Radviz.png</icon>
<priority>1100</priority>
"""
# Radviz.py
#
# Show data using radviz visualization method
# 

from OWWidget import *
from OWRadvizOptions import *
from random import betavariate 
from OWRadvizGraph import *
from OData import *
import OWVisAttrSelection
from OWVisTools import *
import time

###########################################################################################
##### WIDGET : Radviz visualization
###########################################################################################
class OWRadviz(OWWidget):
    settingsList = ["pointWidth", "attrContOrder", "attrDiscOrder", "jitteringType", "jitterSize", "graphCanvasColor", "globalValueScaling", "kNeighbours", "enhancedTooltips"]
    spreadType=["none","uniform","triangle","beta"]
    attributeContOrder = ["None","RelieF"]
    attributeDiscOrder = ["None","RelieF","GainRatio","Gini", "Oblivious decision graphs"]
    jitterSizeList = ['0.1','0.5','1','2','3','4','5','7', '10', '15', '20']
    jitterSizeNums = [0.1,   0.5,  1,  2 , 3,  4 , 5 , 7 ,  10,   15,   20]
    kNeighboursList = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '15', '17', '20', '25', '30', '40', '60', '80', '100', '150', '200']
    kNeighboursNums = [ 1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7 ,  8 ,  9 ,  10 ,  12 ,  15 ,  17 ,  20 ,  25 ,  30 ,  40 ,  60 ,  80 ,  100 ,  150 ,  200 ]
        
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Radviz", "Show data using Radviz visualization method", TRUE, TRUE)

        #set default settings
        self.pointWidth = 5
        self.attrDiscOrder = "RelieF"
        self.attrContOrder = "RelieF"
        self.jitteringType = "uniform"
        self.attrOrdering = "Original"
        self.enhancedTooltips = 1
        self.globalValueScaling = 1
        self.jitterSize = 1
        self.kNeighbours = 1
        
        self.graphCanvasColor = str(Qt.white.name())
        self.data = None

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.options = OWRadvizOptions()        

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWRadvizGraph(self.mainArea)
        self.box.addWidget(self.graph)
        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
        
        self.statusBar.message("")
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # graph main tmp variables
        self.addInput("cdata")
        self.addInput("selection")

        #connect settingsbutton to show options
        self.connect(self.options.widthSlider, SIGNAL("valueChanged(int)"), self.setPointWidth)
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)
        self.connect(self.options.spreadButtons, SIGNAL("clicked(int)"), self.setSpreadType)
        self.connect(self.options.jitterSize, SIGNAL("activated(int)"), self.setJitteringSize)
        self.connect(self.options.globalValueScaling, SIGNAL("clicked()"), self.setGlobalValueScaling)
        self.connect(self.options.useEnhancedTooltips, SIGNAL("clicked()"), self.setUseEnhancedTooltips)
        self.connect(self.options.attrContButtons, SIGNAL("clicked(int)"), self.setAttrContOrderType)
        self.connect(self.options.attrDiscButtons, SIGNAL("clicked(int)"), self.setAttrDiscOrderType)
        self.connect(self.options, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)

        #add controls to self.controlArea widget
        self.selClass = QVGroupBox(self.space)
        self.attrOrderingButtons = QVButtonGroup("Attribute ordering", self.space)
        self.shownAttribsGroup = QVGroupBox(self.space)
        self.addRemoveGroup = QHButtonGroup(self.space)
        self.hiddenAttribsGroup = QVGroupBox(self.space)
        self.selClass.setTitle("Class attribute")
        self.shownAttribsGroup.setTitle("Shown attributes")
        self.hiddenAttribsGroup.setTitle("Hidden attributes")
        

        self.classCombo = QComboBox(self.selClass)
        self.showContinuousCB = QCheckBox('show continuous', self.selClass)
        self.connect(self.showContinuousCB, SIGNAL("clicked()"), self.setClassCombo)

        self.shownAttribsLB = QListBox(self.shownAttribsGroup)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)

        self.optimizationDlgButton = QPushButton('Optimization dialog', self.attrOrderingButtons)
        
        self.optimizationDlg = OptimizationDialog(None)
        self.optimizationDlg.parentName = "Radviz"
        self.optimizationDlg.kValue = self.kNeighbours
        
        self.connect(self.optimizationDlgButton, SIGNAL("clicked()"), self.optimizationDlg.show)
        self.connect(self.optimizationDlg.interestingList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        
        self.attrButtonGroup = QHButtonGroup(self.shownAttribsGroup)
        self.buttonUPAttr = QPushButton("Attr UP", self.attrButtonGroup)
        self.buttonDOWNAttr = QPushButton("Attr DOWN", self.attrButtonGroup)

        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        self.progressGroup = QVGroupBox(self.space)
        self.progressGroup.setTitle("Optimization progress")
        self.progressBar = QProgressBar(100, self.progressGroup, "progress bar")
        self.progressBar.setCenterIndicator(1)

        self.showGnuplotButton = QPushButton("Show with Gnuplot", self.space)
        self.saveGnuplotButton = QPushButton("Save Gnuplot picture", self.space)
        self.connect(self.showGnuplotButton, SIGNAL("clicked()"), self.drawGnuplot)
        self.connect(self.saveGnuplotButton, SIGNAL("clicked()"), self.saveGnuplot)

        self.saveProjectionAsTabButton = QPushButton("Save projection as TAB", self.space)
        self.connect(self.saveProjectionAsTabButton, SIGNAL("clicked()"), self.saveProjectionAsTab)
        self.currentFileIndex = 1
            
        #connect controls to appropriate functions
        self.connect(self.classCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)
        self.connect(self.optimizationDlg.optimizeSeparationButton, SIGNAL("clicked()"), self.optimizeSeparation)
        self.connect(self.optimizationDlg.optimizeAllSubsetSeparationButton, SIGNAL("clicked()"), self.optimizeAllSubsetSeparation)
        self.connect(self.optimizationDlg.attrKNeighbour, SIGNAL("activated(int)"), self.setKNeighbours)

        self.connect(self.buttonUPAttr, SIGNAL("clicked()"), self.moveAttrUP)
        self.connect(self.buttonDOWNAttr, SIGNAL("clicked()"), self.moveAttrDOWN)

        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)

        # add a settings dialog and initialize its values
        self.activateLoadedSettings()

        self.resize(900, 700)

    def saveProjectionAsTab(self):
        self.graph.saveProjectionAsTabData(self.getShownAttributeList(), str(self.classCombo.currentText()), "tabData" + str(self.currentFileIndex)+".tab")
        self.currentFileIndex+=1


    def drawGnuplot(self):
        self.graph.drawGnuplot(self.getShownAttributeList(), str(self.classCombo.currentText()))

    def saveGnuplot(self):
        self.graph.saveGnuplot(self.getShownAttributeList(), str(self.classCombo.currentText()))
    
    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.options.spreadButtons.setButton(self.spreadType.index(self.jitteringType))
        self.options.attrContButtons.setButton(self.attributeContOrder.index(self.attrContOrder))
        self.options.attrDiscButtons.setButton(self.attributeDiscOrder.index(self.attrDiscOrder))
        self.options.gSetCanvasColor.setNamedColor(str(self.graphCanvasColor))
        self.options.widthSlider.setValue(self.pointWidth)
        self.options.widthLCD.display(self.pointWidth)
        self.options.globalValueScaling.setChecked(self.globalValueScaling)
        self.options.useEnhancedTooltips.setChecked(self.enhancedTooltips)

        # set items in jitter size combo
        for i in range(len(self.jitterSizeList)):
            self.options.jitterSize.insertItem(self.jitterSizeList[i])
        self.options.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        # set items in k neighbours combo
        for i in range(len(self.kNeighboursList)):
            self.optimizationDlg.attrKNeighbour.insertItem(self.kNeighboursList[i])
        self.optimizationDlg.attrKNeighbour.setCurrentItem(self.kNeighboursNums.index(self.kNeighbours))

        self.graph.setEnhancedTooltips(self.enhancedTooltips)        
        self.graph.setJitteringOption(self.jitteringType)
        self.graph.setPointWidth(self.pointWidth)
        self.graph.setCanvasColor(self.options.gSetCanvasColor)
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setJitterSize(self.jitterSize)

    def setPointWidth(self, n):
        self.pointWidth = n
        self.graph.setPointWidth(n)
        self.updateGraph()
        
    # jittering options
    def setSpreadType(self, n):
        self.jitteringType = self.spreadType[n]
        self.graph.setJitteringOption(self.spreadType[n])
        self.graph.setData(self.data)
        self.updateGraph()

    def setUseEnhancedTooltips(self):
        self.enhancedTooltips = self.options.useEnhancedTooltips.isChecked()
        self.graph.setEnhancedTooltips(self.enhancedTooltips)
        self.updateGraph()

    # jittering options
    def setJitteringSize(self, n):
        self.jitterSize = self.jitterSizeNums[n]
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setData(self.data)
        self.updateGraph()

    def setKNeighbours(self, n):
        self.kNeighbours = self.kNeighboursNums[n]
        self.optimizationDlg.kValue = self.kNeighbours

    # continuous attribute ordering
    def setAttrContOrderType(self, n):
        self.attrContOrder = self.attributeContOrder[n]
        if self.data != None:
            self.setShownAttributeList(self.data)
        self.updateGraph()

    # discrete attribute ordering
    def setAttrDiscOrderType(self, n):
        self.attrDiscOrder = self.attributeDiscOrder[n]
        if self.data != None:
            self.setShownAttributeList(self.data)
        self.updateGraph()

    # ####################################
    # find optimal class separation for shown attributes
    def optimizeSeparation(self):
        if self.data != None:
            if len(self.getShownAttributeList()) > 7:
                res = QMessageBox.information(self,'Radviz','This operation could take a long time, because of large number of attributes. Continue?','Yes','No', QString.null,0,1)
                if res != 0: return

            self.graph.percentDataUsed = self.optimizationDlg.percentDataUsed
            text = str(self.optimizationDlg.exactlyLenCombo.currentText())
            if text == "ALL":
                fullList = self.graph.getOptimalSeparation(self.getShownAttributeList(), str(self.classCombo.currentText()), self.kNeighbours, progressBar = self.progressBar)
            else:
                select = int(text)
                total = len(self.getShownAttributeList())
                combin = combinations(select, total)
                self.progressBar.setTotalSteps(combin)
                self.progressBar.setProgress(0)
                self.graph.totalPossibilities = combin
                self.graph.triedPossibilities = 0
                self.graph.startTime = time.time()
                self.graph.minExamples = int(str(self.optimizationDlg.minTableLenEdit.text()))
                fullList = self.graph.getOptimalExactSeparation(self.getShownAttributeList(), [], str(self.classCombo.currentText()), self.kNeighbours, int(text), int(str(self.optimizationDlg.resultListCombo.currentText())), progressBar = self.progressBar)
                
            if len(fullList) == 0: return

            # fill the "interesting visualizations" list box
            #self.optimizationDlg.clear()
            for i in range(min(len(fullList), int(str(self.optimizationDlg.resultListCombo.currentText())))):
                (accuracy, tableLen, list) = max(fullList)
                self.optimizationDlg.insertItem(accuracy, tableLen, list)  
                fullList.remove((accuracy, tableLen, list))
                
            self.optimizationDlg.updateNewResults()
            self.optimizationDlg.save("temp.proj")
            self.optimizationDlg.interestingList.setCurrentItem(0)

   
    # #############################################
    # find optimal separation for all possible subsets of shown attributes
    def optimizeAllSubsetSeparation(self):
        if self.data != None:
            if len(self.getShownAttributeList()) > 7:
                res = QMessageBox.information(self,'Radviz','This operation could take a long time, because of large number of attributes. Continue?','Yes','No', QString.null,0,1)
                if res != 0: return

            text = str(self.optimizationDlg.maxLenCombo.currentText())
            if text == "ALL":
                maxLen = len(self.getShownAttributeList())
            else:
                maxLen = int(text)

            # compute the number of possible subsets so that when computing we can give a feedback on the progress
            allVisible = len(self.getShownAttributeList())
            table = []; total = 0
            for i in range(2,maxLen+1):
                possible = fact(allVisible) / (fact(i) * fact(allVisible-i))
                table.append(possible)
                total += possible

            self.graph.possibleSubsetsTable = table
            self.graph.totalPossibleSubsets = total
            self.graph.minExamples = int(str(self.optimizationDlg.minTableLenEdit.text()))
            self.graph.percentDataUsed = self.optimizationDlg.percentDataUsed
            maxResultsLen = int(str(self.optimizationDlg.resultListCombo.currentText()))
            fullList = self.graph.getOptimalSubsetSeparation(self.getShownAttributeList(), str(self.classCombo.currentText()), self.kNeighbours, maxLen, maxResultsLen, self.progressBar)
            if len(fullList) == 0: return
            
            # fill the "interesting visualizations" list box
            #self.optimizationDlg.clear()
            for i in range(min(len(fullList), maxResultsLen)):
                (accuracy, itemCount, list) = max(fullList)
                self.optimizationDlg.insertItem(accuracy, itemCount, list)
                fullList.remove((accuracy, itemCount, list))
                
            self.optimizationDlg.updateNewResults()
            self.optimizationDlg.save("temp.proj")
            self.optimizationDlg.interestingList.setCurrentItem(0)


    # ####################################
    # show selected interesting projection
    def showSelectedAttributes(self):
        if self.optimizationDlg.interestingList.count() == 0: return
        index = self.optimizationDlg.interestingList.currentItem()
        (accuracy, tableLen, list, strList) = self.optimizationDlg.optimizedListFiltered[index]

        attrNames = []
        for attr in self.data.domain:
            attrNames.append(attr.name)
        
        for item in list:
            if not item in attrNames:
                print "invalid settings"
                return
        
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()
        for attr in list: self.shownAttribsLB.insertItem(attr)
        for attr in self.data.domain:
            if attr.name not in list: self.hiddenAttribsLB.insertItem(attr.name)
        self.updateGraph()

        
    def setCanvasColor(self, c):
        self.graphCanvasColor = c
        self.graph.setCanvasColor(c)

    def setGlobalValueScaling(self):
        self.globalValueScaling = self.options.globalValueScaling.isChecked()
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setData(self.data)

        # this is not optimal, because we do the rescaling twice (TO DO)
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
            
        self.updateGraph()

        
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
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
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
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
        self.updateGraph()
        self.graph.replot()

    # #####################

    def updateGraph(self):
        self.graph.updateData(self.getShownAttributeList(), str(self.classCombo.currentText()), self.statusBar)
        self.graph.update()
        self.repaint()

    # set combo box values with attributes that can be used for coloring the data
    def setClassCombo(self):
        exText = str(self.classCombo.currentText())
        self.classCombo.clear()
        if self.data == None:
            return

        # add possible class attributes
        self.classCombo.insertItem('(One color)')
        for i in range(len(self.data.domain)):
            attr = self.data.domain[i]
            if attr.varType == orange.VarTypes.Discrete or self.showContinuousCB.isOn() == 1:
                self.classCombo.insertItem(attr.name)

        for i in range(self.classCombo.count()):
            if str(self.classCombo.text(i)) == exText:
                self.classCombo.setCurrentItem(i)
                return

        for i in range(self.classCombo.count()):
            if str(self.classCombo.text(i)) == self.data.domain.classVar.name:
                self.classCombo.setCurrentItem(i)
                return


    # ###### SHOWN ATTRIBUTE LIST ##############
    # set attribute list
    def setShownAttributeList(self, data):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if data == None: return

        self.hiddenAttribsLB.insertItem(data.domain.classVar.name)
        shown, hidden = OWVisAttrSelection.selectAttributes(data, self.attrContOrder, self.attrDiscOrder)
        for attr in shown:
            if attr == data.domain.classVar.name: continue
            self.shownAttribsLB.insertItem(attr)
        for attr in hidden:
            if attr == data.domain.classVar.name: continue
            self.hiddenAttribsLB.insertItem(attr)
        
    def getShownAttributeList (self):
        list = []
        for i in range(self.shownAttribsLB.count()):
            list.append(str(self.shownAttribsLB.text(i)))
        return list
    # #############################################
    
    
    # ###### CDATA signal ################################
    # receive new data and update all fields
    def cdata(self, data):
        self.optimizationDlg.clear()
        #self.data = orange.Preprocessor_dropMissing(data.data)
        self.data = data.data
        self.graph.setData(self.data)
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()
        self.setClassCombo()

        if self.data == None:
            self.repaint()
            return
        
        self.setShownAttributeList(self.data)
        self.updateGraph()
    # ################################################


    # ###### SELECTION signal ################################
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
    # ################################################

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRadviz()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
