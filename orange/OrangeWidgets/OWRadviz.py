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
from OWkNNOptimization import *
import time
from qt import *

###########################################################################################
##### WIDGET : Radviz visualization
###########################################################################################
class OWRadviz(OWWidget):
    settingsList = ["pointWidth", "attrContOrder", "attrDiscOrder", "jitteringType", "jitterSize", "graphCanvasColor", "globalValueScaling", "enhancedTooltips", "showFilledSymbols", "scaleFactor", "showLegend"]
    spreadType=["none","uniform","triangle","beta"]
    attributeContOrder = ["None","RelieF"]
    attributeDiscOrder = ["None","RelieF","GainRatio","Gini", "Oblivious decision graphs"]
    jitterSizeList = ['0.1','0.5','1','2','3','4','5','7', '10', '15', '20']
    jitterSizeNums = [0.1,   0.5,  1,  2 , 3,  4 , 5 , 7 ,  10,   15,   20]
    scaleFactorList = ["1.0", "1.1","1.2","1.3","1.4","1.5","1.6","1.7","1.8","1.9","2.0","2.2","2.4","2.6","2.8", "3.0"]
        
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Radviz", "Show data using Radviz visualization method", TRUE, TRUE)

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1), ("Selection", list, self.selection, 1)]
        self.outputs = [] 

        #set default settings
        self.pointWidth = 4
        self.attrDiscOrder = "RelieF"
        self.attrContOrder = "RelieF"
        self.jitteringType = "uniform"
        self.attrOrdering = "Original"
        self.enhancedTooltips = 1
        self.globalValueScaling = 0
        self.jitterSize = 1
        self.scaleFactor = 1.0
        self.showLegend = 1
        self.showFilledSymbols = 1
        
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
        self.graph.updateSettings(statusBar = self.statusBar)
        
        self.statusBar.message("")
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        #connect settingsbutton to show options
        self.connect(self.options.widthSlider, SIGNAL("valueChanged(int)"), self.setPointWidth)
        self.connect(self.options.scaleCombo, SIGNAL("activated(int)"), self.setScaleFactor)
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)
        self.connect(self.options.spreadButtons, SIGNAL("clicked(int)"), self.setSpreadType)
        self.connect(self.options.jitterSize, SIGNAL("activated(int)"), self.setJitteringSize)
        self.connect(self.options.globalValueScaling, SIGNAL("clicked()"), self.setGlobalValueScaling)
        self.connect(self.options.useEnhancedTooltips, SIGNAL("clicked()"), self.setUseEnhancedTooltips)
        self.connect(self.options.showFilledSymbols, SIGNAL("clicked()"), self.setShowFilledSymbols)
        self.connect(self.options.showLegend, SIGNAL("clicked()"), self.setShowLegend)
        self.connect(self.options.attrContButtons, SIGNAL("clicked(int)"), self.setAttrContOrderType)
        self.connect(self.options.attrDiscButtons, SIGNAL("clicked(int)"), self.setAttrDiscOrderType)
        self.connect(self.options, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)

        #add controls to self.controlArea widget
        self.shownAttribsGroup = QVGroupBox(self.space)
        self.addRemoveGroup = QHButtonGroup(self.space)
        self.hiddenAttribsGroup = QVGroupBox(self.space)
        self.shownAttribsGroup.setTitle("Shown attributes")
        self.hiddenAttribsGroup.setTitle("Hidden attributes")
        self.attrOrderingButtons = QVButtonGroup("Attribute ordering", self.space)
        
        self.shownAttribsLB = QListBox(self.shownAttribsGroup)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)

        self.optimizationDlgButton = QPushButton('kNN Optimization dialog', self.attrOrderingButtons)
        self.optimizationDlg = kNNOptimization(None)
        self.optimizationDlg.parentName = "Radviz"
        self.graph.kNNOptimization = self.optimizationDlg

        
        self.connect(self.optimizationDlgButton, SIGNAL("clicked()"), self.optimizationDlg.show)
        self.connect(self.optimizationDlg.interestingList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        
        self.hbox2 = QHBox(self.shownAttribsGroup)
        self.buttonUPAttr = QPushButton("Attr UP", self.hbox2)
        self.buttonDOWNAttr = QPushButton("Attr DOWN", self.hbox2)

        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        #self.showGnuplotButton = QPushButton("Show with Gnuplot", self.space)
        #self.saveGnuplotButton = QPushButton("Save Gnuplot picture", self.space)
        #self.connect(self.showGnuplotButton, SIGNAL("clicked()"), self.saveProjectionAsTab)
        #self.connect(self.showGnuplotButton, SIGNAL("clicked()"), self.drawGnuplot)
        #self.connect(self.saveGnuplotButton, SIGNAL("clicked()"), self.saveGnuplot)
        
        self.currentFileIndex = 1
            
        #connect controls to appropriate functions
        self.connect(self.optimizationDlg.optimizeSeparationButton, SIGNAL("clicked()"), self.optimizeSeparation)
        self.connect(self.optimizationDlg.optimizeAllSubsetSeparationButton, SIGNAL("clicked()"), self.optimizeAllSubsetSeparation)
        self.connect(self.optimizationDlg.reevaluateResults, SIGNAL("clicked()"), self.testCurrentProjections)

        self.connect(self.optimizationDlg.evaluateButton, SIGNAL("clicked()"), self.evaluateCurrentProjection)
        self.connect(self.optimizationDlg.showKNNCorrectButton, SIGNAL("clicked()"), self.showKNNCorect)
        self.connect(self.optimizationDlg.showKNNWrongButton, SIGNAL("clicked()"), self.showKNNWrong)
        self.connect(self.optimizationDlg.showKNNResetButton, SIGNAL("clicked()"), self.updateGraph)

        self.connect(self.buttonUPAttr, SIGNAL("clicked()"), self.moveAttrUP)
        self.connect(self.buttonDOWNAttr, SIGNAL("clicked()"), self.moveAttrDOWN)

        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)

        # add a settings dialog and initialize its values
        self.activateLoadedSettings()

        self.resize(900, 700)

    def saveProjectionAsTab(self):
        self.graph.saveProjectionAsTabData(self.getShownAttributeList(), "tabData" + str(self.currentFileIndex)+".tab")
        self.currentFileIndex+=1


    def drawGnuplot(self):
        self.graph.drawGnuplot(self.getShownAttributeList())

    def saveGnuplot(self):
        self.graph.saveGnuplot(self.getShownAttributeList())

 
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
        self.options.showFilledSymbols.setChecked(self.showFilledSymbols)
        self.options.showLegend.setChecked(self.showLegend)

        # set items in jitter size combo
        self.options.jitterSize.clear()
        for i in range(len(self.jitterSizeList)):
            self.options.jitterSize.insertItem(self.jitterSizeList[i])
        self.options.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        self.options.scaleCombo.clear()
        for i in range(len(self.scaleFactorList)):
            self.options.scaleCombo.insertItem(self.scaleFactorList[i])
        self.options.scaleCombo.setCurrentItem(self.scaleFactorList.index(str(self.scaleFactor)))

        self.graph.updateSettings(showLegend = self.showLegend, showFilledSymbols = self.showFilledSymbols)
        self.graph.setEnhancedTooltips(self.enhancedTooltips)        
        self.graph.setJitteringOption(self.jitteringType)
        self.graph.setPointWidth(self.pointWidth)
        self.graph.setCanvasColor(self.options.gSetCanvasColor)
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setScaleFactor(self.scaleFactor)

    # #########################
    # KNN OPTIMIZATION BUTTON EVENTS
    # #########################

    def evaluateCurrentProjection(self):
        acc = self.graph.getProjectionQuality(self.getShownAttributeList())
        if self.data.domain.classVar.varType == orange.VarTypes.Discrete:
            QMessageBox.information( None, "Radviz", 'Accuracy of kNN model is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
        else:
            QMessageBox.information( None, "Radviz", 'Mean square error of kNN model is %.2f'%(acc), QMessageBox.Ok + QMessageBox.Default)
        
    def showKNNCorect(self):
        self.graph.updateData(self.getShownAttributeList(), showKNNModel = 1, showCorrect = 1)
        self.graph.update()
        self.repaint()

    def showKNNWrong(self):
        self.graph.updateData(self.getShownAttributeList(), showKNNModel = 1, showCorrect = 0)
        self.graph.update()
        self.repaint()

    def testCurrentProjections(self):
        kList = [3,5,10,15,20,30,50,70,100,150,200]
        #kList = [60]
        results = []
        for i in range(self.optimizationDlg.interestingList.count()):
        #for i in range(1):
            (accuracy, tableLen, list, strList) = self.optimizationDlg.optimizedListFull[i]
            sumAcc = 0.0
            print "Experiment %2.d - %s" % (i, str(list))
            for k in kList: sumAcc += self.graph.getProjectionQuality(list)
            results.append((sumAcc/float(len(kList)), tableLen, list))

        self.optimizationDlg.clear()
        while results != []:
            (accuracy, tableLen, list) = max(results)
            self.optimizationDlg.insertItem(accuracy, tableLen, list)  
            results.remove((accuracy, tableLen, list))

        self.optimizationDlg.updateNewResults()
        self.optimizationDlg.save("temp.proj")
        self.optimizationDlg.interestingList.setCurrentItem(0)

    # ####################################
    # find optimal class separation for shown attributes
    def optimizeSeparation(self):
        if self.data != None:
            """
            if len(self.getShownAttributeList()) > 7:
                res = QMessageBox.information(self,'Radviz','This operation could take a long time, because of large number of attributes. Continue?','Yes','No', QString.null,0,1)
                if res != 0: return
            """
            text = str(self.optimizationDlg.exactlyLenCombo.currentText())
            if text == "ALL":
                fullList = self.graph.getOptimalSeparation(self.getShownAttributeList())
            else:
                select = int(text)
                total = len(self.getShownAttributeList())
                combin = combinations(select, total)
                #self.progressBar.setTotalSteps(combin)
                #self.progressBar.setProgress(0)
                self.graph.totalPossibilities = combin
                self.graph.triedPossibilities = 0
                self.graph.startTime = time.time()
                fullList = self.graph.getOptimalExactSeparation(self.getShownAttributeList(), [], int(text), int(str(self.optimizationDlg.resultListCombo.currentText())))
                
            if len(fullList) == 0: return

            if self.data.domain.classVar.varType == orange.VarTypes.Discrete: funct = max
            else: funct = min
            # fill the "interesting visualizations" list box
            self.optimizationDlg.clear()
            while fullList != []:
                (accuracy, tableLen, list) = funct(fullList)
                self.optimizationDlg.insertItem(accuracy, tableLen, list)  
                fullList.remove((accuracy, tableLen, list))
                
            self.optimizationDlg.updateNewResults()
            self.optimizationDlg.save("temp.proj")
            self.optimizationDlg.interestingList.setCurrentItem(0)

   
    # #############################################
    # find optimal separation for all possible subsets of shown attributes
    def optimizeAllSubsetSeparation(self):
        if self.data != None:
            """
            if len(self.getShownAttributeList()) > 7:
                res = QMessageBox.information(self,'Radviz','This operation could take a long time, because of large number of attributes. Continue?','Yes','No', QString.null,0,1)
                if res != 0: return
            """
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

            self.graph.updateSettings(possibleSubsetsTable = table, totalPossibleSubsets = total)
            fullList = self.graph.getOptimalSubsetSeparation(self.getShownAttributeList(), maxLen)
            if len(fullList) == 0: return

            if self.data.domain.classVar.varType == orange.VarTypes.Discrete: funct = max
            else: funct = min
            # fill the "interesting visualizations" list box
            self.optimizationDlg.clear()
            #for i in range(min(maxResultsLen, len(fullList))):
            while fullList != []:
                (accuracy, itemCount, list) = funct(fullList)
                self.optimizationDlg.insertItem(accuracy, itemCount, list)
                fullList.remove((accuracy, itemCount, list))
                
            self.optimizationDlg.updateNewResults()
            self.optimizationDlg.save("temp.proj")
            self.optimizationDlg.interestingList.setCurrentItem(0)


    # #########################
    # RADVIZ EVENTS
    # #########################
                
    def setScaleFactor(self, n):
        self.scaleFactor = float(self.scaleFactorList[n])
        self.graph.setScaleFactor(self.scaleFactor)
        self.updateGraph()

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

    def setShowFilledSymbols(self):
        self.showFilledSymbols = not self.showFilledSymbols
        self.graph.updateSettings(showFilledSymbols = self.showFilledSymbols)
        self.updateGraph()


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
        self.graph.updateData(self.getShownAttributeList())
        self.graph.update()
        self.repaint()

    def setShowLegend(self):
        self.showLegend = self.options.showLegend.isChecked()
        self.graph.updateSettings(showLegend = self.showLegend)
        self.updateGraph()

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
        print "radviz data"
        self.optimizationDlg.clear()
        self.data = data
        self.graph.setData(self.data)
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()
        
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
