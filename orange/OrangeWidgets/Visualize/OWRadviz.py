"""
<name>Radviz</name>
<description>Shows data using radviz visualization method</description>
<category>Visualization</category>
<icon>icons/Radviz.png</icon>
<priority>3100</priority>
"""
# Radviz.py
#
# Show data using radviz visualization method
# 

from OWWidget import *
from random import betavariate 
from OWRadvizGraph import *
import OWVisAttrSelection
from OWkNNOptimization import *
import time
import OWToolbars
import OWGUI

###########################################################################################
##### WIDGET : Radviz visualization
###########################################################################################
class OWRadviz(OWWidget):
    settingsList = ["pointWidth", "attrContOrder", "attrDiscOrder", "jitterSize", "graphCanvasColor", "globalValueScaling", "enhancedTooltips", "showFilledSymbols", "scaleFactor", "showLegend", "optimizedDrawing", "useDifferentSymbols", "autoSendSelection", "sendShownAttributes"]
    #spreadType=["none","uniform","triangle","beta"]
    jitterSizeList = ['0.0', '0.1','0.5','1','2','3','4','5','7', '10', '15', '20']
    jitterSizeNums = [0.0, 0.1,   0.5,  1,  2 , 3,  4 , 5 , 7 ,  10,   15,   20]
    scaleFactorList = ["1.0", "1.1","1.2","1.3","1.4","1.5","1.6","1.7","1.8","1.9","2.0","2.2","2.4","2.6","2.8", "3.0"]
        
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Radviz", "Show data using Radviz visualization method", FALSE, TRUE)

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1), ("Selection", list, self.selection, 1)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass)]
 
        #set default settings
        self.pointWidth = 4
        self.attrDiscOrder = "None"
        self.attrContOrder = "None"
        #self.jitteringType = "uniform"
        self.attrOrdering = "Original"
        self.enhancedTooltips = 1
        self.globalValueScaling = 0
        self.jitterSize = 1
        self.scaleFactor = 1.0
        self.showLegend = 1
        self.showFilledSymbols = 1
        self.optimizedDrawing = 1
        self.useDifferentSymbols = 1
        self.autoSendSelection = 0
        self.sendShownAttributes = 1
        self.graphCanvasColor = str(Qt.white.name())
        self.data = None
        self.callbackDeposit = []

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        #self.GeneralTab.setFrameShape(QFrame.NoFrame)
        self.SettingsTab = GroupRadvizOptions(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWRadvizGraph(self, self.mainArea)
        self.box.addWidget(self.graph)
        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
        self.graph.updateSettings(statusBar = self.statusBar)

        self.statusBar.message("")

        #add controls to self.controlArea widget
        self.shownAttribsGroup = QVGroupBox(self.GeneralTab)
        self.addRemoveGroup = QHButtonGroup(self.GeneralTab)
        self.hiddenAttribsGroup = QVGroupBox(self.GeneralTab)
        self.shownAttribsGroup.setTitle("Shown attributes")
        self.hiddenAttribsGroup.setTitle("Hidden attributes")
        self.attrOrderingButtons = QVButtonGroup("Attribute ordering", self.GeneralTab)
        
        self.shownAttribsLB = QListBox(self.shownAttribsGroup)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)

        self.optimizationDlgButton = QPushButton('kNN Optimization dialog', self.attrOrderingButtons)
        self.optimizationDlg = kNNOptimization(None)
        self.optimizationDlg.parentName = "Radviz"
        self.graph.kNNOptimization = self.optimizationDlg

        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)
                               
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

        # ####################################
        #K-NN OPTIMIZATION functionality
        self.connect(self.optimizationDlgButton, SIGNAL("clicked()"), self.optimizationDlg.reshow)
        self.connect(self.optimizationDlg.interestingList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        
        self.connect(self.optimizationDlg.optimizeSeparationButton, SIGNAL("clicked()"), self.optimizeSeparation)
        self.connect(self.optimizationDlg.optimizeAllSubsetSeparationButton, SIGNAL("clicked()"), self.optimizeAllSubsetSeparation)
        self.connect(self.optimizationDlg.reevaluateResults, SIGNAL("clicked()"), self.testCurrentProjections)

        self.connect(self.optimizationDlg.evaluateProjectionButton, SIGNAL("clicked()"), self.evaluateCurrentProjection)
        self.connect(self.optimizationDlg.saveProjectionButton, SIGNAL("clicked()"), self.saveCurrentProjection)
        self.connect(self.optimizationDlg.showKNNCorrectButton, SIGNAL("clicked()"), self.showKNNCorect)
        self.connect(self.optimizationDlg.showKNNWrongButton, SIGNAL("clicked()"), self.showKNNWrong)
        self.connect(self.optimizationDlg.showKNNResetButton, SIGNAL("clicked()"), self.updateGraph)

        self.connect(self.buttonUPAttr, SIGNAL("clicked()"), self.moveAttrUP)
        self.connect(self.buttonDOWNAttr, SIGNAL("clicked()"), self.moveAttrDOWN)

        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)

        # ####################################
        # SETTINGS functionality
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        self.connect(self.SettingsTab.widthSlider, SIGNAL("valueChanged(int)"), self.setPointWidth)
        self.connect(self.SettingsTab.scaleCombo, SIGNAL("activated(int)"), self.setScaleFactor)
        #self.connect(self.SettingsTab.spreadButtons, SIGNAL("clicked(int)"), self.setSpreadType)
        self.connect(self.SettingsTab.jitterSize, SIGNAL("activated(int)"), self.setJitteringSize)
        self.connect(self.SettingsTab.globalValueScaling, SIGNAL("clicked()"), self.setGlobalValueScaling)
        self.connect(self.SettingsTab.useEnhancedTooltips, SIGNAL("clicked()"), self.setUseEnhancedTooltips)
        self.connect(self.SettingsTab.showFilledSymbols, SIGNAL("clicked()"), self.setShowFilledSymbols)
        self.connect(self.SettingsTab.showLegend, SIGNAL("clicked()"), self.setShowLegend)
        self.connect(self.SettingsTab.differentSymbols, SIGNAL("clicked()"), self.setDifferentSymbols)
        self.connect(self.SettingsTab.optimizedDrawing, SIGNAL("clicked()"), self.setOptmizedDrawing)
        self.connect(self.SettingsTab.autoSendSelection, SIGNAL("clicked()"), self.setAutoSendSelection)
        self.connect(self.SettingsTab.sendShownAttributes, SIGNAL("clicked()"), self.setSendShownAttributes)
        self.connect(self.SettingsTab, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)
        self.graph.autoSendSelectionCallback = self.setAutoSendSelection

        # add a settings dialog and initialize its values
        self.activateLoadedSettings()

        self.resize(900, 700)

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        #self.SettingsTab.spreadButtons.setButton(self.spreadType.index(self.jitteringType))
        self.SettingsTab.widthSlider.setValue(self.pointWidth)
        self.SettingsTab.widthLCD.display(self.pointWidth)
        self.SettingsTab.globalValueScaling.setChecked(self.globalValueScaling)
        self.SettingsTab.useEnhancedTooltips.setChecked(self.enhancedTooltips)
        self.SettingsTab.showFilledSymbols.setChecked(self.showFilledSymbols)
        self.SettingsTab.showLegend.setChecked(self.showLegend)
        self.SettingsTab.differentSymbols.setChecked(self.useDifferentSymbols)
        self.SettingsTab.optimizedDrawing.setChecked(self.optimizedDrawing)
        self.SettingsTab.autoSendSelection.setChecked(self.autoSendSelection)
        self.SettingsTab.sendShownAttributes.setChecked(self.sendShownAttributes)
        self.setAutoSendSelection() # update send button state

        # set items in jitter size combo
        self.SettingsTab.jitterSize.clear()
        for i in range(len(self.jitterSizeList)):
            self.SettingsTab.jitterSize.insertItem(self.jitterSizeList[i])
        self.SettingsTab.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        self.SettingsTab.scaleCombo.clear()
        for i in range(len(self.scaleFactorList)):
            self.SettingsTab.scaleCombo.insertItem(self.scaleFactorList[i])
        self.SettingsTab.scaleCombo.setCurrentItem(self.scaleFactorList.index(str(self.scaleFactor)))

        self.graph.updateSettings(showLegend = self.showLegend, showFilledSymbols = self.showFilledSymbols, optimizedDrawing = self.optimizedDrawing)
        self.graph.setEnhancedTooltips(self.enhancedTooltips)        
        #self.graph.setJitteringOption(self.jitteringType)
        self.graph.setPointWidth(self.pointWidth)
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setScaleFactor(self.scaleFactor)
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        self.graph.useDifferentSymbols = self.useDifferentSymbols

    # #########################
    # KNN OPTIMIZATION BUTTON EVENTS
    # #########################
    def saveCurrentProjection(self):
        qname = QFileDialog.getSaveFileName( os.path.realpath(".") + "/Radviz_projection.tab", "Orange Example Table (*.tab)", self, "", "Save File")
        if qname.isEmpty(): return
        name = str(qname)
        if len(name) < 4 or name[-4] != ".":
            name = name + ".tab"
        self.graph.saveProjectionAsTabData(name, self.getShownAttributeList())


    # evaluate knn accuracy on current projection
    def evaluateCurrentProjection(self):
        acc = self.graph.getProjectionQuality(self.getShownAttributeList())
        if self.data.domain.classVar.varType == orange.VarTypes.Continuous:
            QMessageBox.information( None, "Radviz", 'Mean square error of kNN model is %.2f'%(acc), QMessageBox.Ok + QMessageBox.Default)
        else:
            if self.optimizationDlg.getQualityMeasure() == CLASS_ACCURACY:
                QMessageBox.information( None, "Radviz", 'Classification accuracy of kNN model is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
            elif self.optimizationDlg.getQualityMeasure() == AVERAGE_CORRECT:
                QMessageBox.information( None, "Radviz", 'Average probability of correct classification is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
            else:
                QMessageBox.information( None, "Radviz", 'Brier score of kNN model is %.2f' % (acc), QMessageBox.Ok + QMessageBox.Default)
            
    # show quality of knn model by coloring accurate predictions with darker color and bad predictions with light color        
    def showKNNCorect(self):
        self.graph.updateData(self.getShownAttributeList(), showKNNModel = 1, showCorrect = 1)
        self.graph.update()
        self.repaint()

    # show quality of knn model by coloring accurate predictions with lighter color and bad predictions with dark color
    def showKNNWrong(self):
        self.graph.updateData(self.getShownAttributeList(), showKNNModel = 1, showCorrect = 0)
        self.graph.update()
        self.repaint()

    # reevaluate projections in result list with different k values
    def testCurrentProjections(self):
        kListStr = "3,5,10,15,20,30,50,70,100,150,200"
        (Qstring,ok) = QInputDialog.getText("K values", "K values to test (separated with comma)", kListStr)
        if not ok: return
        ks = str(Qstring)
        kListStr = ks.split(",")
        kList = []
        for k in kListStr: kList.append(int(k))
        
        results = []
        count = self.optimizationDlg.interestingList.count()
        self.progressBarInit()
        self.optimizationDlg.disableControls()
        oldKValue = self.optimizationDlg.kValue
        for i in range(count):
            (accuracy, tableLen, list, strList) = self.optimizationDlg.optimizedListFull[i]
            sumAcc = 0.0
            print "Experiment %2.d - %s" % (i+1, str(list))
            for k in kList:
                self.optimizationDlg.kValue = k
                sumAcc += self.graph.getProjectionQuality(list)
            results.append((sumAcc/float(len(kList)), tableLen, list))
            self.progressBarSet(100*i/float(count))

        self.optimizationDlg.kValue = oldKValue
        self.optimizationDlg.clear()
        while results != []:
            (accuracy, tableLen, list) = max(results)
            self.optimizationDlg.insertItem(accuracy, tableLen, list)  
            results.remove((accuracy, tableLen, list))

        self.optimizationDlg.updateNewResults()

        self.progressBarFinished()
        self.optimizationDlg.enableControls()

    # ####################################
    # find optimal class separation for shown attributes
    # numberOfAttrs is different than None only when optimizeSeparation is called by optimizeAllSubsetSeparation
    def optimizeSeparation(self, numberOfAttrs = None):
        if self.data == None: return

        self.progressBarInit()
        self.graph.totalPossibilities = 0
        self.graph.triedPossibilities = 0
        self.optimizationDlg.disableControls()

        if not numberOfAttrs:
            text = str(self.optimizationDlg.exactlyLenCombo.currentText())
            if text == "ALL": number = len(self.getShownAttributeList())
            else:             number = int(text)
        else:                 number = numberOfAttrs
            
        total = len(self.getShownAttributeList())
        startTime = time.time()
        if not self.optimizationDlg.useHeuristics or number < 5:    # use exhaustive search if number of attrs = 3 or 4
            self.graph.totalPossibilities = combinations(number, total)*fact(number-1)/2
            self.graph.triedPossibilities = 0
            self.graph.startTime = time.time()
            fullList = self.graph.getOptimalExactSeparation(self.getShownAttributeList(), [], number)
        else:
            self.optimizationDlg.currentSubset = 0
            self.optimizationDlg.totalSubsets = combinations(number, total)
            print "Total number of feature subsets with %d attributes is %d" % (number, self.optimizationDlg.totalSubsets)
            interestingSubsets = self.optimizationDlg.kNNGetInterestingSubsets(number, self.getShownAttributeList(), self.optimizationDlg.bestSubsets, self.data)
            self.graph.totalPossibilities = len(interestingSubsets) * fact(number-1)/2
            self.graph.triedPossibilities = 0
            self.graph.startTime = time.time()
            fullList = self.graph.getOptimalListSeparation(interestingSubsets)


        self.progressBarFinished()
        self.optimizationDlg.enableControls()
        
        if not numberOfAttrs:
            secs = time.time() - startTime
            print "Used time: %d min, %d sec" %(secs/60, secs%60)
            self.optimizationDlg.clear()
            # fill the "interesting visualizations" list box
            if self.data.domain.classVar.varType == orange.VarTypes.Discrete and self.optimizationDlg.getQualityMeasure() != BRIER_SCORE: funct = max
            else: funct = min
            while fullList != []:
                (accuracy, tableLen, list) = funct(fullList)
                self.optimizationDlg.insertItem(accuracy, tableLen, list)  
                fullList.remove((accuracy, tableLen, list))
            self.optimizationDlg.updateNewResults()
        else:
            return fullList

   
    # #############################################
    # find optimal separation for all possible subsets of shown attributes
    def optimizeAllSubsetSeparation(self):
        if self.data == None: return
        
        """
        if len(self.getShownAttributeList()) > 7:
            res = QMessageBox.information(self,'Radviz','This operation could take a long time, because of large number of attributes. Continue?','Yes','No', QString.null,0,1)
            if res != 0: return
        """
        
        text = str(self.optimizationDlg.maxLenCombo.currentText())
        if text == "ALL": maxLen = len(self.getShownAttributeList())
        else:             maxLen = int(text)

        startTime = time.time()
        fullList = []
        for val in range(3, maxLen+1):
            fullList += self.optimizeSeparation(val)

        self.optimizationDlg.clear()
        if self.data.domain.classVar.varType == orange.VarTypes.Discrete and self.optimizationDlg.getQualityMeasure() != BRIER_SCORE: funct = max
        else: funct = min
        while fullList != []:
            (accuracy, tableLen, list) = funct(fullList)
            self.optimizationDlg.insertItem(accuracy, tableLen, list)  
            fullList.remove((accuracy, tableLen, list))
        self.optimizationDlg.updateNewResults()

        secs = time.time() - startTime
        print "Used time: %d min, %d sec" %(secs/60, secs%60)
            

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

    """        
    # jittering options
    def setSpreadType(self, n):
        self.jitteringType = self.spreadType[n]
        self.graph.setJitteringOption(self.spreadType[n])
        self.graph.setData(self.data)
        self.updateGraph()
    """
    
    def setUseEnhancedTooltips(self):
        self.enhancedTooltips = self.SettingsTab.useEnhancedTooltips.isChecked()
        self.graph.setEnhancedTooltips(self.enhancedTooltips)
        self.updateGraph()

    def setDifferentSymbols(self):
        self.useDifferentSymbols = self.SettingsTab.differentSymbols.isChecked()
        self.graph.useDifferentSymbols = self.useDifferentSymbols
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

       
    def setCanvasColor(self, c):
        self.graphCanvasColor = c
        self.graph.setCanvasColor(c)

    def setGlobalValueScaling(self):
        self.globalValueScaling = self.SettingsTab.globalValueScaling.isChecked()
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setData(self.data)

        # this is not optimal, because we do the rescaling twice (TO DO)
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
            
        self.updateGraph()


    def setAutoSendSelection(self):
        self.autoSendSelection = self.SettingsTab.autoSendSelection.isChecked()
        if self.autoSendSelection:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(0)
            self.sendSelections()
        else:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(1)

    def setSendShownAttributes(self):
        self.sendShownAttributes = self.SettingsTab.sendShownAttributes.isChecked()


    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        if not self.data: return
        (selected, unselected, merged) = self.graph.getSelectionsAsExampleTables(self.getShownAttributeList())
        if not self.sendShownAttributes:
            self.send("Selected Examples",selected)
            self.send("Unselected Examples",unselected)
            self.send("Example Distribution", merged)
        else:
            attrs = self.getShownAttributeList() + [self.data.domain.classVar.name]
            if selected:    self.send("Selected Examples", selected.select(attrs))
            else:           self.send("Selected Examples", None)
            if unselected:  self.send("Unselected Examples", unselected.select(attrs))
            else:           self.send("Unselected Examples", None)
            if merged:
                attrs += [merged.domain.classVar.name]
                self.send("Example Distribution", merged.select(attrs))
            else:           self.send("Example Distribution", None)
            

        
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

    def updateGraph(self, *args):
        self.graph.updateData(self.getShownAttributeList())
        self.graph.update()
        self.repaint()

    def setShowLegend(self):
        self.showLegend = self.SettingsTab.showLegend.isChecked()
        self.graph.updateSettings(showLegend = self.showLegend)
        self.updateGraph()

    def setOptmizedDrawing(self):
        self.optimizedDrawing = self.SettingsTab.optimizedDrawing.isChecked()
        self.graph.updateSettings(optimizedDrawing = self.optimizedDrawing)
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
        self.optimizationDlg.clear()
        exData = self.data
        self.data = data
        self.graph.setData(self.data)

        if not (data and exData and str(exData.domain.attributes) == str(data.domain.attributes)): # preserve attribute choice if the domain is the same                
            self.setShownAttributeList(self.data)
        self.updateGraph()
    # ################################################


    # ###### SELECTION signal ################################
    # receive info about which attributes to show
    def selection(self, list):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if self.data == None: return

        for attr in self.data.domain:
            if attr.name in list: self.shownAttribsLB.insertItem(attr.name)
            else:                 self.hiddenAttribsLB.insertItem(attr.name)

        self.updateGraph()
    # ################################################

class GroupRadvizOptions(QVGroupBox):
    def __init__(self,parent=None,name=None):
        QVGroupBox.__init__(self, parent, name)
        self.parent = parent

        # ####
        # point width
        widthBox = QHGroupBox("Point Width", self)
        QToolTip.add(widthBox, "The width of points")
        self.widthSlider = QSlider(2, 19, 1, 5, QSlider.Horizontal, widthBox)
        self.widthSlider.setTickmarks(QSlider.Below)
        self.widthLCD = QLCDNumber(2, widthBox)

        # ####
        # scale point position
        self.positionBox = QHGroupBox("Point scaling", self)
        self.scaleBox= QHBox(self.positionBox, "scale")
        self.scaleLabel = QLabel("Scale point position by: ", self.scaleBox)
        self.scaleCombo = QComboBox(self.scaleBox)
        

        # ####
        # jittering
        """
        self.spreadButtons = QVButtonGroup("Jittering type", self)
        QToolTip.add(self.spreadButtons, "Selected the type of jittering for discrete variables")
        self.spreadButtons.setExclusive(TRUE)
        self.spreadNone = QRadioButton('none', self.spreadButtons)
        self.spreadUniform = QRadioButton('uniform', self.spreadButtons)
        self.spreadTriangle = QRadioButton('triangle', self.spreadButtons)
        self.spreadBeta = QRadioButton('beta', self.spreadButtons)
        """

        # #####
        # jittering size
        self.jitteringOptionsBG = QVButtonGroup("Jittering options for discrete attribute", self)
        QToolTip.add(self.jitteringOptionsBG, "Percents of a discrete value to be jittered")
        self.hbox = QHBox(self.jitteringOptionsBG, "jittering size")
        self.jitterLabel = QLabel('Jittering size (%)', self.hbox)
        self.jitterSize = QComboBox(self.hbox)

        # #####
        # general settings
        self.graphSettingsBG = QVButtonGroup("General graph settings", self)
        self.useEnhancedTooltips = QCheckBox("Use enhanced tooltips", self.graphSettingsBG)
        self.globalValueScaling  = QCheckBox("Use global value scaling", self.graphSettingsBG)
        self.showFilledSymbols   = QCheckBox('Show filled symbols', self.graphSettingsBG)
        self.showLegend = QCheckBox('Show legend', self.graphSettingsBG)
        self.optimizedDrawing = QCheckBox('Optimize drawing (biased)', self.graphSettingsBG)
        self.differentSymbols = QCheckBox("Use different symbols", self.graphSettingsBG)

        self.sendingSelectionsBG = QVButtonGroup("Sending selections", self)
        self.autoSendSelection = QCheckBox("Auto send selected data", self.sendingSelectionsBG)
        self.sendShownAttributes = QCheckBox("Send only shown attributes", self.sendingSelectionsBG)

        # ####
        #self.gSetCanvasColorB = QPushButton("Canvas Color", self)
        #self.connect(self.widthSlider, SIGNAL("valueChanged(int)"), self.widthLCD, SLOT("display(int)"))
        #self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)

    def setGraphCanvasColor(self):
        newColor = QColorDialog.getColor(QColor(self.parent.graphCanvasColor))
        if newColor.isValid():
            self.parent.graphCanvasColor = str(newColor.name())
            self.parent.graph.setCanvasColor(QColor(newColor))


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRadviz()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
