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
    #spreadType=["none","uniform","triangle","beta"]
    settingsList = ["pointWidth", "jitterSize", "graphCanvasColor", "globalValueScaling", "enhancedTooltips", "showFilledSymbols", "scaleFactor", "showLegend", "optimizedDrawing", "useDifferentSymbols", "autoSendSelection", "sendShownAttributes", "optimizeForPrinting"]
    jitterSizeNums = [0.0, 0.1,   0.5,  1,  2 , 3,  4 , 5, 7, 10, 15, 20]
    jitterSizeList = [str(x) for x in jitterSizeNums]
    scaleFactorNums = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    scaleFactorList = [str(x) for x in scaleFactorNums]
        
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Radviz", "Show data using Radviz visualization method", FALSE, TRUE)

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1), ("Selection", list, self.selection, 1)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass)]

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWRadvizGraph(self, self.mainArea)
        self.box.addWidget(self.graph)
        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
        self.graph.updateSettings(statusBar = self.statusBar)
        self.statusBar.message("")
        self.optimizationDlg = kNNOptimization(None)

        self.pointWidth = 4
        self.enhancedTooltips = 1
        self.globalValueScaling = 0
        self.jitterSize = 1
        self.scaleFactor = 1.0
        self.showLegend = 1
        self.showFilledSymbols = 1
        self.optimizedDrawing = 1
        self.useDifferentSymbols = 0
        self.optimizeForPrinting = 0
        self.autoSendSelection = 0
        self.sendShownAttributes = 1
        self.graphCanvasColor = str(Qt.white.name())
        self.data = None 

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        #self.GeneralTab.setFrameShape(QFrame.NoFrame)
        self.SettingsTab = QVGroupBox(self)
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")

        
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

        self.optimizationDlgButton = OWGUI.button(self.attrOrderingButtons, self, "VizRank optimization dialog", callback = self.optimizationDlg.reshow)
        
        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)
                               
        self.hbox2 = QHBox(self.shownAttribsGroup)
        self.buttonUPAttr = QPushButton("Attr UP", self.hbox2)
        self.buttonDOWNAttr = QPushButton("Attr DOWN", self.hbox2)

        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        # ####################################
        # SETTINGS TAB
        # #####
        OWGUI.hSlider(self.SettingsTab, self, 'pointWidth', box='Point Width', minValue=1, maxValue=15, step=1, callback=self.setPointWidth, ticks=1)
        OWGUI.comboBoxWithCaption(self.SettingsTab, self, "jitterSize", 'Jittering size (% of size)  ', box = " Jittering options ", callback = self.setJitteringSize, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)
        OWGUI.comboBoxWithCaption(self.SettingsTab, self, "scaleFactor", 'Scale point position by: ', box = " Point scaling ", callback = self.setScaleFactor, items = self.scaleFactorNums, sendSelectedValue = 1, valueType = float)

        box2 = OWGUI.widgetBox(self.SettingsTab, " General graph settings ")
        OWGUI.checkBox(box2, self, 'enhancedTooltips', 'Use enhanced tooltips', callback = self.setUseEnhancedTooltips)
        OWGUI.checkBox(box2, self, 'showLegend', 'Show legend', callback = self.setShowLegend)
        OWGUI.checkBox(box2, self, 'globalValueScaling', 'Use global value scaling', callback = self.setGlobalValueScaling)
        OWGUI.checkBox(box2, self, 'optimizedDrawing', 'Optimize drawing (biased)', callback = self.setOptmizedDrawing, tooltip = "Speed up drawing by drawing all point belonging to one class value at once")
        OWGUI.checkBox(box2, self, 'useDifferentSymbols', 'Use different symbols', callback = self.setDifferentSymbols, tooltip = "Show different class values using different symbols")
        OWGUI.checkBox(box2, self, 'showFilledSymbols', 'Show filled symbols', callback = self.setShowFilledSymbols)
        OWGUI.checkBox(box2, self, 'optimizeForPrinting', 'Optimize for printing', callback = self.setOptmizeForPrinting, tooltip = "use symbols that will be printer-friendly")

        box3 = OWGUI.widgetBox(self.SettingsTab, " Sending selection ")
        OWGUI.checkBox(box3, self, 'autoSendSelection', 'Auto send selected data', callback = self.setAutoSendSelection, tooltip = "Send signals with selected data whenever the selection changes.")
        OWGUI.checkBox(box3, self, 'sendShownAttributes', 'Send only shown attributes')

        # ####
        self.gSetCanvasColorB = QPushButton("Canvas Color", self.SettingsTab)
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)


        # ####################################
        #K-NN OPTIMIZATION functionality
        self.optimizationDlg.parentName = "Radviz"
        self.graph.kNNOptimization = self.optimizationDlg
        
        self.connect(self.optimizationDlg.resultList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        self.connect(self.optimizationDlg.startOptimizationButton , SIGNAL("clicked()"), self.startOptimization)
        self.connect(self.optimizationDlg.reevaluateResults, SIGNAL("clicked()"), self.reevaluateProjections)

        self.connect(self.optimizationDlg.evaluateProjectionButton, SIGNAL("clicked()"), self.evaluateCurrentProjection)
        self.connect(self.optimizationDlg.saveProjectionButton, SIGNAL("clicked()"), self.saveCurrentProjection)
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

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.graph.updateSettings(showLegend = self.showLegend, showFilledSymbols = self.showFilledSymbols, optimizedDrawing = self.optimizedDrawing)
        self.graph.enhancedTooltips = self.enhancedTooltips
        self.graph.setPointWidth(self.pointWidth)
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setScaleFactor(self.scaleFactor)
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        self.graph.useDifferentSymbols = self.useDifferentSymbols
        self.graph.optimizeForPrinting = self.optimizeForPrinting

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
        #self.graph.update()
        #self.repaint()

    # show quality of knn model by coloring accurate predictions with lighter color and bad predictions with dark color
    def showKNNWrong(self):
        self.graph.updateData(self.getShownAttributeList(), showKNNModel = 1, showCorrect = 0)
        #self.graph.update()
        #self.repaint()

    # reevaluate projections in result list with different k values
    def reevaluateProjections(self):
        results = list(self.optimizationDlg.getShownResults())
        self.optimizationDlg.clearResults()

        self.progressBarInit()
        self.optimizationDlg.disableControls()

        testIndex = 0
        for (acc, tableLen, attrList, strList) in results:
            testIndex += 1
            self.progressBarSet(100.0*testIndex/float(len(results)))

            accuracy = self.graph.getProjectionQuality(attrList)            
            self.optimizationDlg.addResult(self.data, accuracy, tableLen, attrList, strList)

        self.progressBarFinished()
        self.optimizationDlg.enableControls()
        self.optimizationDlg.finishedAddingResults()
        

    def startOptimization(self):
        if self.optimizationDlg.getOptimizationType() == self.optimizationDlg.EXACT_NUMBER_OF_ATTRS:
            self.optimizeSeparation()
        else:
            self.optimizeAllSubsetSeparation()
            
            
    # ####################################
    # find optimal class separation for shown attributes
    # numberOfAttrs is different than None only when optimizeSeparation is called by optimizeAllSubsetSeparation
    def optimizeSeparation(self, numberOfAttrs = None, listOfAttributes = None):
        if self.data == None: return

        if not listOfAttributes:
            listOfAttributes = self.getShownAttributeList()

        if not numberOfAttrs:
            text = str(self.optimizationDlg.attributeCountCombo.currentText())
            if text == "ALL": number = len(listOfAttributes)
            else:             number = int(text)

            self.optimizationDlg.clearResults()
            total = len(listOfAttributes)
            if total < number: return
            self.graph.totalPossibilities = combinations(number, total)*fact(number-1)/2
            self.graph.triedPossibilities = 0
        
            if self.graph.totalPossibilities > 20000:
                res = QMessageBox.information(self,'Radviz','There are %d possible radviz projections using currently visualized attributes. Since their evaluation will probably take a long time, we suggest \n removing some attributes or decreasing the number of attributes in projections. Do you wish to cancel?' % (self.graph.totalPossibilities),'Yes','No', QString.null,0,1)
                if res == 0: return []
            
            self.progressBarInit()
            self.optimizationDlg.disableControls()

            startTime = time.time()
            self.graph.startTime = time.time()
        else:
            number = numberOfAttrs

        self.graph.getOptimalExactSeparation(listOfAttributes, [], number, self.optimizationDlg.addResult)

        if not numberOfAttrs:
            self.progressBarFinished()
            self.optimizationDlg.enableControls()
            self.optimizationDlg.finishedAddingResults()
        
            secs = time.time() - startTime
            print "Used time: %d min, %d sec" %(secs/60, secs%60)

   
    # #############################################
    # find optimal separation for all possible subsets of shown attributes
    def optimizeAllSubsetSeparation(self):
        if self.data == None: return

        listOfAttributes = self.getShownAttributeList()
        text = str(self.optimizationDlg.attributeCountCombo.currentText())
        if text == "ALL": maxLen = len(listOfAttributes)
        else:             maxLen = int(text)
        total = len(listOfAttributes) 

        # compute the number of possible projections
        proj = 0
        for i in range(3, maxLen+1):
            proj += combinations(i, total)*fact(i-1)/2
        self.graph.triedPossibilities = 0
        self.graph.totalPossibilities = proj
       
        startTime = time.time()
        self.graph.startTime = time.time()
        self.progressBarInit()
        self.optimizationDlg.clearResults()
        self.optimizationDlg.disableControls()

        for val in range(3, maxLen+1):
            self.optimizeSeparation(val, listOfAttributes)

        self.progressBarFinished()
        self.optimizationDlg.enableControls()
        self.optimizationDlg.finishedAddingResults()
        secs = time.time() - startTime
        print "Used time: %d min, %d sec" %(secs/60, secs%60)
            

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
            

    # ####################################
    # show selected interesting projection
    def showSelectedAttributes(self):
        val = self.optimizationDlg.getSelectedProjection()
        if not val: return
        (accuracy, tableLen, list, strList) = val
        
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

    def getShownAttributeList(self):
        list = []
        for i in range(self.shownAttribsLB.count()):
            list.append(str(self.shownAttribsLB.text(i)))
        return list


    # #########################
    # RADVIZ SIGNALS
    # #########################    
    
    # ###### CDATA signal ################################
    # receive new data and update all fields
    def cdata(self, data):
        self.optimizationDlg.clearResults()
        exData = self.data
        self.data = None
        if data: self.data = orange.Preprocessor_dropMissingClasses(data)
        self.graph.setData(self.data)

        if not (data and exData and str(exData.domain.attributes) == str(data.domain.attributes)): # preserve attribute choice if the domain is the same                
            self.shownAttribsLB.clear()
            self.hiddenAttribsLB.clear()
            if data:
                for attr in data.domain.attributes: self.shownAttribsLB.insertItem(attr.name)
                if data.domain.classVar: self.hiddenAttribsLB.insertItem(data.domain.classVar.name)
                
        self.updateGraph()


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

    # #########################
    # RADVIZ EVENTS
    # #########################
    def setJitteringSize(self):
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setData(self.data)
        self.updateGraph()
                
    def setScaleFactor(self):
        self.graph.setScaleFactor(self.scaleFactor)
        self.updateGraph()

    def setPointWidth(self):
        self.graph.setPointWidth(self.pointWidth)
        self.updateGraph()

    def setUseEnhancedTooltips(self):
        self.graph.setEnhancedTooltips(self.enhancedTooltips)
        self.updateGraph()

    def setDifferentSymbols(self):
        self.graph.useDifferentSymbols = self.useDifferentSymbols
        self.updateGraph()

    def setShowFilledSymbols(self):
        self.graph.updateSettings(showFilledSymbols = self.showFilledSymbols)
        self.updateGraph()

    def setOptmizeForPrinting(self):
        self.graph.updateSettings(optimizeForPrinting = self.optimizeForPrinting)
        self.updateGraph()
        
    def setGlobalValueScaling(self):
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setData(self.data)
        self.updateGraph()

    def setShowLegend(self):
        self.graph.updateSettings(showLegend = self.showLegend)
        self.updateGraph()

    def setOptmizedDrawing(self):
        self.graph.updateSettings(optimizedDrawing = self.optimizedDrawing)
        self.updateGraph()

    def setAutoSendSelection(self):
        if self.autoSendSelection:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(0)
            self.sendSelections()
        else:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(1)
        

    def setGraphCanvasColor(self):
        newColor = QColorDialog.getColor(QColor(self.graphCanvasColor))
        if newColor.isValid():
            self.graphCanvasColor = str(newColor.name())
            self.graph.setCanvasColor(QColor(newColor))


    
#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRadviz()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
