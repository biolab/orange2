"""
<name>Polyviz</name>
<description>Shows data using Polyviz visualization method</description>
<category>Visualization</category>
<icon>icons/Polyviz.png</icon>
<priority>3150</priority>
"""
# Polyviz.py
#
# Show data using Polyviz visualization method
# 

from OWWidget import *
from random import betavariate 
from OWPolyvizGraph import *
import OWVisAttrSelection
from OWkNNOptimization import *
import time, math
import OWToolbars

###########################################################################################
##### WIDGET : Polyviz visualization
###########################################################################################
class OWPolyviz(OWWidget):
    #spreadType=["none","uniform","triangle","beta"]
    settingsList = ["pointWidth", "lineLength", "jitterSize", "graphCanvasColor", "globalValueScaling", "enhancedTooltips", "scaleFactor", "showLegend", "showFilledSymbols", "optimizedDrawing", "useDifferentSymbols", "autoSendSelection", "sendShownAttributes", "optimizeForPrinting"]
    jitterSizeNums = [0.0, 0.1,   0.5,  1,  2 , 3,  4 , 5, 7, 10, 15, 20]
    jitterSizeList = [str(x) for x in jitterSizeNums]
    scaleFactorNums = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    scaleFactorList = [str(x) for x in scaleFactorNums]
        
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Polyviz", "Show data using Polyviz visualization method", FALSE, TRUE, icon = "Polyviz.png")

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1), ("Selection", list, self.selection, 1)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass)]

        #set default settings
        self.pointWidth = 5
        self.lineLength = 2
        self.scaleFactor = 1.0
        self.enhancedTooltips = 1
        self.globalValueScaling = 0
        self.jitterSize = 1
        self.attributeReverse = {}  # dictionary with bool values - do we want to reverse attribute values
        self.showLegend = 1
        self.showFilledSymbols = 1
        self.optimizedDrawing = 1
        self.useDifferentSymbols = 0
        self.optimizeForPrinting = 0
        self.autoSendSelection = 0
        self.sendShownAttributes = 1
        self.rotateAttributes = 0
        self.graphCanvasColor = str(Qt.white.name())
        
        self.data = None

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWPolyvizGraph(self, self.mainArea)
        self.box.addWidget(self.graph)
        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
        self.graph.updateSettings(statusBar = self.statusBar)
        self.statusBar.message("")

        #add controls to self.controlArea widget
        self.shownAttribsGroup = OWGUI.widgetBox(self.GeneralTab, " Shown attributes ")
        self.hbox2 = OWGUI.widgetBox(self.GeneralTab, "", orientation = "horizontal")
        self.hiddenAttribsGroup = OWGUI.widgetBox(self.GeneralTab, " Hidden attributes ")
        self.attrOrderingButtons = QVButtonGroup("Attribute ordering", self.GeneralTab)

        self.shownAttribsLB = QListBox(self.shownAttribsGroup)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)
        
        self.optimizationDlgButton = QPushButton('VizRank optimization dialog', self.attrOrderingButtons)
        OWGUI.checkBox(self.attrOrderingButtons, self, "rotateAttributes", "Rotate attributes", tooltip = "When searching for optimal projections also evaluate projections with rotated attributes. \nThis will significantly increase the number of possible projections.")

        self.optimizationDlg = kNNOptimization(None)
        self.optimizationDlg.parentName = "Polyviz"
        self.graph.kNNOptimization = self.optimizationDlg

        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        self.hbox = OWGUI.widgetBox(self.shownAttribsGroup, "", orientation = "horizontal")
        self.buttonUPAttr = QPushButton("Attr UP", self.hbox)
        self.buttonDOWNAttr = QPushButton("Attr DOWN", self.hbox)

        self.attrAddButton = QPushButton("Add attr.", self.hbox2)
        self.attrRemoveButton = QPushButton("Remove attr.", self.hbox2)

        # ####################################
        # SETTINGS TAB
        # #####
        OWGUI.hSlider(self.SettingsTab, self, 'pointWidth', box='Point width', minValue=1, maxValue=15, step=1, callback=self.setPointWidth, ticks=1)
        OWGUI.hSlider(self.SettingsTab, self, 'lineLength', box='Line length', minValue=1, maxValue=5, step=1, callback=self.setLineLength, ticks=1)

        box = OWGUI.widgetBox(self.SettingsTab, " Jittering options ")
        OWGUI.comboBoxWithCaption(box, self, "jitterSize", 'Jittering size (% of size)  ', callback = self.setJitteringSize, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)

        OWGUI.comboBoxWithCaption(self.SettingsTab, self, "scaleFactor", 'Scale point position by: ', box = " Point scaling ", callback = self.setScaleFactor, items = self.scaleFactorNums, sendSelectedValue = 1, valueType = float)

        box2 = OWGUI.widgetBox(self.SettingsTab, " General graph settings ")
        OWGUI.checkBox(box2, self, 'enhancedTooltips', 'Use enhanced tooltips', callback = self.setUseEnhancedTooltips)
        OWGUI.checkBox(box2, self, 'showLegend', 'Show legend', callback = self.setShowLegend)
        OWGUI.checkBox(box2, self, 'globalValueScaling', 'Use global value scaling', callback = self.setGlobalValueScaling, tooltip = "Scale values of all attributes based on min and max value of all attributes. Usually unchecked.")
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
        self.connect(self.optimizationDlgButton, SIGNAL("clicked()"), self.optimizationDlg.reshow)
        self.connect(self.optimizationDlg.resultList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        self.connect(self.optimizationDlg.startOptimizationButton , SIGNAL("clicked()"), self.startOptimization)
        self.connect(self.optimizationDlg.reevaluateResults, SIGNAL("clicked()"), self.reevaluateProjections)
        self.connect(self.optimizationDlg.evaluateProjectionButton, SIGNAL("clicked()"), self.evaluateCurrentProjection)
        self.connect(self.optimizationDlg.saveProjectionButton, SIGNAL("clicked()"), self.saveCurrentProjection)
        self.connect(self.optimizationDlg.showKNNCorrectButton, SIGNAL("clicked()"), self.showKNNCorect)
        self.connect(self.optimizationDlg.showKNNWrongButton, SIGNAL("clicked()"), self.showKNNWrong)
        self.connect(self.optimizationDlg.showKNNResetButton, SIGNAL("clicked()"), self.updateGraph)        
        
        self.connect(self.shownAttribsLB, SIGNAL('doubleClicked(QListBoxItem *)'), self.reverseSelectedAttribute)

        self.connect(self.buttonUPAttr, SIGNAL("clicked()"), self.moveAttrUP)
        self.connect(self.buttonDOWNAttr, SIGNAL("clicked()"), self.moveAttrDOWN)

        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        
        # add a settings dialog and initialize its values
        self.activateLoadedSettings()

        self.resize(900, 700)


    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.graph.updateSettings(showLegend = self.showLegend, showFilledSymbols = self.showFilledSymbols, optimizedDrawing = self.optimizedDrawing)
        self.graph.setEnhancedTooltips(self.enhancedTooltips)
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
        qname = QFileDialog.getSaveFileName( os.path.realpath(".") + "/Polyviz_projection.tab", "Orange Example Table (*.tab)", self, "", "Save File")
        if qname.isEmpty(): return
        name = str(qname)
        if len(name) < 4 or name[-4] != ".":
            name = name + ".tab"

        self.graph.saveProjectionAsTabData(name, self.getShownAttributeList(), self.attributeReverse)

    # evaluate knn accuracy on current projection
    def evaluateCurrentProjection(self):
        acc = self.graph.getProjectionQuality(self.getShownAttributeList(), self.attributeReverse)
        if self.data.domain.classVar.varType == orange.VarTypes.Continuous:
            QMessageBox.information( None, "Polyviz", 'Mean square error of kNN model is %.2f'%(acc), QMessageBox.Ok + QMessageBox.Default)
        else:
            if self.optimizationDlg.getQualityMeasure() == CLASS_ACCURACY:
                QMessageBox.information( None, "Polyviz", 'Classification accuracy of kNN model is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
            elif self.optimizationDlg.getQualityMeasure() == AVERAGE_CORRECT:
                QMessageBox.information( None, "Polyviz", 'Average probability of correct classification is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
            else:
                QMessageBox.information( None, "Polyviz", 'Brier score of kNN model is %.2f' % (acc), QMessageBox.Ok + QMessageBox.Default)
            
    # show quality of knn model by coloring accurate predictions with darker color and bad predictions with light color        
    def showKNNCorect(self):
        self.graph.updateData(self.getShownAttributeList(), self.attributeReverse, showKNNModel = 1, showCorrect = 1)
        #self.repaint()

    # show quality of knn model by coloring accurate predictions with lighter color and bad predictions with dark color
    def showKNNWrong(self):
        self.graph.updateData(self.getShownAttributeList(), self.attributeReverse, showKNNModel = 1, showCorrect = 0)
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

            reverseDict = self.buildOrientationDictFromString(attrList, strList)
            accuracy = self.graph.getProjectionQuality(attrList, reverseDict)
            self.optimizationDlg.addResult(self.data, accuracy, tableLen, attrList, strList)

        self.optimizationDlg.finishedAddingResults()
        self.optimizationDlg.enableControls()
        self.optimizationDlg.resultList.setCurrentItem(0)
        self.progressBarFinished()

    def startOptimization(self):
        print self.optimizationDlg.getOptimizationType()
        if self.optimizationDlg.getOptimizationType() == self.optimizationDlg.EXACT_NUMBER_OF_ATTRS:
            self.optimizeSeparation()
        else:
            self.optimizeAllSubsetSeparation()

    def buildProjections(self, attributes, currentProjection, number, projections):
        if number == 0:
            if len(currentProjection) != 1: projections.append(currentProjection)
            return projections
        if attributes == []: return projections

        temp = list(currentProjection) + [attributes[0][1]]
        temp[0] += attributes[0][0]
        projections = self.buildProjections(attributes[1:], temp, number-1, projections)
        
        projections = self.buildProjections(attributes[1:], currentProjection, number, projections)
        return projections
            

    # ####################################
    # find optimal class separation for shown attributes
    # numberOfAttrs is different than None only when optimizeSeparation is called by optimizeAllSubsetSeparation
    def optimizeSeparation(self, numberOfAttrs = None, listOfAttributes = None):
        if self.data == None: return
    
        if not listOfAttributes:
            listOfAttributes = self.optimizationDlg.getEvaluatedAttributes(self.data)

        if self.rotateAttributes: reverseList = None
        else: reverseList = self.attributeReverse
        
        if not numberOfAttrs:
            text = str(self.optimizationDlg.attributeCountCombo.currentText())
            if text == "ALL": number = len(listOfAttributes)
            else:             number = int(text)
        
            self.optimizationDlg.clearResults()
            total = len(listOfAttributes)
            if total < number: return
            if not self.rotateAttributes: combin = combinations(number, total) * fact(number-1)
            else: combin = combinations(number, total) * fact(number-1) * math.pow(2, number)/2
            self.graph.updateSettings(totalPossibilities = combin, triedPossibilities = 0, startTime = time.time())
        
            if self.graph.totalPossibilities > 20000:
                res = QMessageBox.information(self,'Polyviz','There are %d possible polyviz projections using currently visualized attributes. Since their evaluation will probably take a long time, we suggest removing some attributes or decreasing the number of attributes in projections. Do you wish to cancel?' % (combin),'Yes','No', QString.null,0,1)
                if res == 0: return
            
            self.progressBarInit()
            self.optimizationDlg.disableControls()
            startTime = time.time()
            
        else:
            number = numberOfAttrs

        # create a sorted list of attribute subsets to evaluate
        projections = self.buildProjections(listOfAttributes, [0.0], number, [])
        projections.sort()
        projections.reverse()

        self.graph.getOptimalSeparation(number, reverseList, projections, self.addInterestingProjection)

        if not numberOfAttrs:
            self.progressBarFinished()
            self.optimizationDlg.enableControls()
            self.optimizationDlg.finishedAddingResults()
        
            secs = time.time() - startTime
            print "Number of possible projections: %d\nUsed time: %d min, %d sec" %(combin, secs/60, secs%60)


    # #############################################
    # find optimal separation for all possible subsets of shown attributes
    def optimizeAllSubsetSeparation(self):
        if self.data == None: return

        listOfAttributes = self.optimizationDlg.getEvaluatedAttributes(self.data)
        
        text = str(self.optimizationDlg.attributeCountCombo.currentText())
        if text == "ALL": maxLen = len(listOfAttributes)
        else:              maxLen = int(text)
        total = len(listOfAttributes)

        # compute the number of possible projections
        proj = 0
        for i in range(3, maxLen+1):
            if not self.rotateAttributes: proj += combinations(i, total) * fact(i-1)
            else: proj += combinations(i, total) * fact(i-1) * math.pow(2, i)/2

        if proj > 20000:
            res = QMessageBox.information(self,'Polyviz','There are %d possible polyviz projections using currently visualized attributes. Since their evaluation will probably take a long time, we suggest removing some attributes or decreasing the number of attributes in projections. Do you wish to cancel?' % (combin),'Yes','No', QString.null,0,1)
            if res == 0: return

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
        print "Number of possible projections: %d\nUsed time: %d min, %d sec" %(proj, secs/60, secs%60)

    
    def addInterestingProjection(self, data, accuracy, tableLen, attrList, reverse):
        strList = "["
        for i in range(len(attrList)):
            if reverse[self.graph.attributeNames.index(attrList[i])] == 1:
                strList += attrList[i] + "-, "
            else:
                strList += attrList[i] + "+, "
        strList = strList[:-2] + "]"
        self.optimizationDlg.addResult(data, accuracy, tableLen, attrList, strList)
        

    def reverseSelectedAttribute(self, item):
        text = str(item.text())
        name = text[:-2]
        self.attributeReverse[name] = not self.attributeReverse[name]

        for i in range(self.shownAttribsLB.count()):
            if str(self.shownAttribsLB.item(i).text()) == str(item.text()):
                self.shownAttribsLB.removeItem(i)
                if self.attributeReverse[name] == 1:    self.shownAttribsLB.insertItem(name + ' -', i)
                else:                                   self.shownAttribsLB.insertItem(name + ' +', i)
                self.shownAttribsLB.setCurrentItem(i)
                self.updateGraph()
                return
        
  
    # ####################################
    # show selected interesting projection
    def showSelectedAttributes(self):
        val = self.optimizationDlg.getSelectedProjection()
        if not val: return
        (accuracy, tableLen, list, strList) = val
        
        # check if all attributes in list really exist in domain        
        attrNames = []
        for attr in self.data.domain:
            attrNames.append(attr.name)
        
        for item in list:
            if not item in attrNames:
                print "invalid settings"
                return
        
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        reverseDict = self.buildOrientationDictFromString(list, strList)
        for attr in attrNames:
            if reverseDict.has_key(attr):
                if reverseDict[attr]: self.shownAttribsLB.insertItem(attr + " -")
                else: self.shownAttribsLB.insertItem(attr + " +")
                self.attributeReverse[attr] = reverseDict[attr]
            else:
                self.hiddenAttribsLB.insertItem(attr + " +")
                self.attributeReverse[attr] = 0
        
        self.updateGraph()

    def buildOrientationDictFromString(self, attrList, strList):
        ret = {}
        for attr in attrList:
            if strList.find(attr + "+,") >=0 or strList.find(attr + "+]") >=0:
                ret[attr] = 0
            else:
                ret[attr] = 1
        return ret

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):   
        if not self.data: return
        (selected, unselected, merged) = self.graph.getSelectionsAsExampleTables(self.getShownAttributeList(), self.attributeReverse)
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
        #self.graph.replot()

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
        #self.graph.replot()

    # #####################

    def updateGraph(self, *args):
        self.graph.updateData(self.getShownAttributeList(), self.attributeReverse)
        self.graph.update()
        self.repaint()

        
    def getShownAttributeList (self):
        list = []
        for i in range(self.shownAttribsLB.count()):
            list.append(str(self.shownAttribsLB.text(i))[:-2])
        return list

    ##############################################
    
    
    # ###### CDATA signal ################################
    # receive new data and update all fields
    def cdata(self, data):
        self.optimizationDlg.clearResults()
        self.optimizationDlg.setData(data)  # set k value to sqrt(n)
        exData = self.data
        self.data = None
        if data: self.data = orange.Preprocessor_dropMissingClasses(data)
        self.graph.setData(self.data)

        if not (data and exData and str(exData.domain.attributes) == str(data.domain.attributes)):    # preserve attribute choice if the domain is the same
            self.shownAttribsLB.clear()
            self.hiddenAttribsLB.clear()
            self.attributeReverse = {}
            
            if data:
                for attr in data.domain: self.attributeReverse[attr.name] = 0   # set reverse parameter to 0
                for attr in data.domain.attributes: self.shownAttribsLB.insertItem(attr.name + " +")
                if data.domain.classVar: self.hiddenAttribsLB.insertItem(data.domain.classVar.name + " +")
        
        self.updateGraph()

    ####### SELECTION signal ################################
    # receive info about which attributes to show
    def selection(self, list):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if self.data == None: return

        for attr in self.data.domain:
            if attr.name in list: self.shownAttribsLB.insertItem(attr.name)
            else:                 self.hiddenAttribsLB.insertItem(attr.name)

        self.updateGraph()


    # #########################
    # POLYVIZ EVENTS
    # #########################
    def setPointWidth(self):
        self.graph.setPointWidth(self.pointWidth)
        self.updateGraph()

    def setLineLength(self):
        self.graph.setLineLength(self.lineLength)
        self.updateGraph()

    def setJitteringSize(self):
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setData(self.data)
        self.updateGraph()

    def setScaleFactor(self):
        self.graph.setScaleFactor(self.scaleFactor)
        self.updateGraph()

    def setUseEnhancedTooltips(self):
        self.graph.setEnhancedTooltips(self.enhancedTooltips)
        self.updateGraph()

    def setShowFilledSymbols(self):
        self.graph.updateSettings(showFilledSymbols = self.showFilledSymbols)
        self.updateGraph()

    def setDifferentSymbols(self):
        self.graph.useDifferentSymbols = self.useDifferentSymbols
        self.updateGraph()

    def setShowLegend(self):
        self.graph.updateSettings(showLegend = self.showLegend)
        self.updateGraph()

    def setOptmizedDrawing(self):
        self.graph.updateSettings(optimizedDrawing = self.optimizedDrawing)
        self.updateGraph()

    def setOptmizeForPrinting(self):
        self.graph.updateSettings(optimizeForPrinting = self.optimizeForPrinting)
        self.updateGraph()


    def setGlobalValueScaling(self):
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setData(self.data)
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
    ow=OWPolyviz()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
