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
from OWkNNOptimization import *
import time
import OWToolbars
import OWGUI

###########################################################################################
##### WIDGET : Radviz visualization
###########################################################################################
class OWRadviz(OWWidget):
    #spreadType=["none","uniform","triangle","beta"]
    settingsList = ["pointWidth", "jitterSize", "graphCanvasColor", "globalValueScaling", "showFilledSymbols", "scaleFactor", "showLegend", "optimizedDrawing", "useDifferentSymbols", "autoSendSelection", "useDifferentColors", "tooltipKind", "tooltipValue", "toolbarSelection"]
    jitterSizeNums = [0.0, 0.01, 0.1,   0.5,  1,  2 , 3,  4 , 5, 7, 10, 15, 20]
    jitterSizeList = [str(x) for x in jitterSizeNums]
    scaleFactorNums = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    scaleFactorList = [str(x) for x in scaleFactorNums]
        
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Radviz", "Show data using Radviz visualization method", FALSE, TRUE, icon = "Radviz.png")

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata), ("Example Subset", ExampleTable, self.subsetdata, 1, 1), ("Selection", list, self.selection)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass), ("Attribute Selection List", AttributeList)]

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWRadvizGraph(self, self.mainArea)
        self.box.addWidget(self.graph)
        self.optimizationDlg = kNNOptimization(None, self.graph)

        self.pointWidth = 4

        self.globalValueScaling = 0
        self.jitterSize = 1
        self.jitterContinuous = 0
        self.scaleFactor = 1.0
        self.showLegend = 1
        self.showFilledSymbols = 1
        self.optimizedDrawing = 1
        self.useDifferentSymbols = 0
        self.useDifferentColors = 1
        self.autoSendSelection = 1
        self.tooltipKind = 0
        self.tooltipValue = 0
        self.graphCanvasColor = str(Qt.white.name())
        self.data = None
        self.toolbarSelection = 0

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
        
        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection)
        self.graph.autoSendSelectionCallback = self.setAutoSendSelection
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)
                               
        self.hbox2 = QHBox(self.shownAttribsGroup)
        self.buttonUPAttr = QPushButton("Attr UP", self.hbox2)
        self.buttonDOWNAttr = QPushButton("Attr DOWN", self.hbox2)

        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        # ####################################
        # SETTINGS TAB
        # #####
        OWGUI.hSlider(self.SettingsTab, self, 'pointWidth', box='Point Width', minValue=1, maxValue=15, step=1, callback = self.updateValues, ticks=1)

        box = OWGUI.widgetBox(self.SettingsTab, " Jittering options ")
        OWGUI.comboBoxWithCaption(box, self, "jitterSize", 'Jittering size (% of size)  ', callback = self.setJitteringSize, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)
        OWGUI.checkBox(box, self, 'jitterContinuous', 'Jitter continuous attributes', callback = self.setJitterCont, tooltip = "Does jittering apply also on continuous attributes?")
        OWGUI.comboBoxWithCaption(self.SettingsTab, self, "scaleFactor", 'Scale point position by: ', box = " Point scaling ", callback = self.updateValues, items = self.scaleFactorNums, sendSelectedValue = 1, valueType = float)

        box3 = OWGUI.widgetBox(self.SettingsTab, " General graph settings ")
        
        OWGUI.checkBox(box3, self, 'showLegend', 'Show legend', callback = self.updateValues)
        OWGUI.checkBox(box3, self, 'globalValueScaling', 'Use global value scaling', callback = self.setGlobalValueScaling)
        OWGUI.checkBox(box3, self, 'optimizedDrawing', 'Optimize drawing (biased)', callback = self.updateValues, tooltip = "Speed up drawing by drawing all point belonging to one class value at once")
        OWGUI.checkBox(box3, self, 'useDifferentSymbols', 'Use different symbols', callback = self.updateValues, tooltip = "Show different class values using different symbols")
        OWGUI.checkBox(box3, self, 'useDifferentColors', 'Use different colors', callback = self.updateValues, tooltip = "Show different class values using different colors")
        OWGUI.checkBox(box3, self, 'showFilledSymbols', 'Show filled symbols', callback = self.updateValues)

        box2 = OWGUI.widgetBox(self.SettingsTab, " Tooltips settings ")
        OWGUI.comboBox(box2, self, "tooltipKind", items = ["Show line tooltips", "Show visible attributes", "Show all attributes"], callback = self.updateValues)
        OWGUI.comboBox(box2, self, "tooltipValue", items = ["Tooltips show data values", "Tooltips show spring values"], callback = self.updateValues, tooltip = "Do you wish that tooltips would show you original values of visualized attributes or the 'spring' values (values between 0 and 1). \nSpring values are scaled values that are used for determining the position of shown points. Observing these values will therefore enable you to \nunderstand why the points are placed where they are.")

        box4 = OWGUI.widgetBox(self.SettingsTab, " Sending selection ")
        OWGUI.checkBox(box4, self, 'autoSendSelection', 'Auto send selected data', callback = self.setAutoSendSelection, tooltip = "Send signals with selected data whenever the selection changes.")
        self.setAutoSendSelection()

        # ####
        self.gSetCanvasColorB = QPushButton("Canvas Color", self.SettingsTab)
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)


        # ####################################
        # K-NN OPTIMIZATION functionality
        self.optimizationDlg.parentName = "Radviz"
        self.graph.kNNOptimization = self.optimizationDlg
        
        self.connect(self.optimizationDlg.resultList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        self.connect(self.optimizationDlg.startOptimizationButton , SIGNAL("clicked()"), self.startOptimization)
        self.connect(self.optimizationDlg.reevaluateResults, SIGNAL("clicked()"), self.reevaluateProjections)

        self.connect(self.optimizationDlg.evaluateProjectionButton, SIGNAL("clicked()"), self.evaluateCurrentProjection)
        #self.connect(self.optimizationDlg.saveProjectionButton, SIGNAL("clicked()"), self.saveCurrentProjection)
        self.connect(self.optimizationDlg.showKNNCorrectButton, SIGNAL("clicked()"), self.showKNNCorect)
        self.connect(self.optimizationDlg.showKNNWrongButton, SIGNAL("clicked()"), self.showKNNWrong)
        self.connect(self.optimizationDlg.showKNNResetButton, SIGNAL("clicked()"), self.updateGraph)

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
        self.graph.updateSettings(showLegend = self.showLegend, showFilledSymbols = self.showFilledSymbols, optimizedDrawing = self.optimizedDrawing, tooltipValue = self.tooltipValue, tooltipKind = self.tooltipKind)
        self.graph.useDifferentSymbols = self.useDifferentSymbols
        self.graph.useDifferentColors = self.useDifferentColors
        self.graph.pointWidth = self.pointWidth
        self.graph.globalValueScaling = self.globalValueScaling
        self.graph.jitterSize = self.jitterSize
        self.graph.scaleFactor = self.scaleFactor
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        apply([self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection, self.zoomSelectToolbar.actionPolygonSelection][self.toolbarSelection], [])
        

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
        acc, other_results = self.graph.getProjectionQuality(self.getShownAttributeList())
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
    def reevaluateProjections(self):
        results = list(self.optimizationDlg.getShownResults())
        self.optimizationDlg.clearResults()

        self.progressBarInit()
        self.optimizationDlg.disableControls()

        testIndex = 0
        for (acc, tableLen, other, attrList, strList) in results:
            if self.optimizationDlg.isOptimizationCanceled(): continue
            testIndex += 1
            self.progressBarSet(100.0*testIndex/float(len(results)))

            accuracy, other_results = self.graph.getProjectionQuality(attrList)            
            self.optimizationDlg.addResult(accuracy, other_results, tableLen, attrList, strList)

        self.progressBarFinished()
        self.optimizationDlg.enableControls()
        self.optimizationDlg.finishedAddingResults()
        

    def startOptimization(self):
        if self.data == None: return
        listOfAttributes = self.optimizationDlg.getEvaluatedAttributes(self.data)

        text = str(self.optimizationDlg.attributeCountCombo.currentText())
        if text == "ALL": maxLen = len(listOfAttributes)
        else:             maxLen = int(text)
        
        if self.optimizationDlg.getOptimizationType() == self.optimizationDlg.EXACT_NUMBER_OF_ATTRS:
            minLen = maxLen
        else:
            minLen = 3

        self.optimizationDlg.clearResults()

        possibilities = 0
        for i in range(minLen, maxLen+1):
            possibilities += combinations(i, len(listOfAttributes))*fact(i-1)/2
            
        self.graph.totalPossibilities = possibilities
        self.graph.triedPossibilities = 0
    
        if self.graph.totalPossibilities > 20000:
            proj = str(self.graph.totalPossibilities)
            l = len(proj)
            for i in range(len(proj)-2, 0, -1):
                if (l-i)%3 == 0: proj = proj[:i] + "," + proj[i:]
            self.warning("There are %s possible radviz projections using currently visualized attributes"% (proj))
        
        self.optimizationDlg.disableControls()

        startTime = time.time()
        self.graph.startTime = time.time()

        self.graph.getOptimalSeparation(listOfAttributes, minLen, maxLen, self.optimizationDlg.addResult)

        self.progressBarFinished()
        self.optimizationDlg.enableControls()
        self.optimizationDlg.finishedAddingResults()
    
        secs = time.time() - startTime
        print "----------------------------\nNumber of possible projections: %d\nUsed time: %d min, %d sec" %(possibilities, secs/60, secs%60)


    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        if not self.data: return
        (selected, unselected, merged) = self.graph.getSelectionsAsExampleTables(self.getShownAttributeList())
    
        self.send("Selected Examples",selected)
        self.send("Unselected Examples",unselected)
        self.send("Example Distribution", merged)

    def sendShownAttributes(self):
        attributes = []
        for i in range(self.shownAttribsLB.count()):
            attributes.append(str(self.shownAttribsLB.text(i)))
        self.send("Attribute Selection List", attributes)


    # ####################################
    # show selected interesting projection
    def showSelectedAttributes(self):
        self.graph.removeAllSelections()
        val = self.optimizationDlg.getSelectedProjection()
        if not val: return
        (accuracy, other_results, tableLen, list, strList) = val
        
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
        self.sendShownAttributes()
        
    # ####################
    # LIST BOX FUNCTIONS
    # ####################

    # move selected attribute in "Attribute Order" list one place up
    def moveAttrUP(self):
        self.graph.removeAllSelections()
        for i in range(self.shownAttribsLB.count()):
            if self.shownAttribsLB.isSelected(i) and i != 0:
                text = self.shownAttribsLB.text(i)
                self.shownAttribsLB.removeItem(i)
                self.shownAttribsLB.insertItem(text, i-1)
                self.shownAttribsLB.setSelected(i-1, TRUE)
        self.sendShownAttributes()
        self.updateGraph()

    # move selected attribute in "Attribute Order" list one place down  
    def moveAttrDOWN(self):
        self.graph.removeAllSelections()
        count = self.shownAttribsLB.count()
        for i in range(count-2,-1,-1):
            if self.shownAttribsLB.isSelected(i):
                text = self.shownAttribsLB.text(i)
                self.shownAttribsLB.removeItem(i)
                self.shownAttribsLB.insertItem(text, i+1)
                self.shownAttribsLB.setSelected(i+1, TRUE)
        self.sendShownAttributes()
        self.updateGraph()

    def addAttribute(self):
        self.graph.removeAllSelections()
        count = self.hiddenAttribsLB.count()
        pos   = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.hiddenAttribsLB.isSelected(i):
                text = self.hiddenAttribsLB.text(i)
                self.hiddenAttribsLB.removeItem(i)
                self.shownAttribsLB.insertItem(text, pos)
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
        self.sendShownAttributes()
        self.updateGraph()
        self.graph.replot()

    def removeAttribute(self):
        self.graph.removeAllSelections()
        count = self.shownAttribsLB.count()
        pos   = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.shownAttribsLB.isSelected(i):
                text = self.shownAttribsLB.text(i)
                self.shownAttribsLB.removeItem(i)
                self.hiddenAttribsLB.insertItem(text, pos)
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
        self.sendShownAttributes()
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
        self.optimizationDlg.setData(data)  # set k value to sqrt(n)
        exData = self.data
        self.data = None
        if data: self.data = orange.Preprocessor_dropMissingClasses(data)
        self.graph.setData(self.data)

        if not (data and exData and str(exData.domain.attributes) == str(data.domain.attributes)): # preserve attribute choice if the domain is the same                
            self.shownAttribsLB.clear()
            self.hiddenAttribsLB.clear()
            if data:
                for i in range(len(data.domain.attributes)):
                    if i < 50: self.shownAttribsLB.insertItem(data.domain.attributes[i].name)
                    else: self.hiddenAttribsLB.insertItem(data.domain.attributes[i].name)
                if data.domain.classVar: self.hiddenAttribsLB.insertItem(data.domain.classVar.name)
                
        self.updateGraph()
        self.sendSelections()
        self.sendShownAttributes()

    def subsetdata(self, data):
        self.graph.subsetData = data
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
    def updateValues(self):
        self.graph.updateSettings(optimizedDrawing = self.optimizedDrawing, useDifferentSymbols = self.useDifferentSymbols, useDifferentColors = self.useDifferentColors)
        self.graph.updateSettings(showFilledSymbols = self.showFilledSymbols, tooltipKind = self.tooltipKind, tooltipValue = self.tooltipValue)
        self.graph.updateSettings(showLegend = self.showLegend, pointWidth = self.pointWidth, scaleFactor = self.scaleFactor)
        self.updateGraph()

    def setJitteringSize(self):
        self.graph.jitterSize = self.jitterSize
        self.graph.setData(self.data)
        self.updateGraph()

    def setJitterCont(self):
        self.graph.updateSettings(jitterContinuous = self.jitterContinuous)
        self.graph.setData(self.data)
        self.updateGraph()

    def setGlobalValueScaling(self):
        self.graph.globalValueScaling = self.globalValueScaling
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
    ow=OWRadviz()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
