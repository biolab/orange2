"""
<name>Scatterplot</name>
<description>Shows data using scatterplot</description>
<category>Visualization</category>
<icon>icons/ScatterPlot.png</icon>
<priority>100</priority>
"""
# ScatterPlot.py
#
# Show data using scatterplot
# 

from OWWidget import *
from OWScatterPlotGraph import *
from OWkNNOptimization import *
import OWGUI
import OWToolbars

###########################################################################################
##### WIDGET : Scatterplot visualization
###########################################################################################
class OWScatterPlot(OWWidget):
    settingsList = ["pointWidth", "showXAxisTitle",
                    "showYAxisTitle", "showVerticalGridlines", "showHorizontalGridlines",
                    "showLegend", "graphGridColor", "graphCanvasColor", "jitterSize", "jitterContinuous", "showFilledSymbols", "showDistributions", "autoSendSelection"]
    jitterSizeList = ['0.0', '0.1','0.5','1','2','3','4','5','7', '10', '15', '20', '30', '40', '50']
    jitterSizeNums = [0.0, 0.1,   0.5,  1,  2 , 3,  4 , 5 , 7 ,  10,   15,   20 ,  30 ,  40 ,  50 ]

    def __init__(self, parent=None):
        #OWWidget.__init__(self, parent, "ScatterPlot", "Show data using scatterplot", TRUE, TRUE)
        apply(OWWidget.__init__, (self, parent, "ScatterPlot", "Show data using scatterplot", FALSE, TRUE))

        self.inputs = [("Examples", ExampleTable, self.cdata, 1), ("Attribute selection", list, self.attributeSelection, 1)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass)]

        #set default settings
        self.pointWidth = 5
        self.showXAxisTitle = 1
        self.showYAxisTitle = 1
        self.showVerticalGridlines = 0
        self.showHorizontalGridlines = 0
        self.showLegend = 1
        self.showDistributions = 0
        
        self.jitterContinuous = 0
        self.jitterSize = 5

        self.showFilledSymbols = 1
        self.autoSendSelection = 0

        self.graphCanvasColor = str(Qt.white.name())
        self.graphGridColor = str(Qt.black.name())
        self.data = None

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")

        #add a graph widget
        self.graph = OWScatterPlotGraph(self, self.mainArea)
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        #x attribute
        self.attrX = ""
        self.attrXCombo = OWGUI.comboBox(self.GeneralTab, self, "attrX", " X axis attribute ", callback = self.removeSelectionsAndUpdateGraph, sendString = 1)

        # y attribute
        self.attrY = ""
        self.attrYCombo = OWGUI.comboBox(self.GeneralTab, self, "attrY", " Y axis attribute ", callback = self.removeSelectionsAndUpdateGraph, sendString = 1)

        # coloring
        self.showColorLegend = 0
        self.attrColor = ""
        box = OWGUI.widgetBox(self.GeneralTab, " Color attribute")
        OWGUI.checkBox(box, self, 'showColorLegend', 'Show color legend', callback = self.updateGraph)
        self.attrColorCombo = OWGUI.comboBox(box, self, "attrColor", callback = self.updateGraph, sendString=1)
        
        # shaping
        self.attrShape = ""
        self.attrShapeCombo = OWGUI.comboBox(self.GeneralTab, self, "attrShape", " Shape attribute ", callback = self.updateGraph, sendString=1)
                
        # sizing
        self.attrSize = ""
        self.attrSizeCombo = OWGUI.comboBox(self.GeneralTab, self, "attrSize", " Size attribute ", callback = self.updateGraph, sendString=1)
        
        # optimization
        self.optimizationDlg = kNNOptimization(None)
        self.optimizationDlg.parentName = "ScatterPlot"
        self.optimizationDlg.optimizeAllSubsetSeparationButton.setEnabled(0)
        self.optimizationDlg.maxLenCombo.setEnabled(0)
        self.optimizationDlg.exactlyLenCombo.setEnabled(0)
        self.optimizationDlg.optimizeSeparationButton.setText("Optimize separation")
        self.graph.kNNOptimization = self.optimizationDlg
        
        self.optimizationDlgButton = OWGUI.button(self.GeneralTab, self, "VizRank optimization dialog", callback = self.optimizationDlg.reshow)
        
        # zooming / selection
        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        # ####################################
        #K-NN OPTIMIZATION functionality
        self.connect(self.optimizationDlg.reevaluateResults, SIGNAL("clicked()"), self.testCurrentProjections)
        self.connect(self.optimizationDlg.evaluateProjectionButton, SIGNAL("clicked()"), self.evaluateCurrentProjection)
        self.optimizationDlg.saveProjectionButton.setEnabled(0)
        self.connect(self.optimizationDlg.showKNNCorrectButton, SIGNAL("clicked()"), self.showKNNCorect)
        self.connect(self.optimizationDlg.showKNNWrongButton, SIGNAL("clicked()"), self.showKNNWrong)
        self.connect(self.optimizationDlg.showKNNResetButton, SIGNAL("clicked()"), self.updateGraph)

        self.connect(self.optimizationDlgButton, SIGNAL("clicked()"), self.optimizationDlg.reshow)
        self.connect(self.optimizationDlg.interestingList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        
        self.connect(self.optimizationDlg.optimizeSeparationButton, SIGNAL("clicked()"), self.optimizeSeparation)

        # ####################################
        # SETTINGS TAB

        # #####
        # point width
        OWGUI.hSlider(self.SettingsTab, self, 'pointWidth', box='Point Width', minValue=1, maxValue=20, step=1, callback=self.setPointWidth, ticks=1)

        # #####
        # jittering options
        box = OWGUI.widgetBox(self.SettingsTab, " Jittering options ")
        box2 = OWGUI.widgetBox(box, orientation = "horizontal")
        self.jitterLabel = QLabel('Jittering size (% of size)  ', box2)
        self.jitterSizeCombo = OWGUI.comboBox(box2, self, "jitterSize", callback = self.setJitterSize, items = self.jitterSizeList)
        OWGUI.checkBox(box, self, 'jitterContinuous', 'Jitter continuous attributes', callback = self.setJitterCont)
        

        # ####
        # general graph settings
        box = OWGUI.widgetBox(self.SettingsTab, " General graph settings ")
        OWGUI.checkBox(box, self, 'showXAxisTitle', 'X axis title', callback = self.updateGraph)
        OWGUI.checkBox(box, self, 'showYAxisTitle', 'Y axis title', callback = self.updateGraph)
        OWGUI.checkBox(box, self, 'showVerticalGridlines', 'Vertical gridlines', callback = self.setVerticalGridlines)
        OWGUI.checkBox(box, self, 'showHorizontalGridlines', 'Horizontal gridlines', callback = self.setHorizontalGridlines)
        OWGUI.checkBox(box, self, 'showLegend', 'Show legend', callback = self.setShowLegend)
        OWGUI.checkBox(box, self, 'showDistributions', 'Show distributions', callback = self.updateGraph)
        OWGUI.checkBox(box, self, 'showFilledSymbols', 'Show filled symbols', callback = self.setFilledSymbols)

        OWGUI.checkBox(self.SettingsTab, self, 'autoSendSelection', 'Auto send selected data', box = "Data selection", callback = self.setAutoSendSelection)
        self.graph.autoSendSelectionCallback = self.setAutoSendSelection
        
        self.gSetGridColorB = QPushButton("Grid Color", self.SettingsTab)
        self.gSetCanvasColorB = QPushButton("Canvas Color", self.SettingsTab)
        
        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)

        self.activateLoadedSettings()
        self.resize(900, 700)


    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        """
        #self.SettingsTab.jitteringButtons.setButton(self.spreadType.index(self.jitteringType))
        self.SettingsTab.gSetXaxisCB.setChecked(self.showXAxisTitle)
        self.SettingsTab.gSetYaxisCB.setChecked(self.showYAxisTitle)
        self.SettingsTab.gSetVgridCB.setChecked(self.showVerticalGridlines)
        self.SettingsTab.gSetHgridCB.setChecked(self.showHorizontalGridlines)
        self.SettingsTab.gSetLegendCB.setChecked(self.showLegend)
        self.SettingsTab.gShowFilledSymbolsCB.setChecked(self.showFilledSymbols)
        self.SettingsTab.showDistributionsCB.setChecked(self.showDistributions)
        self.SettingsTab.jitterContinuous.setChecked(self.jitterContinuous)
        self.SettingsTab.autoSendSelection.setChecked(self.autoSendSelection)
        self.setAutoSendSelection() # update send button state
        
        self.SettingsTab.jitterSize.clear()
        for i in range(len(self.jitterSizeList)):
            self.SettingsTab.jitterSize.insertItem(self.jitterSizeList[i])
        self.SettingsTab.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        self.SettingsTab.widthSlider.setValue(self.pointWidth)
        self.SettingsTab.widthLCD.display(self.pointWidth)
        """

        self.graph.setPointWidth(self.pointWidth)
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setShowXaxisTitle(self.showXAxisTitle)
        self.graph.setShowYLaxisTitle(self.showYAxisTitle)
        self.graph.enableGridXB(self.showVerticalGridlines)
        self.graph.enableGridYL(self.showHorizontalGridlines)
        self.graph.updateSettings(enabledLegend = self.showLegend)
        
        self.graph.updateSettings(showDistributions = self.showDistributions)
        self.graph.updateSettings(jitterContinuous = self.jitterContinuous)
        self.graph.updateSettings(showFilledSymbols = self.showFilledSymbols)
        
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        self.graph.setGridPen(QPen(QColor(self.graphGridColor)))


    def setVerticalGridlines(self):
        self.graph.enableGridXB(self.showVerticalGridlines)

    def setHorizontalGridlines(self):
        self.graph.enableGridYL(self.showHorizontalGridlines)

    def setJitterCont(self):
        self.graph.updateSettings(jitterContinuous = self.jitterContinuous)
        self.updateGraph()

    def setJitterSize(self):
        self.graph.setJitterSize(self.jitterSizeNums[self.jitterSize])
        self.updateGraph()

    def setFilledSymbols(self):
        self.graph.updateSettings(showFilledSymbols = self.showFilledSymbols)
        self.updateGraph()

    def setPointWidth(self):
        self.graph.setPointWidth(self.pointWidth)
        self.updateGraph()

    def setShowLegend(self):
        self.graph.updateSettings(enabledLegend = self.showLegend)
        self.updateGraph()

    def setAutoSendSelection(self):
        if self.autoSendSelection:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(0)
            self.sendSelections()
        else:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(1)
            

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        (selected, unselected, merged) = self.graph.getSelectionsAsExampleTables(self.attrX, self.attrY)
        self.send("Selected Examples",selected)
        self.send("Unselected Examples",unselected)
        self.send("Example Distribution", merged)



    # #######################################################################################################
    # KNN OPTIMIZATION BUTTON EVENTS
    # #######################################################################################################
   
    # evaluate knn accuracy on current projection
    def evaluateCurrentProjection(self):
        acc = self.graph.getProjectionQuality(self.attrX, self.attrY, self.attrColor)
        if self.data.domain.classVar.varType == orange.VarTypes.Continuous:
            QMessageBox.information( None, "Scatterplot", 'Mean square error of kNN model is %.2f'%(acc), QMessageBox.Ok + QMessageBox.Default)
        else:
            if self.optimizationDlg.getQualityMeasure() == CLASS_ACCURACY:
                QMessageBox.information( None, "Scatterplot", 'Classification accuracy of kNN model is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
            elif self.optimizationDlg.getQualityMeasure() == AVERAGE_CORRECT:
                QMessageBox.information( None, "Scatterplot", 'Average probability of correct classification is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
            else:
                QMessageBox.information( None, "Scatterplot", 'Brier score of kNN model is %.2f' % (acc), QMessageBox.Ok + QMessageBox.Default)
         
    # show quality of knn model by coloring accurate predictions with darker color and bad predictions with light color        
    def showKNNCorect(self):
        self.graph.updateData(self.attrX, self.attrY, "", "", "", 0, self.statusBar, showKNNModel = 1, showCorrect = 1)
        self.graph.update()
        self.repaint()

    # show quality of knn model by coloring accurate predictions with lighter color and bad predictions with dark color
    def showKNNWrong(self):
        self.graph.updateData(self.attrX, self.attrY, "", "", "", 0, self.statusBar, showKNNModel = 1, showCorrect = 0)
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
                sumAcc += self.graph.getProjectionQuality(list[0], list[1], self.data.domain.classVar.name)
            results.append((sumAcc/float(len(kList)), tableLen, list))
            self.progressBarSet(100*i/float(count))

        self.optimizationDlg.kValue = oldKValue
        self.optimizationDlg.clear()
        while results != []:
            (accuracy, tableLen, list) = max(results)
            self.optimizationDlg.insertItem(accuracy, tableLen, list)  
            results.remove((accuracy, tableLen, list))

        self.optimizationDlg.updateNewResults()
        #self.optimizationDlg.save("temp.proj")
        self.optimizationDlg.interestingList.setCurrentItem(0)

        self.progressBarFinished()
        self.optimizationDlg.enableControls()


    # ####################################
    # find optimal class separation for shown attributes
    def optimizeSeparation(self):
        if self.data != None:
            self.progressBarInit()
            self.optimizationDlg.disableControls()
        
            self.graph.percentDataUsed = self.optimizationDlg.percentDataUsed
            fullList = self.graph.getOptimalSeparation(None, self.data.domain.classVar.name)
            if fullList == []: return

            # fill the "interesting visualizations" list box
            self.optimizationDlg.clear()

            if self.data.domain.classVar.varType == orange.VarTypes.Discrete and self.optimizationDlg.getQualityMeasure() != BRIER_SCORE: funct = max
            else: funct = min
            while fullList != []:
                (accuracy, tableLen, list) = funct(fullList)
                self.optimizationDlg.insertItem(accuracy, tableLen, list)  
                fullList.remove((accuracy, tableLen, list))
                
            self.optimizationDlg.updateNewResults()
            self.optimizationDlg.interestingList.setCurrentItem(0)

            self.progressBarFinished()
            self.optimizationDlg.enableControls()

    #update status on progress bar - gets called by OWScatterplotGraph
    def updateProgress(self, current, total):
        self.progressBar.setTotalSteps(total)
        self.progressBar.setProgress(current)

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

        self.attrX = list[0]
        self.attrY = list[1]
        if len(list)>2: self.attrShape = list[2]
        else: self.attrShape = "(One shape)"
        if len(list)>3: self.attrSize = list[3]
        else: self.attrSize = "(One size)"
        self.attrColor = self.data.domain.classVar.name
        
        self.updateGraph()

        
    # #############################
    # ATTRIBUTE SELECTION
    # #############################
    def initAttrValues(self):
        self.attrXCombo.clear()
        self.attrYCombo.clear()
        self.attrColorCombo.clear()
        self.attrShapeCombo.clear()
        self.attrSizeCombo.clear()

        if self.data == None: return

        self.attrColorCombo.insertItem("(One color)")
        self.attrShapeCombo.insertItem("(One shape)")
        self.attrSizeCombo.insertItem("(One size)")

        contList = []
        discList = []
        for attr in self.data.domain:
            self.attrXCombo.insertItem(attr.name)
            self.attrYCombo.insertItem(attr.name)
            self.attrColorCombo.insertItem(attr.name)
            self.attrSizeCombo.insertItem(attr.name)
            if attr.varType == orange.VarTypes.Discrete: self.attrShapeCombo.insertItem(attr.name)


        self.attrX = str(self.attrXCombo.text(0))
        if self.attrYCombo.count() > 1: self.attrY = str(self.attrYCombo.text(1))
        else:                           self.attrY = str(self.attrYCombo.text(0))
            
        if self.data.domain.classVar: self.attrColor = self.data.domain.classVar.name
        else:                         self.attrColor = "(One color)"
        self.attrShape = "(One shape)"
        self.attrSize= "(One size)"
        
    # set text to "text" in combo box "combo"
    def setText(self, combo, text):
        for i in range(combo.count()):
            if str(combo.text(i)) == text:
                combo.setCurrentItem(i)
                return 1
        return 0

    def removeSelectionsAndUpdateGraph(self, *args):
        self.graph.removeAllSelections()
        self.updateGraph()

    def updateGraph(self, *args):
        self.graph.updateData(self.attrX, self.attrY, self.attrColor, self.attrShape, self.attrSize, self.showColorLegend, self.statusBar)
        self.graph.update()
        #self.graph.replot()

    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        self.optimizationDlg.clear()
        exData = self.data
        self.data = None
        if data: self.data = orange.Preprocessor_dropMissingClasses(data)
        self.graph.setData(self.data)
       
        if not (self.data and exData and str(exData.domain.attributes) == str(self.data.domain.attributes)): # preserve attribute choice if the domain is the same
            self.initAttrValues()
        
        self.updateGraph()
       
    #################################################

    ####### VIEW ################################
    # receive information about which attributes we want to show on x and y axis
    def attributeSelection(self, list):
        if not self.data or len(list) < 2: return

        self.attrX = list[0]
        self.attrY = list[1]
        self.updateGraph()       
    #################################################



#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlot()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
