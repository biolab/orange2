"""
<name>Scatterplot</name>
<description>Shows data using scatterplot</description>
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
                    "showLegend", "graphGridColor", "graphCanvasColor", "jitterSize", "jitterContinuous", "showFilledSymbols", "showDistributions", "autoSendSelection", "optimizedDrawing", "toolbarSelection"]
    jitterSizeList = ['0.0', '0.1','0.5','1','2','3','4','5','7', '10', '15', '20', '30', '40', '50']
    jitterSizeNums = [0.0, 0.1,   0.5,  1,  2 , 3,  4 , 5 , 7 ,  10,   15,   20 ,  30 ,  40 ,  50 ]

    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, "ScatterPlot", TRUE)

        self.inputs = [("Examples", ExampleTable, self.cdata), ("Example Subset", ExampleTable, self.subsetdata, 1, 1), ("Attribute selection", list, self.attributeSelection)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass)]

        #set default settings
        self.pointWidth = 5
        self.showXAxisTitle = 1
        self.showYAxisTitle = 1
        self.showVerticalGridlines = 0
        self.showHorizontalGridlines = 0
        self.showLegend = 1
        self.showDistributions = 0
        self.optimizedDrawing = 1
        self.tooltipKind = 1
        self.toolbarSelection = 0
        
        self.jitterContinuous = 0
        self.jitterSize = 5

        self.showFilledSymbols = 1
        self.autoSendSelection = 1

        self.graphCanvasColor = str(Qt.white.name())
        self.graphGridColor = str(Qt.black.name())
        self.data = None

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = QTabWidget(self.controlArea, 'tabWidget')
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
        self.attrXCombo = OWGUI.comboBox(self.GeneralTab, self, "attrX", " X axis attribute ", callback = self.removeSelectionsAndUpdateGraph, sendSelectedValue = 1, valueType = str)

        # y attribute
        self.attrY = ""
        self.attrYCombo = OWGUI.comboBox(self.GeneralTab, self, "attrY", " Y axis attribute ", callback = self.removeSelectionsAndUpdateGraph, sendSelectedValue = 1, valueType = str)

        # coloring
        self.showColorLegend = 0
        self.attrColor = ""
        box = OWGUI.widgetBox(self.GeneralTab, " Color attribute")
        OWGUI.checkBox(box, self, 'showColorLegend', 'Show color legend', callback = self.updateGraph)
        self.attrColorCombo = OWGUI.comboBox(box, self, "attrColor", callback = self.updateGraph, sendSelectedValue=1, valueType = str)
        
        # shaping
        self.attrShape = ""
        self.attrShapeCombo = OWGUI.comboBox(self.GeneralTab, self, "attrShape", " Shape attribute ", callback = self.updateGraph, sendSelectedValue=1, valueType = str)
                
        # sizing
        self.attrSize = ""
        self.attrSizeCombo = OWGUI.comboBox(self.GeneralTab, self, "attrSize", " Size attribute ", callback = self.updateGraph, sendSelectedValue=1, valueType = str)
        
        # optimization
        self.optimizationDlg = kNNOptimization(None, self.graph)
        self.optimizationDlg.parentName = "ScatterPlot"
        self.optimizationDlg.label1.hide()
        self.optimizationDlg.optimizationTypeCombo.hide()
        self.optimizationDlg.attributeCountCombo.hide()
        self.optimizationDlg.attributeLabel.hide()
        self.graph.kNNOptimization = self.optimizationDlg
        
        self.optimizationDlgButton = OWGUI.button(self.GeneralTab, self, "VizRank optimization dialog", callback = self.optimizationDlg.reshow)
        
        # zooming / selection
        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        # ####################################
        #K-NN OPTIMIZATION functionality
        self.connect(self.optimizationDlg.reevaluateResults, SIGNAL("clicked()"), self.reevaluateProjections)
        self.connect(self.optimizationDlg.evaluateProjectionButton, SIGNAL("clicked()"), self.evaluateCurrentProjection)
        self.connect(self.optimizationDlg.showKNNCorrectButton, SIGNAL("clicked()"), self.showKNNCorect)
        self.connect(self.optimizationDlg.showKNNWrongButton, SIGNAL("clicked()"), self.showKNNWrong)
        self.connect(self.optimizationDlg.showKNNResetButton, SIGNAL("clicked()"), self.updateGraph)

        self.connect(self.optimizationDlgButton, SIGNAL("clicked()"), self.optimizationDlg.reshow)
        self.connect(self.optimizationDlg.resultList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        
        self.connect(self.optimizationDlg.startOptimizationButton , SIGNAL("clicked()"), self.optimizeSeparation)

        # ####################################
        # SETTINGS TAB

        # point width
        OWGUI.hSlider(self.SettingsTab, self, 'pointWidth', box='Point Width', minValue=1, maxValue=20, step=1, callback = self.updateValues, ticks=1)

        # #####
        # jittering options
        box = OWGUI.widgetBox(self.SettingsTab, " Jittering options ")
        box2 = OWGUI.widgetBox(box, orientation = "horizontal")
        self.jitterLabel = QLabel('Jittering size (% of size)  ', box2)
        self.jitterSizeCombo = OWGUI.comboBox(box2, self, "jitterSize", callback = self.updateValues, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)
        OWGUI.checkBox(box, self, 'jitterContinuous', 'Jitter continuous attributes', callback = self.updateValues, tooltip = "Does jittering apply also on continuous attributes?")
        
        # general graph settings
        box = OWGUI.widgetBox(self.SettingsTab, " General graph settings ")
        OWGUI.checkBox(box, self, 'showXAxisTitle', 'X axis title', callback = self.updateAxisTitle)
        OWGUI.checkBox(box, self, 'showYAxisTitle', 'Y axis title', callback = self.updateAxisTitle)
        OWGUI.checkBox(box, self, 'showVerticalGridlines', 'Vertical gridlines', callback = self.setVerticalGridlines)
        OWGUI.checkBox(box, self, 'showHorizontalGridlines', 'Horizontal gridlines', callback = self.setHorizontalGridlines)
        OWGUI.checkBox(box, self, 'showLegend', 'Show legend', callback = self.updateValues)
        #OWGUI.checkBox(box, self, 'showDistributions', 'Show distributions', callback = self.updateValues, tooltip = "When visualizing discrete attributes on x and y axis show pie chart for better distribution perception")
        OWGUI.checkBox(box, self, 'showFilledSymbols', 'Show filled symbols', callback = self.updateValues)
        OWGUI.checkBox(box, self, 'optimizedDrawing', 'Optimize drawing (biased)', callback = self.updateValues, tooltip = "Speed up drawing by drawing all point belonging to one class value at once")

        box3 = OWGUI.widgetBox(self.SettingsTab, " Tooltips settings ")
        OWGUI.comboBox(box3, self, "tooltipKind", items = ["Don't show tooltips", "Show visible attributes", "Show all attributes"], callback = self.updateValues)

        OWGUI.checkBox(self.SettingsTab, self, 'autoSendSelection', 'Auto send selected data', box = "Data selection", callback = self.setAutoSendSelection, tooltip = "Send signals with selected data whenever the selection changes.")
        self.graph.autoSendSelectionCallback = self.setAutoSendSelection
        
        self.gSetGridColorB = QPushButton("Grid Color", self.SettingsTab)
        self.gSetCanvasColorB = QPushButton("Canvas Color", self.SettingsTab)
        self.connect(self.gSetGridColorB, SIGNAL("clicked()"), self.setGraphGridColor)
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)
        
        self.activateLoadedSettings()
        self.resize(900, 700)


    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.graph.pointWidth = self.pointWidth
        self.graph.jitterSize = self.jitterSize
        self.graph.setShowXaxisTitle(self.showXAxisTitle)
        self.graph.setShowYLaxisTitle(self.showYAxisTitle)
        self.graph.enableGridXB(self.showVerticalGridlines)
        self.graph.enableGridYL(self.showHorizontalGridlines)

        self.graph.updateSettings(enabledLegend = self.showLegend, showDistributions = self.showDistributions)
        self.graph.updateSettings(jitterContinuous = self.jitterContinuous, showFilledSymbols = self.showFilledSymbols, tooltipKind = self.tooltipKind)
        
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        self.graph.setGridPen(QPen(QColor(self.graphGridColor)))
                
        apply([self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection, self.zoomSelectToolbar.actionPolygonSelection][self.toolbarSelection], [])
        

    # #######################################################################################################
    # KNN OPTIMIZATION BUTTON EVENTS
    # #######################################################################################################
   
    # evaluate knn accuracy on current projection
    def evaluateCurrentProjection(self):
        acc, other_results = self.graph.getProjectionQuality(self.attrX, self.attrY, self.attrColor)
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
        self.graph.updateData(self.attrX, self.attrY, "", "", "", 0, showKNNModel = 1, showCorrect = 1)
        self.graph.update()
        self.repaint()

    # show quality of knn model by coloring accurate predictions with lighter color and bad predictions with dark color
    def showKNNWrong(self):
        self.graph.updateData(self.attrX, self.attrY, "", "", "", 0, showKNNModel = 1, showCorrect = 0)
        self.graph.update()
        self.repaint()

    # reevaluate projections in result list with different k values
    def reevaluateProjections(self):
        results = list(self.optimizationDlg.getShownResults())
        self.optimizationDlg.clearResults()

        self.progressBarInit()
        self.optimizationDlg.disableControls()

        # create a dataset with scaled data
        fullData = orange.ExampleTable(self.data.domain)
        for i in range(len(self.data)):
            fullData.append([self.graph.noJitteringScaledData[ind][i] for ind in range(len(self.data.domain.attributes))] + [self.data[i][self.data.domain.classVar.name]])

        
        testIndex = 0
        for (acc, tableLen, other, [xattr, yattr], strList) in results:
            if self.optimizationDlg.isOptimizationCanceled(): continue
            testIndex += 1
            self.progressBarSet(100.0*testIndex/float(len(results)))
            
            table = fullData.select([xattr,yattr, self.data.domain.classVar.name])
            table = orange.Preprocessor_dropMissing(table)
            if len(table) < self.optimizationDlg.minExamples: continue

            accuracy, other_results = self.optimizationDlg.kNNComputeAccuracy(table)
            self.optimizationDlg.addResult(accuracy, other_results, len(table), [xattr, yattr])

        self.progressBarFinished()
        self.optimizationDlg.enableControls()
        self.optimizationDlg.finishedAddingResults()
        

    # ####################################
    # find optimal class separation for shown attributes
    def optimizeSeparation(self):
        if self.data == None: return
        
        self.optimizationDlg.clearResults()
        self.progressBarInit()
        self.optimizationDlg.disableControls()

        startTime = time.time()
        attributeNameOrder = self.optimizationDlg.getEvaluatedAttributes(self.data)
        attributeNameOrder.sort()

        if len(attributeNameOrder) > 1000:
            self.warning("Since there were too many attributes, all but best 1000 attributes were removed.")
            attributeNameOrder = attributeNameOrder[:1000]

        projections = []
        for i in range(len(attributeNameOrder)):
            for j in range(i+1, len(attributeNameOrder)):
                projections.append((attributeNameOrder[i][0] + attributeNameOrder[j][0], attributeNameOrder[i][1], attributeNameOrder[j][1]))

        # sort projections using heuristics
        projections.sort()
        projections.reverse()

        self.graph.percentDataUsed = self.optimizationDlg.percentDataUsed
        self.graph.getOptimalSeparation(projections, self.optimizationDlg.addResult)

        self.progressBarFinished()
        self.optimizationDlg.enableControls()
        self.optimizationDlg.finishedAddingResults()

        secs = time.time() - startTime
        print "----------------------------\nNumber of possible projections: %d\nUsed time: %d min, %d sec" %(len(projections), secs/60, secs%60)


    #update status on progress bar - gets called by OWScatterplotGraph
    def updateProgress(self, current, total):
        self.progressBar.setTotalSteps(total)
        self.progressBar.setProgress(current)

    def showSelectedAttributes(self):
        self.graph.removeAllSelections()
        val = self.optimizationDlg.getSelectedProjection()
        if not val: return
        (accuracy, other_results, tableLen, list, strList) = val

        attrNames = [attr.name for attr in self.data.domain]
        for item in list:
            if not item in attrNames:
                print "invalid settings"
                return

        self.attrX = list[0]
        self.attrY = list[1]
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
        self.graph.updateData(self.attrX, self.attrY, self.attrColor, self.attrShape, self.attrSize, self.showColorLegend)
        self.graph.repaint()

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        (selected, unselected, merged) = self.graph.getSelectionsAsExampleTables(self.attrX, self.attrY)
        self.send("Selected Examples",selected)
        self.send("Unselected Examples",unselected)
        self.send("Example Distribution", merged)


    # #######################################
    # SCATTERPLOT SIGNALS
    # #######################################

    # receive new data and update all fields
    def cdata(self, data):
        self.optimizationDlg.clearResults()
        exData = self.data
        self.data = None
        if data: self.data = orange.Preprocessor_dropMissingClasses(data)
        self.graph.setData(self.data)
        self.optimizationDlg.setData(data)  # set k value to sqrt(n)
       
        if not (self.data and exData and str(exData.domain.variables) == str(self.data.domain.variables)): # preserve attribute choice if the domain is the same
            self.initAttrValues()
        
        self.updateGraph()
        self.sendSelections()

    def subsetdata(self, data):
        self.graph.subsetData = data
        self.updateGraph()
       
    
    # receive information about which attributes we want to show on x and y axis
    def attributeSelection(self, list):
        if not self.data or not list or len(list) < 2: return

        self.attrX = list[0]
        self.attrY = list[1]
        self.updateGraph()       
    # ################################################


    # #######################################
    # SCATTERPLOT SETTINGS
    # #######################################

    def updateValues(self):
        self.graph.jitterSize = self.jitterSize
        self.graph.showFilledSymbols = self.showFilledSymbols
        self.graph.pointWidth = self.pointWidth
        self.graph.jitterContinuous = self.jitterContinuous
        self.graph.enabledLegend = self.showLegend
        self.graph.showDistributions = self.showDistributions
        self.graph.optimizedDrawing = self.optimizedDrawing
        self.graph.tooltipKind = self.tooltipKind
        self.updateGraph()
        
    def updateAxisTitle(self):
        self.graph.setShowXaxisTitle(self.showXAxisTitle)
        self.graph.setShowYLaxisTitle(self.showYAxisTitle)
        self.updateGraph()

    def setVerticalGridlines(self):
        self.graph.enableGridXB(self.showVerticalGridlines)

    def setHorizontalGridlines(self):
        self.graph.enableGridYL(self.showHorizontalGridlines)

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

    def setGraphGridColor(self):
        newColor = QColorDialog.getColor(QColor(self.graphGridColor))
        if newColor.isValid():
            self.graphGridColor = str(newColor.name())
            self.graph.setGridColor(newColor)


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlot()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
