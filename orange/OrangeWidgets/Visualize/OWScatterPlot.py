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
from OWClusterOptimization import *
import OWGUI
import OWToolbars

###########################################################################################
##### WIDGET : Scatterplot visualization
###########################################################################################
class OWScatterPlot(OWWidget):
    settingsList = ["pointWidth", "showXAxisTitle",
                    "showYAxisTitle", "showVerticalGridlines", "showHorizontalGridlines",
                    "showLegend", "graphGridColor", "graphCanvasColor", "jitterSize", "jitterContinuous", "showFilledSymbols",
                    "showDistributions", "autoSendSelection", "optimizedDrawing", "toolbarSelection", "showClusters", "VizRankClassifierName", "clusterClassifierName"]
    jitterSizeList = ['0.0', '0.1','0.5','1','2','3','4','5','7', '10', '15', '20', '30', '40', '50']
    jitterSizeNums = [0.0, 0.1,   0.5,  1,  2 , 3,  4 , 5 , 7 ,  10,   15,   20 ,  30 ,  40 ,  50 ]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "ScatterPlot", TRUE)

        self.inputs = [("Examples", ExampleTable, self.cdata), ("Example Subset", ExampleTable, self.subsetdata, 1, 1), ("Attribute selection", list, self.attributeSelection)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass), ("VizRank learner", orange.Learner), ("Cluster learner", orange.Learner)]

        #set default settings
        self.pointWidth = 5
        self.showXAxisTitle = 1
        self.showYAxisTitle = 1
        self.showVerticalGridlines = 0
        self.showHorizontalGridlines = 0
        self.showLegend = 1
        self.showDistributions = 0
        self.optimizedDrawing = 1
        self.showClusters = 0
        self.tooltipKind = 1
        self.toolbarSelection = 0
        self.showAxisScale = 1
        self.VizRankClassifierName = "VizRank classifier (Scatterplot)"
        self.clusterClassifierName = "Visual cluster classifier (Scatterplot)"
        
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
        self.optimizationDlg = kNNOptimization(self, self.signalManager, self.graph, "ScatterPlot")
        self.optimizationDlg.label1.hide()
        self.optimizationDlg.optimizationTypeCombo.hide()
        self.optimizationDlg.attributeCountCombo.hide()
        self.optimizationDlg.attributeLabel.hide()
        self.graph.kNNOptimization = self.optimizationDlg

        # cluster dialog
        self.clusterDlg = ClusterOptimization(self, self.signalManager, self.graph, "ScatterPlot")
        self.clusterDlg.label1.hide()
        self.clusterDlg.optimizationTypeCombo.hide()
        self.clusterDlg.attributeCountCombo.hide()
        self.clusterDlg.attributeLabel.hide()
        self.graph.clusterOptimization = self.clusterDlg
        
        self.optimizationDlgButton = OWGUI.button(self.GeneralTab, self, "VizRank optimization dialog", callback = self.optimizationDlg.reshow)
        self.clusterDetectionDlgButton = OWGUI.button(self.GeneralTab, self, "Cluster detection dialog", callback = self.clusterDlg.reshow)
        self.connect(self.clusterDlg.startOptimizationButton , SIGNAL("clicked()"), self.optimizeClusters)
        self.connect(self.clusterDlg.resultList, SIGNAL("selectionChanged()"),self.showSelectedCluster)
        self.graph.clusterOptimization = self.clusterDlg

        # zooming / selection
        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        # ####################################
        #K-NN OPTIMIZATION functionality
        self.connect(self.optimizationDlg.reevaluateResults, SIGNAL("clicked()"), self.reevaluateProjections)
        self.connect(self.optimizationDlg.evaluateProjectionButton, SIGNAL("clicked()"), self.evaluateCurrentProjection)
        self.connect(self.optimizationDlg.showKNNCorrectButton, SIGNAL("clicked()"), self.showKNNCorect)
        self.connect(self.optimizationDlg.showKNNWrongButton, SIGNAL("clicked()"), self.showKNNWrong)

        self.connect(self.optimizationDlgButton, SIGNAL("clicked()"), self.optimizationDlg.reshow)
        self.connect(self.optimizationDlg.resultList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        
        self.connect(self.optimizationDlg.startOptimizationButton , SIGNAL("clicked()"), self.optimizeSeparation)

        # ####################################
        # SETTINGS TAB

        # point width
        OWGUI.hSlider(self.SettingsTab, self, 'pointWidth', box='Point Width', minValue=1, maxValue=20, step=1, callback = self.replotCurves, ticks=1)

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
        OWGUI.checkBox(box, self, 'showAxisScale', 'Show axis scale', callback = self.setAxisScale)
        OWGUI.checkBox(box, self, 'showLegend', 'Show legend', callback = self.updateValues)
        #OWGUI.checkBox(box, self, 'showDistributions', 'Show distributions', callback = self.updateValues, tooltip = "When visualizing discrete attributes on x and y axis show pie chart for better distribution perception")
        OWGUI.checkBox(box, self, 'showFilledSymbols', 'Show filled symbols', callback = self.updateValues)
        OWGUI.checkBox(box, self, 'optimizedDrawing', 'Optimize drawing (biased)', callback = self.updateValues, tooltip = "Speed up drawing by drawing all point belonging to one class value at once")
        OWGUI.checkBox(box, self, 'showClusters', 'Show clusters', callback = self.updateValues, tooltip = "Show a line boundary around a significant cluster")

        box3 = OWGUI.widgetBox(self.SettingsTab, " Tooltips settings ")
        OWGUI.comboBox(box3, self, "tooltipKind", items = ["Don't show tooltips", "Show visible attributes", "Show all attributes"], callback = self.updateValues)

        OWGUI.checkBox(self.SettingsTab, self, 'autoSendSelection', 'Auto send selected data', box = "Data selection", callback = self.setAutoSendSelection, tooltip = "Send signals with selected data whenever the selection changes.")
        self.graph.autoSendSelectionCallback = self.setAutoSendSelection
        
        self.gSetGridColorB = QPushButton("Grid Color", self.SettingsTab)
        self.gSetCanvasColorB = QPushButton("Canvas Color", self.SettingsTab)
        self.connect(self.gSetGridColorB, SIGNAL("clicked()"), self.setGraphGridColor)
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)

        self.connect(self.optimizationDlg.classifierNameEdit, SIGNAL("textChanged(const QString &)"), self.VizRankClassifierNameChanged)
        self.connect(self.clusterDlg.classifierNameEdit, SIGNAL("textChanged(const QString &)"), self.clusterClassifierNameChanged)
        
        self.activateLoadedSettings()
        self.resize(900, 700)

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.graph.pointWidth = self.pointWidth
        self.graph.jitterSize = self.jitterSize
        self.graph.showAxisScale = self.showAxisScale
        self.setAxisScale()
        self.graph.setShowXaxisTitle(self.showXAxisTitle)
        self.graph.setShowYLaxisTitle(self.showYAxisTitle)
        self.graph.enableGridXB(self.showVerticalGridlines)
        self.graph.enableGridYL(self.showHorizontalGridlines)
        self.graph.showClusters = self.showClusters
        self.graph.updateSettings(enabledLegend = self.showLegend, showDistributions = self.showDistributions)
        self.graph.updateSettings(jitterContinuous = self.jitterContinuous, showFilledSymbols = self.showFilledSymbols, tooltipKind = self.tooltipKind)
        
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        self.graph.setGridPen(QPen(QColor(self.graphGridColor)))
                
        apply([self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection, self.zoomSelectToolbar.actionPolygonSelection][self.toolbarSelection], [])

        self.optimizationDlg.classifierName = self.VizRankClassifierName
        self.optimizationDlg.classifierNameChanged(self.VizRankClassifierName)
        self.clusterDlg.classifierName = self.clusterClassifierName
        self.clusterDlg.classifierNameChanged(self.clusterClassifierName)

    def VizRankClassifierNameChanged(self, text):
        self.VizRankClassifierName = self.optimizationDlg.classifierName

    def clusterClassifierNameChanged(self, text):
        self.clusterClassifierName = self.clusterDlg.classifierName

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

    def showKNNCorect(self):
        self.optimizationDlg.showKNNWrongButton.setOn(0)
        self.showSelectedAttributes()

    # show quality of knn model by coloring accurate predictions with lighter color and bad predictions with dark color
    def showKNNWrong(self):
        self.optimizationDlg.showKNNCorrectButton.setOn(0) 
        self.showSelectedAttributes()

         
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
        for (acc, tableLen, other, [xattr, yattr], tryIndex, strList) in results:
            if self.optimizationDlg.isOptimizationCanceled(): continue
            testIndex += 1
            self.progressBarSet(100.0*testIndex/float(len(results)))
            
            table = fullData.select([xattr,yattr, self.data.domain.classVar.name])
            table = orange.Preprocessor_dropMissing(table)
            if len(table) < self.optimizationDlg.minExamples: continue

            accuracy, other_results = self.optimizationDlg.kNNComputeAccuracy(table)
            self.optimizationDlg.addResult(accuracy, other_results, len(table), [xattr, yattr], tryIndex, strList)

        self.progressBarFinished()
        self.optimizationDlg.enableControls()
        self.optimizationDlg.finishedAddingResults()
        

    # ################################################################################################
    # find projections where different class values are well separated
    def optimizeSeparation(self):
        if self.data == None: return
        
        self.optimizationDlg.clearResults()
        self.progressBarInit()
        self.optimizationDlg.disableControls()

        startTime = time.time()
        attributeNameOrder = self.optimizationDlg.getEvaluatedAttributes(self.data)

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
        print "Number of possible projections: %d\n----------------------------" % (len(projections))
        print "Used time: %d min, %d sec" %(secs/60, secs%60)


    # ################################################################################################
    # find projections that have tight clusters of points that belong to the same class value
    def optimizeClusters(self):
        if self.data == None: return
        
        self.clusterDlg.clearResults()
        self.clusterDlg.clusterStabilityButton.setOn(0)
        self.clusterDlg.pointStability = None
        self.progressBarInit()
        self.clusterDlg.disableControls()

        startTime = time.time()
        attributeNameOrder = self.clusterDlg.getEvaluatedAttributes(self.data)

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

        self.graph.getOptimalClusters(projections, self.clusterDlg.addResult)

        self.progressBarFinished()
        self.clusterDlg.enableControls()
        self.clusterDlg.finishedAddingResults()

        secs = time.time() - startTime
        print "Number of possible projections: %d\n----------------------------" % (len(projections))
        print "Used time: %d min, %d sec" %(secs/60, secs%60)


    #update status on progress bar - gets called by OWScatterplotGraph
    def updateProgress(self, current, total):
        self.progressBar.setTotalSteps(total)
        self.progressBar.setProgress(current)

    def showSelectedAttributes(self):
        self.graph.removeAllSelections()
        val = self.optimizationDlg.getSelectedProjection()
        if not val: return
        (accuracy, other_results, tableLen, list, tryIndex, strList) = val
        kNNValues = None
        if self.optimizationDlg.showKNNCorrectButton.isOn() or self.optimizationDlg.showKNNWrongButton.isOn():
            kNNValues = self.optimizationDlg.kNNClassifyData(self.graph.createProjectionAsExampleTable(self.attrX, self.attrY))
            if self.optimizationDlg.showKNNCorrectButton.isOn(): kNNValues = [1.0 - val for val in kNNValues]
            clusterClosure = self.graph.clusterClosure
        else: clusterClosure = None

        self.showAttributes(list, kNNValues, clusterClosure)
        

    def showSelectedCluster(self):
        self.graph.removeAllSelections()
        val = self.clusterDlg.getSelectedCluster()
        if not val: return
        (value, closure, vertices, attrList, classValue, enlargedClosure, other, strList) = val

        if self.clusterDlg.clusterStabilityButton.isOn():
            validData = self.graph.getValidList([self.graph.attributeNames.index(self.attrX), self.graph.attributeNames.index(self.attrY)])
            insideColors = Numeric.compress(validData, self.clusterDlg.pointStability)
        else: insideColors = None

        self.showAttributes(attrList, insideColors, clusterClosure = (closure, enlargedClosure, classValue))

        """
        if type(tryIndex[0]) == tuple:
            for vals in tryIndex:
                print "class = %s\nvalue = %.2f   points = %d\ndist = %.4f\n-------" % (vals[0], vals[1], vals[2], vals[3])
        else:
            print "class = %s\nvalue = %.2f   points = %d\ndist = %.4f\n-------" % (tryIndex[0], tryIndex[1], tryIndex[2], tryIndex[3])
        print "---------------------------"
        """
        
    def showAttributes(self, attrList, insideColors = None, clusterClosure = None):
        attrNames = [attr.name for attr in self.data.domain]
        for item in attrList:
            if not item in attrNames:
                print "invalid settings"
                return

        self.attrX = attrList[0]; self.attrY = attrList[1]
        self.attrColor = self.data.domain.classVar.name

        self.graph.updateData(self.attrX, self.attrY, self.attrColor, self.attrShape, self.attrSize, self.showColorLegend, insideColors = insideColors, clusterClosure = clusterClosure)
        self.graph.repaint()        

        
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
        self.graph.insideColors = None
        self.graph.clusterClosure = None
        self.updateGraph()

    def updateGraph(self, *args):
        self.graph.updateData(self.attrX, self.attrY, self.attrColor, self.attrShape, self.attrSize, self.showColorLegend)
        self.graph.update()  # don't know if this is necessary
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
        if self.data != None and data != None and self.data.checksum() == data.checksum(): return    # check if the new data set is the same as the old one
        self.optimizationDlg.clearResults()
        self.clusterDlg.clearResults()
        exData = self.data
        self.data = None
        if data: self.data = orange.Preprocessor_dropMissingClasses(data)
        self.graph.setData(self.data)
        self.optimizationDlg.setData(data)  # set k value to sqrt(n)
        self.clusterDlg.setData(data)
        self.graph.insideColors = None; self.graph.clusterClosure = None
       
        if not (self.data and exData and str(exData.domain.variables) == str(self.data.domain.variables)): # preserve attribute choice if the domain is the same
            self.initAttrValues()
        
        self.updateGraph()
        self.sendSelections()

    # set an example table with a data subset subset of the data. if called by a visual classifier, the update parameter will be 0
    def subsetdata(self, data, update = 1):
        if self.graph.subsetData != None and data != None and self.graph.subsetData.checksum() == data.checksum(): return    # check if the new data set is the same as the old one
        self.graph.subsetData = data
        qApp.processEvents()            # TODO: find out why scatterplot crashes if we remove this line and send a subset of data that is not in self.rawdata - as in cluster argumentation
        if update: self.updateGraph()
        self.optimizationDlg.setSubsetData(data)
        self.clusterDlg.setSubsetData(data)
       
    
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

    def setAxisScale(self):
        self.graph.showAxisScale = self.showAxisScale
        if not self.showAxisScale:
            self.graph.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
            self.graph.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
            self.graph.axisScaleDraw(QwtPlot.xBottom).setTickLength(0, 0, 0)
            self.graph.axisScaleDraw(QwtPlot.yLeft).setTickLength(0, 0, 0)
            self.graph.axisScaleDraw(QwtPlot.xBottom).setOptions(0) 
            self.graph.axisScaleDraw(QwtPlot.yLeft).setOptions(0) 
        else:
            self.graph.setAxisScaleDraw(QwtPlot.xBottom, QwtScaleDraw())
            self.graph.setAxisScaleDraw(QwtPlot.yLeft, QwtScaleDraw())
            self.graph.axisScaleDraw(QwtPlot.xBottom).setTickLength(1, 1, 3)
            self.graph.axisScaleDraw(QwtPlot.yLeft).setTickLength(1, 1, 3)
            self.graph.axisScaleDraw(QwtPlot.xBottom).setOptions(1) 
            self.graph.axisScaleDraw(QwtPlot.yLeft).setOptions(1)
        self.graph.repaint()


    def replotCurves(self):
        self.graph.pointWidth = self.pointWidth
        for key in self.graph.curveKeys():
            symbol = self.graph.curveSymbol(key)
            self.graph.setCurveSymbol(key, QwtSymbol(symbol.style(), symbol.brush(), symbol.pen(), QSize(self.pointWidth, self.pointWidth)))
        self.graph.repaint()

    def updateValues(self):
        self.graph.jitterSize = self.jitterSize
        self.graph.showFilledSymbols = self.showFilledSymbols
        self.graph.pointWidth = self.pointWidth
        self.graph.jitterContinuous = self.jitterContinuous
        self.graph.enabledLegend = self.showLegend
        self.graph.showDistributions = self.showDistributions
        self.graph.optimizedDrawing = self.optimizedDrawing
        self.graph.tooltipKind = self.tooltipKind
        self.graph.showClusters = self.showClusters
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

    def setMinimalGraphProperties(self):
        attrs = ["pointWidth", "showLegend", "showClusters", "showXAxisTitle", "showYAxisTitle", "showVerticalGridlines", "showHorizontalGridlines", "showAxisScale", "autoSendSelection"]
        self.oldSettings = dict([(attr, getattr(self, attr)) for attr in attrs])

        self.pointWidth = 4
        self.showLegend = 0
        self.showClusters = 0
        self.showXAxisTitle = 0
        self.showYAxisTitle = 0
        self.showVerticalGridlines = 0
        self.showHorizontalGridlines = 0
        self.autoSendSelection = 0
        self.showAxisScale = 0
        #self.setAxisScale()
        #self.updateValues()

    def restoreGraphProperties(self):
        if hasattr(self, "oldSettings"):
            for key in self.oldSettings:
                self.__setattr__(key, self.oldSettings[key])


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlot()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
