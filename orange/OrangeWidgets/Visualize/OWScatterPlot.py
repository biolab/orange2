"""
<name>Scatterplot</name>
<description>Shows data using scatterplot</description>
<author>Gregor Leban (gregor.leban@fri.uni-lj.si)</author>
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
    settingsList = ["graph.pointWidth", "graph.showXaxisTitle", "graph.showYLaxisTitle", "showVerticalGridlines", "showHorizontalGridlines", "graph.showAxisScale",
                    "graph.enabledLegend", "graphGridColor", "graphCanvasColor", "graph.jitterSize", "graph.jitterContinuous", "graph.showFilledSymbols",
                    "graph.showDistributions", "autoSendSelection", "graph.optimizedDrawing", "toolbarSelection", "graph.showClusters",
                    "VizRankClassifierName", "clusterClassifierName", "learnerIndex"]
    jitterSizeList = ['0.0', '0.1','0.5','1','2','3','4','5','7', '10', '15', '20', '30', '40', '50']
    jitterSizeNums = [0.0, 0.1,   0.5,  1,  2 , 3,  4 , 5 , 7 ,  10,   15,   20 ,  30 ,  40 ,  50 ]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "ScatterPlot", TRUE)

        self.inputs = [("Examples", ExampleTable, self.cdata), ("Example Subset", ExampleTable, self.subsetdata, 1, 1), ("Attribute selection", list, self.attributeSelection), ("Evaluation Results", orngTest.ExperimentResults, self.test_results)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass), ("Learner", orange.Learner)]

        # local variables    
        self.showVerticalGridlines = 0
        self.showHorizontalGridlines = 0
        self.autoSendSelection = 1
        self.toolbarSelection = 0
        self.VizRankClassifierName = "VizRank classifier (Scatterplot)"
        self.clusterClassifierName = "Visual cluster classifier (Scatterplot)"
        self.graphCanvasColor = str(Qt.white.name())
        self.graphGridColor = str(Qt.black.name())
        self.classificationResults = None
        self.learnerIndex = 0
        self.learnersArray = [None, None]   # VizRank, Cluster

        self.graph = OWScatterPlotGraph(self, self.mainArea)
        self.optimizationDlg = kNNOptimization(self, self.signalManager, self.graph, "ScatterPlot")
        self.clusterDlg = ClusterOptimization(self, self.signalManager, self.graph, "ScatterPlot")
       
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
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        #x attribute
        self.attrX = ""
        self.attrXCombo = OWGUI.comboBox(self.GeneralTab, self, "attrX", " X Axis Attribute ", callback = self.removeSelectionsAndUpdateGraph, sendSelectedValue = 1, valueType = str)

        # y attribute
        self.attrY = ""
        self.attrYCombo = OWGUI.comboBox(self.GeneralTab, self, "attrY", " Y Axis Attribute ", callback = self.removeSelectionsAndUpdateGraph, sendSelectedValue = 1, valueType = str)

        # coloring
        self.showColorLegend = 0
        self.attrColor = ""
        box = OWGUI.widgetBox(self.GeneralTab, " Color Attribute")
        OWGUI.checkBox(box, self, 'showColorLegend', 'Show color legend', callback = self.updateGraph)
        self.attrColorCombo = OWGUI.comboBox(box, self, "attrColor", callback = self.updateGraph, sendSelectedValue=1, valueType = str)
        
        # shaping
        self.attrShape = ""
        self.attrShapeCombo = OWGUI.comboBox(self.GeneralTab, self, "attrShape", " Shape Attribute ", callback = self.updateGraph, sendSelectedValue=1, valueType = str)
                
        # sizing
        self.attrSize = ""
        self.attrSizeCombo = OWGUI.comboBox(self.GeneralTab, self, "attrSize", " Size Attribute ", callback = self.updateGraph, sendSelectedValue=1, valueType = str)
        
        # optimization
        self.optimizationDlg.label1.hide()
        self.optimizationDlg.optimizationTypeCombo.hide()
        self.optimizationDlg.attributeCountCombo.hide()
        self.optimizationDlg.attributeLabel.hide()
        self.optimizationDlg.optimizeBestProjectionCheck.hide()
        self.optimizationDlg.optimizeBestProjectionCombo.hide()
        self.graph.kNNOptimization = self.optimizationDlg

        # cluster dialog
        self.clusterDlg.label1.hide()
        self.clusterDlg.optimizationTypeCombo.hide()
        self.clusterDlg.attributeCountCombo.hide()
        self.clusterDlg.attributeLabel.hide()
        self.graph.clusterOptimization = self.clusterDlg
        
        self.optimizationButtons = OWGUI.widgetBox(self.GeneralTab, " Optimization Dialogs ", orientation = "horizontal")
        OWGUI.button(self.optimizationButtons, self, "VizRank", callback = self.optimizationDlg.reshow)
        OWGUI.button(self.optimizationButtons, self, "Cluster", callback = self.clusterDlg.reshow)
        self.connect(self.clusterDlg.startOptimizationButton , SIGNAL("clicked()"), self.optimizeClusters)
        self.connect(self.clusterDlg.resultList, SIGNAL("selectionChanged()"),self.showSelectedCluster)
        self.graph.clusterOptimization = self.clusterDlg

        # zooming / selection
        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        # ####################################
        #K-NN OPTIMIZATION functionality
        self.optimizationDlg.localOptimizationSettingsBox.hide()
        self.connect(self.optimizationDlg.evaluateProjectionButton, SIGNAL("clicked()"), self.evaluateCurrentProjection)
        self.connect(self.optimizationDlg.showKNNCorrectButton, SIGNAL("clicked()"), self.showKNNCorect)
        self.connect(self.optimizationDlg.showKNNWrongButton, SIGNAL("clicked()"), self.showKNNWrong)

        self.connect(self.optimizationDlg.resultList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        
        self.connect(self.optimizationDlg.startOptimizationButton , SIGNAL("clicked()"), self.optimizeSeparation)

        # ####################################
        # SETTINGS TAB

        # point width
        OWGUI.hSlider(self.SettingsTab, self, 'graph.pointWidth', box=' Point Size ', minValue=1, maxValue=20, step=1, callback = self.replotCurves)

        # #####
        # jittering options
        box = OWGUI.widgetBox(self.SettingsTab, " Jittering Options ")
        box2 = OWGUI.widgetBox(box, orientation = "horizontal")
        self.jitterLabel = QLabel('Jittering size (% of size)  ', box2)
        self.jitterSizeCombo = OWGUI.comboBox(box2, self, "graph.jitterSize", callback = self.updateGraph, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)
        OWGUI.checkBox(box, self, 'graph.jitterContinuous', 'Jitter continuous attributes', callback = self.updateGraph, tooltip = "Does jittering apply also on continuous attributes?")
        
        # general graph settings
        box = OWGUI.widgetBox(self.SettingsTab, " General Graph Settings ")
        OWGUI.checkBox(box, self, 'graph.showXaxisTitle', 'X axis title', callback = self.updateGraph)
        OWGUI.checkBox(box, self, 'graph.showYLaxisTitle', 'Y axis title', callback = self.updateGraph)
        OWGUI.checkBox(box, self, 'graph.showAxisScale', 'Show axis scale', callback = self.updateGraph)
        OWGUI.checkBox(box, self, 'graph.enabledLegend', 'Show legend', callback = self.updateGraph)
        #OWGUI.checkBox(box, self, 'graph.showDistributions', 'Show distributions', callback = self.updateGraph, tooltip = "When visualizing discrete attributes on x and y axis show pie chart for better distribution perception")
        OWGUI.checkBox(box, self, 'graph.showFilledSymbols', 'Show filled symbols', callback = self.updateGraph)
        OWGUI.checkBox(box, self, 'graph.optimizedDrawing', 'Optimize drawing', callback = self.updateGraph, tooltip = "Speed up drawing by drawing all point belonging to one class value at once")
        OWGUI.checkBox(box, self, 'showVerticalGridlines', 'Vertical gridlines', callback = self.setVerticalGridlines)
        OWGUI.checkBox(box, self, 'showHorizontalGridlines', 'Horizontal gridlines', callback = self.setHorizontalGridlines)
        OWGUI.checkBox(box, self, 'graph.showClusters', 'Show clusters', callback = self.updateGraph, tooltip = "Show a line boundary around a significant cluster")

        box3 = OWGUI.widgetBox(self.SettingsTab, " Tooltips Settings ")
        OWGUI.comboBox(box3, self, "graph.tooltipKind", items = ["Don't show tooltips", "Show visible attributes", "Show all attributes"], callback = self.updateGraph)

        self.activeLearnerCombo = OWGUI.comboBox(self.SettingsTab, self, "learnerIndex", box = " Set Active Learner ", items = ["VizRank Learner", "Cluster Learner"], tooltip = "Select which of the possible learners do you want to send on the widget output.")
        self.connect(self.activeLearnerCombo, SIGNAL("activated(int)"), self.setActiveLearner)
    
        OWGUI.checkBox(self.SettingsTab, self, 'autoSendSelection', 'Auto send selected data', box = " Data selection ", callback = self.setAutoSendSelection, tooltip = "Send signals with selected data whenever the selection changes.")
        self.graph.autoSendSelectionCallback = self.setAutoSendSelection
        
        self.colorButtonsBox = OWGUI.widgetBox(self.SettingsTab, " Change Colors ", orientation = "horizontal")
        OWGUI.button(self.colorButtonsBox, self, "Canvas", self.setGraphCanvasColor)
        OWGUI.button(self.colorButtonsBox, self, "Grid", self.setGraphGridColor)
        
        self.activateLoadedSettings()
        self.resize(900, 700)

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.graph.enableGridXB(self.showVerticalGridlines)
        self.graph.enableGridYL(self.showHorizontalGridlines)
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        self.graph.setGridPen(QPen(QColor(self.graphGridColor)))
                
        apply([self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection, self.zoomSelectToolbar.actionPolygonSelection][self.toolbarSelection], [])

        self.optimizationDlg.changeLearnerName(self.VizRankClassifierName)
        self.clusterDlg.changeLearnerName(self.clusterClassifierName)
        self.setActiveLearner(self.learnerIndex)

    def setActiveLearner(self, idx):
        self.send("Learner", self.learnersArray[self.learnerIndex])

    # #######################################################################################################
    # KNN OPTIMIZATION BUTTON EVENTS
    # #######################################################################################################
   
    # evaluate knn accuracy on current projection
    def evaluateCurrentProjection(self):
        acc, other_results = self.graph.getProjectionQuality([self.attrX, self.attrY])
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


    # ################################################################################################
    # find projections where different class values are well separated
    def optimizeSeparation(self):
        if self.data == None: return
        
        self.optimizationDlg.clearResults()
        self.optimizationDlg.disableControls()

        try:
            attributeNameOrder = self.optimizationDlg.getEvaluatedAttributes(self.data) # sort attributes according to the heuristic
            self.graph.getOptimalSeparation(attributeNameOrder, self.optimizationDlg.addResult) # evaluate projections
        except:
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # print the exception

        self.optimizationDlg.enableControls()
        self.optimizationDlg.finishedAddingResults()


    # ################################################################################################
    # find projections that have tight clusters of points that belong to the same class value
    def optimizeClusters(self):
        if self.data == None: return
        
        self.clusterDlg.clearResults()
        self.clusterDlg.clusterStabilityButton.setOn(0)
        self.clusterDlg.pointStability = None
        self.clusterDlg.disableControls()

        try:
            attributeNameOrder = self.clusterDlg.getEvaluatedAttributes(self.data)
            self.graph.getOptimalClusters(attributeNameOrder, self.clusterDlg.addResult)    # evaluate projections
        except:
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # print the exception

        self.clusterDlg.enableControls()
        self.clusterDlg.finishedAddingResults()


    #update status on progress bar - gets called by OWScatterplotGraph
    def updateProgress(self, current, total):
        self.progressBar.setTotalSteps(total)
        self.progressBar.setProgress(current)

    def showSelectedAttributes(self):
        self.graph.removeAllSelections()
        val = self.optimizationDlg.getSelectedProjection()
        if not val: return
        (accuracy, other_results, tableLen, attrs, tryIndex, strList) = val

        values = self.classificationResults
        if self.optimizationDlg.showKNNCorrectButton.isOn() or self.optimizationDlg.showKNNWrongButton.isOn():
            values = self.optimizationDlg.kNNClassifyData(self.graph.createProjectionAsExampleTable([self.graph.attributeNameIndex[self.attrX], self.graph.attributeNameIndex[self.attrY]]))
            if self.optimizationDlg.showKNNCorrectButton.isOn(): values = [1.0 - val for val in values]
            clusterClosure = self.graph.clusterClosure
        else: clusterClosure = None

        self.showAttributes(attrs, values, clusterClosure)
        

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

        if type(other) == dict:
            for vals in other.values():
                print "class = %s\nvalue = %.2f   points = %d\ndist = %.4f   averageDist = %.4f\n-------" % (self.data.domain.classVar.values[vals[0]], vals[1], vals[2], vals[3], vals[5])
        else:
            print "class = %s\nvalue = %.2f   points = %d\ndist = %.4f   averageDist = %.4f\n-------" % (self.data.domain.classVar.values[other[0]], other[1], other[2], other[3], other[5])
        print "---------------------------"
        
        
    def showAttributes(self, attrList, insideColors = None, clusterClosure = None):
        if not self.data: return
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
    def cdata(self, data, clearResults = 1):
        if data:
            name = ""
            if hasattr(data, "name"): name = data.name
            data = orange.Preprocessor_dropMissingClasses(data)
            data.name = name
        if self.data != None and data != None and self.data.checksum() == data.checksum(): return    # check if the new data set is the same as the old one
        exData = self.data
        self.data = data
        self.graph.setData(data)
        self.graph.insideColors = None
        self.graph.clusterClosure = None
        
        self.optimizationDlg.setData(data)
        self.clusterDlg.setData(data, clearResults)
        
        if not (self.data and exData and str(exData.domain.variables) == str(self.data.domain.variables)): # preserve attribute choice if the domain is the same
            self.initAttrValues()

        self.showAttributes([self.attrX, self.attrY], self.classificationResults, clusterClosure = self.graph.clusterClosure)
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

    def test_results(self, results):        
        self.classificationResults = None
        if isinstance(results, orngTest.ExperimentResults) and len(results.results) > 0 and len(results.results[0].probabilities) > 0:
            self.classificationResults = [results.results[i].probabilities[0][results.results[i].actualClass] for i in range(len(results.results))]

        self.showAttributes([self.attrX, self.attrY], self.classificationResults, clusterClosure = self.graph.clusterClosure)
        
    # ################################################

    # #######################################
    # SCATTERPLOT SETTINGS
    # ######################################
    def replotCurves(self):
        for key in self.graph.curveKeys():
            symbol = self.graph.curveSymbol(key)
            self.graph.setCurveSymbol(key, QwtSymbol(symbol.style(), symbol.brush(), symbol.pen(), QSize(self.graph.pointWidth, self.graph.pointWidth)))
        self.graph.repaint()

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
        attrs = ["graph.pointWidth", "graph.enabledLegend", "graph.showClusters", "showXAxisTitle", "showYAxisTitle", "showVerticalGridlines", "showHorizontalGridlines", "graph.showAxisScale", "autoSendSelection"]
        self.oldSettings = dict([(attr, mygetattr(self, attr)) for attr in attrs])

        self.graph.pointWidth = 4
        self.graph.enabledLegend = 0
        self.graph.showClusters = 0
        self.graph.showXaxisTitle = 0
        self.graph.showYLaxisTitle = 0
        self.graph.showAxisScale = 0
        self.showVerticalGridlines = 0
        self.showHorizontalGridlines = 0
        self.autoSendSelection = 0
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
