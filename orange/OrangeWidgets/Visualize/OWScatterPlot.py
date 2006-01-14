"""
<name>Scatterplot</name>
<description>Scatterplot visualization.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/ScatterPlot.png</icon>
<priority>1000</priority>
"""
# ScatterPlot.py
#
# Show data using scatterplot
# 

from OWWidget import *
from OWScatterPlotGraph import *
from OWkNNOptimization import *
import orngVizRank
from OWClusterOptimization import *
import OWGUI, OWToolbars, OWDlgs
from orngScaleData import *
from OWGraph import OWGraph

###########################################################################################
##### WIDGET : Scatterplot visualization
###########################################################################################
class OWScatterPlot(OWWidget):
    settingsList = ["graph.pointWidth", "graph.showXaxisTitle", "graph.showYLaxisTitle", "showGridlines", "graph.showAxisScale",
                    "graph.showLegend", "graph.jitterSize", "graph.jitterContinuous", "graph.showFilledSymbols",
                    "graph.showDistributions", "autoSendSelection", "graph.optimizedDrawing", "toolbarSelection", "graph.showClusters",
                    "VizRankClassifierName", "clusterClassifierName", "learnerIndex", "colorSettings"]
    jitterSizeList = ['0.0', '0.1','0.5','1','2','3','4','5','7', '10', '15', '20', '30', '40', '50']
    jitterSizeNums = [0.0, 0.1,   0.5,  1,  2 , 3,  4 , 5 , 7 ,  10,   15,   20 ,  30 ,  40 ,  50 ]

    contextHandlers = {"": DomainContextHandler("", ["attrX", "attrY", (["attrLabel", "attrShape", "attrSize"], DomainContextHandler.Optional)])}
    
    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "ScatterPlot", TRUE)

        self.inputs =  [("Examples", ExampleTable, self.cdata, Default), ("Example Subset", ExampleTable, self.subsetdata), ("Attribute selection", list, self.attributeSelection), ("Evaluation Results", orngTest.ExperimentResults, self.test_results), ("VizRank Learner", orange.Learner, self.vizRankLearner)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Learner", orange.Learner)]

        # local variables    
        self.showGridlines = 0
        self.autoSendSelection = 1
        self.toolbarSelection = 0
        self.VizRankClassifierName = "VizRank classifier (Scatterplot)"
        self.clusterClassifierName = "Visual cluster classifier (Scatterplot)"
        self.classificationResults = None
        self.outlierValues = None
        self.learnerIndex = 0
        self.learnersArray = [None, None]   # VizRank, Cluster
        self.colorSettings = None

        self.graph = OWScatterPlotGraph(self, self.mainArea, "ScatterPlot")
        self.optimizationDlg = OWVizRank(self, self.signalManager, self.graph, orngVizRank.SCATTERPLOT, "ScatterPlot")
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
        self.attrXCombo = OWGUI.comboBox(self.GeneralTab, self, "attrX", " X Axis Attribute ", callback = self.updateGraph, sendSelectedValue = 1, valueType = str)

        # y attribute
        self.attrY = ""
        self.attrYCombo = OWGUI.comboBox(self.GeneralTab, self, "attrY", " Y Axis Attribute ", callback = self.updateGraph, sendSelectedValue = 1, valueType = str)

        # coloring
        self.showColorLegend = 0
        self.attrColor = ""
        box = OWGUI.widgetBox(self.GeneralTab, " Color Attribute")
        OWGUI.checkBox(box, self, 'showColorLegend', 'Show color legend', callback = self.updateGraph)
        self.attrColorCombo = OWGUI.comboBox(box, self, "attrColor", callback = self.updateGraph, sendSelectedValue=1, valueType = str)
        
        # labelling
        self.attrLabel = ""
        self.attrLabelCombo = OWGUI.comboBox(self.GeneralTab, self, "attrLabel", " Point labelling ", callback = self.updateGraph, sendSelectedValue = 1, valueType = str)

        # shaping
        self.attrShape = ""
        self.attrShapeCombo = OWGUI.comboBox(self.GeneralTab, self, "attrShape", " Shape Attribute ", callback = self.updateGraph, sendSelectedValue=1, valueType = str)
                
        # sizing
        self.attrSize = ""
        self.attrSizeCombo = OWGUI.comboBox(self.GeneralTab, self, "attrSize", " Size Attribute ", callback = self.updateGraph, sendSelectedValue=1, valueType = str)
        
        # cluster dialog
        self.clusterDlg.label1.hide()
        self.clusterDlg.optimizationTypeCombo.hide()
        self.clusterDlg.attributeCountCombo.hide()
        self.clusterDlg.attributeLabel.hide()
        self.graph.clusterOptimization = self.clusterDlg
        
        self.optimizationButtons = OWGUI.widgetBox(self.GeneralTab, " Optimization Dialogs ", orientation = "horizontal")
        OWGUI.button(self.optimizationButtons, self, "VizRank", callback = self.optimizationDlg.reshow, tooltip = "Opens VizRank dialog, where you can search for interesting projections with different subsets of attributes.")
        OWGUI.button(self.optimizationButtons, self, "Cluster", callback = self.clusterDlg.reshow)
        self.connect(self.clusterDlg.startOptimizationButton , SIGNAL("clicked()"), self.optimizeClusters)
        self.connect(self.clusterDlg.resultList, SIGNAL("selectionChanged()"),self.showSelectedCluster)
        self.graph.clusterOptimization = self.clusterDlg

        # zooming / selection
        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        # ####################################
        # SETTINGS TAB
        # point width
        OWGUI.hSlider(self.SettingsTab, self, 'graph.pointWidth', box=' Point Size ', minValue=1, maxValue=20, step=1, callback = self.replotCurves)

        # #####
        # jittering options
        box2 = OWGUI.widgetBox(self.SettingsTab, " Jittering Options ")
        box3 = OWGUI.widgetBox(box2, orientation = "horizontal")
        self.jitterLabel = QLabel('Jittering size (% of size)  ', box3)
        self.jitterSizeCombo = OWGUI.comboBox(box3, self, "graph.jitterSize", callback = self.resetGraphData, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)
        OWGUI.checkBox(box2, self, 'graph.jitterContinuous', 'Jitter continuous attributes', callback = self.resetGraphData, tooltip = "Does jittering apply also on continuous attributes?")
        
        # general graph settings
        box4 = OWGUI.widgetBox(self.SettingsTab, " General Graph Settings ")
        OWGUI.checkBox(box4, self, 'graph.showXaxisTitle', 'X axis title', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showYLaxisTitle', 'Y axis title', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showAxisScale', 'Show axis scale', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showLegend', 'Show legend', callback = self.updateGraph)
        #OWGUI.checkBox(box4, self, 'graph.showDistributions', 'Show distributions', callback = self.updateGraph, tooltip = "When visualizing discrete attributes on x and y axis show pie chart for better distribution perception")
        OWGUI.checkBox(box4, self, 'graph.showFilledSymbols', 'Show filled symbols', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.optimizedDrawing', 'Optimize drawing', callback = self.updateGraph, tooltip = "Speed up drawing by drawing all point belonging to one class value at once")
        OWGUI.checkBox(box4, self, 'showGridlines', 'Show gridlines', callback = self.setShowGridlines)
        OWGUI.checkBox(box4, self, 'graph.showClusters', 'Show clusters', callback = self.updateGraph, tooltip = "Show a line boundary around a significant cluster")

        self.colorButtonsBox = OWGUI.widgetBox(self.SettingsTab, " Colors ", orientation = "horizontal")
        OWGUI.button(self.colorButtonsBox, self, "Set Colors", self.setColors, tooltip = "Set the canvas background color, grid color and color palette for coloring continuous variables")
        
        box5 = OWGUI.widgetBox(self.SettingsTab, " Tooltips Settings ")
        OWGUI.comboBox(box5, self, "graph.tooltipKind", items = ["Don't show tooltips", "Show visible attributes", "Show all attributes"], callback = self.updateGraph)

        self.activeLearnerCombo = OWGUI.comboBox(self.SettingsTab, self, "learnerIndex", box = " Set Active Learner ", items = ["VizRank Learner", "Cluster Learner"], tooltip = "Select which of the possible learners do you want to send on the widget output.")
        self.connect(self.activeLearnerCombo, SIGNAL("activated(int)"), self.setActiveLearner)
    
        OWGUI.checkBox(self.SettingsTab, self, 'autoSendSelection', 'Auto send selected data', box = " Data selection ", callback = self.setAutoSendSelection, tooltip = "Send signals with selected data whenever the selection changes.")
        self.graph.autoSendSelectionCallback = self.setAutoSendSelection
        
        self.SettingsTab.setMinimumWidth(max(self.GeneralTab.sizeHint().width(), self.SettingsTab.sizeHint().width())+20)
        self.icons = self.createAttributeIconDict()
        
        self.activateLoadedSettings()
        self.resize(900, 700)

    
    def activateLoadedSettings(self):
        dlg = self.createColorDialog()
        self.colorPalette = dlg.getColorPalette("colorPalette")

        self.graph.setCanvasBackground(dlg.getColor("Canvas"))
        self.graph.setGridPen(QPen(dlg.getColor("Grid")))
                
        self.graph.enableGridXB(self.showGridlines)
        self.graph.enableGridYL(self.showGridlines)

        apply([self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection, self.zoomSelectToolbar.actionPolygonSelection][self.toolbarSelection], [])

        self.clusterDlg.changeLearnerName(self.clusterClassifierName)
        self.learnersArray[1] = VizRankLearner(SCATTERPLOT, self.optimizationDlg, self.graph)
        self.setActiveLearner(self.learnerIndex)

    # ##############################################################################################################################################################
    # SCATTERPLOT SIGNALS
    # ##############################################################################################################################################################

    def resetGraphData(self):
        self.graph.setData(self.data)
        self.updateGraph()

    # receive new data and update all fields
    def cdata(self, data, clearResults = 1):
        self.closeContext()

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

        self.openContext("", data)

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


    # visualize the results of the classification
    def test_results(self, results):        
        self.classificationResults = None
        if isinstance(results, orngTest.ExperimentResults) and len(results.results) > 0 and len(results.results[0].probabilities) > 0:
            self.classificationResults = [results.results[i].probabilities[0][results.results[i].actualClass] for i in range(len(results.results))]
            self.classificationResults = (self.classificationResults, "Probability of correct classificatioin = %.2f%%")

        self.updateGraph()


    # set the learning method to be used in VizRank
    def vizRankLearner(self, learner):
        self.optimizationDlg.externalLearner = learner

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        #(selected, unselected, merged) = self.graph.getSelectionsAsExampleTables([self.attrX, self.attrY])
        (selected, unselected) = self.graph.getSelectionsAsExampleTables([self.attrX, self.attrY])
        self.send("Selected Examples",selected)
        self.send("Unselected Examples",unselected)
        #self.send("Example Distribution", merged)


    # ##############################################################################################################################################################
    # KNN OPTIMIZATION BUTTON EVENTS
    # ##############################################################################################################################################################

    def setActiveLearner(self, idx):
        self.send("Learner", self.learnersArray[self.learnerIndex])
   
    
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


    def showSelectedAttributes(self):
        val = self.optimizationDlg.getSelectedProjection()
        if not val: return
        (accuracy, other_results, tableLen, attrs, tryIndex, generalDict) = val

        if self.data.domain.classVar:
            self.attrColor = self.data.domain.classVar.name

        self.updateGraph(attrs)
        

    def showSelectedCluster(self):
        val = self.clusterDlg.getSelectedCluster()
        if not val: return
        (value, closure, vertices, attrList, classValue, enlargedClosure, other, strList) = val

        if self.clusterDlg.clusterStabilityButton.isOn():
            validData = self.graph.getValidList([self.graph.attributeNames.index(self.attrX), self.graph.attributeNames.index(self.attrY)])
            insideColors = (Numeric.compress(validData, self.clusterDlg.pointStability), "Point inside a cluster in %.2f%%")
        else: insideColors = None

        self.updateGraph(attrList, insideColors, (closure, enlargedClosure, classValue))

       
    # ##############################################################################################################################################################
    # ATTRIBUTE SELECTION
    # ##############################################################################################################################################################

    def getShownAttributeList(self):
        return [self.attrX, self.attrY]

    def initAttrValues(self):
        self.attrXCombo.clear()
        self.attrYCombo.clear()
        self.attrColorCombo.clear()
        self.attrLabelCombo.clear()
        self.attrShapeCombo.clear()
        self.attrSizeCombo.clear()

        if self.data == None: return

        self.attrColorCombo.insertItem("(One color)")
        self.attrLabelCombo.insertItem("(No labels)")
        self.attrShapeCombo.insertItem("(One shape)")
        self.attrSizeCombo.insertItem("(One size)")

        #labels are usually chosen from meta variables, put them on top
        for metavar in [self.data.domain.getmeta(mykey) for mykey in self.data.domain.getmetas().keys()]:
            self.attrLabelCombo.insertItem(self.icons[metavar.varType], metavar.name)

        contList = []
        discList = []
        for attr in self.data.domain:
            self.attrXCombo.insertItem(self.icons[attr.varType], attr.name)
            self.attrYCombo.insertItem(self.icons[attr.varType], attr.name)
            self.attrColorCombo.insertItem(self.icons[attr.varType], attr.name)
            self.attrSizeCombo.insertItem(self.icons[attr.varType], attr.name)
            if attr.varType == orange.VarTypes.Discrete: self.attrShapeCombo.insertItem(self.icons[attr.varType], attr.name)
            self.attrLabelCombo.insertItem(self.icons[attr.varType], attr.name)

        self.attrX = str(self.attrXCombo.text(0))
        if self.attrYCombo.count() > 1: self.attrY = str(self.attrYCombo.text(1))
        else:                           self.attrY = str(self.attrYCombo.text(0))
            
        if self.data.domain.classVar: self.attrColor = self.data.domain.classVar.name
        else:                         self.attrColor = "(One color)"
        self.attrShape = "(One shape)"
        self.attrSize= "(One size)"


    def updateGraph(self, attrList = None, insideColors = None, clusterClosure = None, *args):
        self.graph.removeAllSelections()
        
        if not self.data:
            return
    
        if attrList:
            self.attrX = attrList[0]
            self.attrY = attrList[1]

        if self.optimizationDlg.showKNNCorrectButton.isOn() or self.optimizationDlg.showKNNWrongButton.isOn():
            kNNExampleAccuracy, probabilities = self.optimizationDlg.kNNClassifyData(self.graph.createProjectionAsExampleTable([self.graph.attributeNameIndex[self.attrX], self.graph.attributeNameIndex[self.attrY]]))
            if self.optimizationDlg.showKNNCorrectButton.isOn(): kNNExampleAccuracy = ([1.0 - val for val in kNNExampleAccuracy], "Probability of wrong classification = %.2f%%")
            else: kNNExampleAccuracy = (kNNExampleAccuracy, "Probability of correct classification = %.2f%%")
        else:
            kNNExampleAccuracy = None
            
        self.graph.insideColors = insideColors or self.classificationResults or kNNExampleAccuracy or self.outlierValues
        self.graph.clusterClosure = clusterClosure

        self.graph.updateData(self.attrX, self.attrY, self.attrColor, self.attrShape, self.attrSize, self.showColorLegend, self.attrLabel)
        #self.graph.update()  # don't know if this is necessary
        self.graph.repaint()

    
    # ##############################################################################################################################################################
    # SCATTERPLOT SETTINGS
    # ##############################################################################################################################################################

    #update status on progress bar - gets called by OWScatterplotGraph
    def updateProgress(self, current, total):
        self.progressBar.setTotalSteps(total)
        self.progressBar.setProgress(current)


    def replotCurves(self):
        for key in self.graph.curveKeys():
            symbol = self.graph.curveSymbol(key)
            self.graph.setCurveSymbol(key, QwtSymbol(symbol.style(), symbol.brush(), symbol.pen(), QSize(self.graph.pointWidth, self.graph.pointWidth)))
        self.graph.repaint()

    def setShowGridlines(self):
        self.graph.enableGridXB(self.showGridlines)
        self.graph.enableGridYL(self.showGridlines)

    def setAutoSendSelection(self):
        if self.autoSendSelection:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(0)
            self.sendSelections()
        else:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(1)

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_loop():
            self.colorSettings = (dlg.getColorSchemas(), dlg.getCurrentSchemeIndex(), dlg.getCurrentState())
            self.colorPalette = dlg.getColorPalette("colorPalette")
            self.graph.setCanvasBackground(dlg.getColor("Canvas"))
            self.graph.setGridPen(QPen(dlg.getColor("Grid")))
            self.updateGraph()

    def createColorDialog(self):
        c = OWDlgs.ColorPalette(self, "Color Palette")
        c.createColorPalette("colorPalette", "Continuous variable palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, "Canvas", "Canvas color", Qt.white)
        box.addSpace(5)
        c.createColorButton(box, "Grid", "Grid color", Qt.black)
        box.addSpace(5)
        box.adjustSize()
        if self.colorSettings:
            c.setColorSchemas(self.colorSettings[0], self.colorSettings[1])
            c.setCurrentState(self.colorSettings[2])
        else:
            c.setColorSchemas()
        return c

    def getColorPalette(self):
        return self.colorPalette

    def setMinimalGraphProperties(self):
        attrs = ["graph.pointWidth", "graph.showLegend", "graph.showClusters", "showXAxisTitle", "showYAxisTitle", "showGridlines", "graph.showAxisScale", "autoSendSelection"]
        self.oldSettings = dict([(attr, mygetattr(self, attr)) for attr in attrs])

        self.graph.pointWidth = 4
        self.graph.showLegend = 0
        self.graph.showClusters = 0
        self.graph.showXaxisTitle = 0
        self.graph.showYLaxisTitle = 0
        self.graph.showAxisScale = 0
        self.showGridlines = 0
        self.autoSendSelection = 0
        #self.updateValues()

    def restoreGraphProperties(self):
        if hasattr(self, "oldSettings"):
            for key in self.oldSettings:
                self.__setattr__(key, self.oldSettings[key])
        
    def destroy(self, dw = 1, dsw = 1):
        self.clusterDlg.hide()
        self.optimizationDlg.hide()
        OWWidget.destroy(self, dw, dsw)

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlot()
    a.setMainWidget(ow)

    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
