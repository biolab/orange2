"""
<name>Radviz</name>
<description>Shows data using radviz visualization method</description>
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
from OWClusterOptimization import *
import time
import OWToolbars, OWGUI, orngTest, orangeom, DESolver

class RadvizSolver(DESolver.DESolver):
    def __init__(self, radvizWidget, dim, pop):
        DESolver.DESolver.__init__(self, dim, pop) # superclass
        self.count = 0
        self.radviz = radvizWidget
        self.testGenerations = 20
        self.classes = [int(x.getclass()) for x in self.radviz.data]

        ai = self.radviz.graph.attributeNameIndex
        self.attrIndices = [ai[attr] for attr in self.radviz.getShownAttributeList()]
        self.data = Numeric.transpose(self.radviz.graph.scaledData).tolist()

    def EnergyFunction(self, trial, bAtSolution):
        anchorData = [(trial[2*i], trial[2*i+1], self.radviz.data.domain.attributes[i].name) for i in self.attrIndices]
        for (x,y,a) in anchorData:
            if x**2 + y**2 > 1: return 999999999999, 0
        E = orangeom.computeEnergy(self.data, self.classes, anchorData, self.attrIndices, self.radviz.attractG, -self.radviz.repelG)
        return E, 0


# #############################################################################
# class that represents kNN classifier that classifies examples based on top evaluated projections
class FreeVizClassifier(orange.Classifier):
    def __init__(self, radvizWidget, data):
        self.radvizWidget = radvizWidget

        keepMinMaxVals = self.radvizWidget.data != None and str(self.radvizWidget.data.domain.attributes) == str(data.domain.attributes)
        self.radvizWidget.cdata(data, keepMinMaxVals = keepMinMaxVals)

        self.radvizWidget.optimize()
        self.classifier = orange.P2NN(data, self.radvizWidget.graph.anchorData)

    # for a given example run argumentation and find out to which class it most often fall        
    def __call__(self, example, returnType):
        example.setclass(0)
        v = self.classifier(example, returnType)
        print "XX", v, v[0], v[1], type(v[0]), type(v[1]), "YY"
        return v
        
        

# #############################################################################
# learner that builds VizRankClassifier
class FreeVizLearner(orange.Learner):
    def __init__(self, radvizWidget):
        self.radvizWidget = radvizWidget
        self.name = "FreeViz"
        
    def __call__(self, examples, weightID = 0):
        return FreeVizClassifier(self.radvizWidget, examples)




###########################################################################################
##### WIDGET : Radviz visualization
###########################################################################################
class OWRadviz(OWWidget):
    settingsList = ["pointWidth", "jitterSize", "graphCanvasColor", "globalValueScaling", "showFilledSymbols", "scaleFactor",
                    "showLegend", "optimizedDrawing", "useDifferentSymbols", "autoSendSelection", "useDifferentColors",
                    "tooltipKind", "tooltipValue", "toolbarSelection", "showClusters", "VizRankClassifierName", "clusterClassifierName",
                    "attractG", "repelG", "hideRadius", "showAnchors", "showOptimizationSteps", "lockToCircle"]
    jitterSizeNums = [0.0, 0.01, 0.1,   0.5,  1,  2 , 3,  4 , 5, 7, 10, 15, 20]
    jitterSizeList = [str(x) for x in jitterSizeNums]
    scaleFactorNums = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    scaleFactorList = [str(x) for x in scaleFactorNums]
        
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Radviz", TRUE)

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata), ("Example Subset", ExampleTable, self.subsetdata, 1, 1), ("Selection", list, self.selection), ("Evaluation Results", orngTest.ExperimentResults, self.test_results)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass), ("Attribute Selection List", AttributeList), ("VizRank learner", orange.Learner), ("Cluster learner", orange.Learner), ("FreeViz learner", orange.Learner)]
        
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWRadvizGraph(self, self.mainArea)
        self.box.addWidget(self.graph)
        
        # cluster dialog
        self.clusterDlg = ClusterOptimization(self, self.signalManager, self.graph, "Radviz")
        self.graph.clusterOptimization = self.clusterDlg

        # optimization dialog
        self.optimizationDlg = kNNOptimization(self, self.signalManager, self.graph, "Radviz")
        self.graph.kNNOptimization = self.optimizationDlg
        self.optimizationDlg.optimizeGivenProjectionButton.show()
        self.freeVizLearner = FreeVizLearner(self)

        # variables
        self.pointWidth = 4
        self.attractG = 1.0
        self.repelG = 1.0
        self.hideRadius = 0
        self.showAnchors = 1
        self.lockToCircle = 0
        self.showOptimizationSteps = 0
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
        self.showClusters = 0
        self.VizRankClassifierName = "VizRank classifier (Scatterplot)"
        self.clusterClassifierName = "Visual cluster classifier (Scatterplot)"
        self.classificationResults = None

        # differential evolution
        self.differentialEvolutionPopSize = 100
        self.DERadvizSolver = None

        #load settings
        self.loadSettings()

        #GUI
        # add a settings dialog and initialize its values
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        #self.GeneralTab.setFrameShape(QFrame.NoFrame)
        self.SettingsTab = QVGroupBox(self)
        self.AnchorsTab = QVGroupBox(self)
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")
        self.tabs.insertTab(self.AnchorsTab, "Anchors")

        
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
        self.clusterDetectionDlgButton = OWGUI.button(self.attrOrderingButtons, self, "Cluster detection dialog", callback = self.clusterDlg.reshow)
        self.connect(self.clusterDlg.startOptimizationButton , SIGNAL("clicked()"), self.optimizeClusters)
        self.connect(self.clusterDlg.resultList, SIGNAL("selectionChanged()"),self.showSelectedCluster)
        
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
        OWGUI.checkBox(box3, self, 'showClusters', 'Show clusters', callback = self.updateValues, tooltip = "Show a line boundary around a significant cluster")

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
        # ANCHORS TAB
        # #####
        vbox = OWGUI.widgetBox(self.AnchorsTab, "Set Anchor Positions")
        hbox1 = OWGUI.widgetBox(vbox, orientation = "horizontal")
        #self.setAnchorButtons = QHButtonGroup("Set Anchor Positions", self.AnchorsTab)
        self.radialAnchorsButton = OWGUI.button(hbox1, self, "Radial", callback = self.radialAnchors)
        self.randomAnchorsButton = OWGUI.button(hbox1, self, "Random", callback = self.randomAnchors)
        self.manualPositioningButton = OWGUI.button(vbox, self, "Manual positioning")
        self.manualPositioningButton.setToggleButton(1)
        self.lockCheckbox = OWGUI.checkBox(vbox, self, "lockToCircle", "Restrain anchors to circle", callback = self.setLockToCircle)

        box = OWGUI.widgetBox(self.AnchorsTab, "Gradient Optimization")
        self.freeAttributesButton = OWGUI.button(box, self, "Single Step", callback = self.singleStep)
        self.freeAttributesButton = OWGUI.button(box, self, "Optimize", callback = self.optimize)
        self.freeAttributesButton = OWGUI.button(box, self, "Animate", callback = self.animate)
        self.freeAttributesButton = OWGUI.button(box, self, "Slow Animate", callback = self.slowAnimate)
        #self.setAnchorsButton = OWGUI.button(box, self, "Cheat", callback = self.setAnchors)

        box = OWGUI.widgetBox(self.AnchorsTab, "Differential Evolution")
        self.populationSizeEdit = OWGUI.lineEdit(box, self, "differentialEvolutionPopSize", "Population size: ", orientation = "horizontal", valueType = int)        
        self.createPopulationButton = OWGUI.button(box, self, "Create population", callback = self.createPopulation)
        self.evolvePopulationButton = OWGUI.button(box, self, "Evolve population", callback = self.evolvePopulation)
    
        box2 = OWGUI.widgetBox(self.AnchorsTab, "Forces")
        OWGUI.qwtHSlider(box2, self, "attractG", label="attractive", minValue=0, maxValue=9, step=1, ticks=0, callback=self.recomputeEnergy)
        OWGUI.qwtHSlider(box2, self, "repelG", label = "repellant", minValue=0, maxValue=9, step=1, ticks=0, callback=self.recomputeEnergy)
#        OWGUI.comboBoxWithCaption(box, self, "showOptimizationSteps", 'Show optimization', items = ["Yes", "Every 10", "No"])

        box = OWGUI.widgetBox(self.AnchorsTab, "Show Anchors")
        OWGUI.checkBox(box, self, 'showAnchors', 'Show anchors', callback = self.updateValues)
        OWGUI.qwtHSlider(box, self, "hideRadius", label="Hide radius", minValue=0, maxValue=9, step=1, ticks=0, callback = self.updateValues)
        self.freeAttributesButton = OWGUI.button(box, self, "Remove hidden attriubtes", callback = self.removeHidden)


        box = OWGUI.widgetBox(self.AnchorsTab, 1)
        self.energyLabel = QLabel(box, "Energy: ")
        
        # ####################################
        # K-NN OPTIMIZATION functionality
        self.connect(self.optimizationDlg.optimizeGivenProjectionButton, SIGNAL("clicked()"), self.optimizeGivenProjectionClick)
        self.connect(self.optimizationDlg.resultList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        self.connect(self.optimizationDlg.startOptimizationButton , SIGNAL("clicked()"), self.optimizeSeparation)

        self.connect(self.optimizationDlg.evaluateProjectionButton, SIGNAL("clicked()"), self.evaluateCurrentProjection)
        self.connect(self.optimizationDlg.showKNNCorrectButton, SIGNAL("clicked()"), self.showKNNCorect)
        self.connect(self.optimizationDlg.showKNNWrongButton, SIGNAL("clicked()"), self.showKNNWrong)

        self.connect(self.buttonUPAttr, SIGNAL("clicked()"), self.moveAttrUP)
        self.connect(self.buttonDOWNAttr, SIGNAL("clicked()"), self.moveAttrDOWN)

        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        self.connect(self.optimizationDlg.classifierNameEdit, SIGNAL("textChanged(const QString &)"), self.VizRankClassifierNameChanged)
        self.connect(self.clusterDlg.classifierNameEdit, SIGNAL("textChanged(const QString &)"), self.clusterClassifierNameChanged)
        
        # add a settings dialog and initialize its values
        self.activateLoadedSettings()
        self.resize(900, 700)
        self.send("FreeViz learner", self.freeVizLearner)


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
        self.graph.showClusters = self.showClusters
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        apply([self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection, self.zoomSelectToolbar.actionPolygonSelection][self.toolbarSelection], [])

        self.optimizationDlg.classifierName = self.VizRankClassifierName
        self.optimizationDlg.updateClassifierChanges()
        self.clusterDlg.classifierName = self.clusterClassifierName
        self.clusterDlg.classifierNameChanged(self.clusterClassifierName)

    def VizRankClassifierNameChanged(self, text):
        self.VizRankClassifierName = self.optimizationDlg.classifierName

    def clusterClassifierNameChanged(self, text):
        self.clusterClassifierName = self.clusterDlg.classifierName

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

    def showKNNCorect(self):
        self.optimizationDlg.showKNNWrongButton.setOn(0)
        self.showSelectedAttributes()

    # show quality of knn model by coloring accurate predictions with lighter color and bad predictions with dark color
    def showKNNWrong(self):
        self.optimizationDlg.showKNNCorrectButton.setOn(0) 
        self.showSelectedAttributes()


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
            
       
    # ################################################################################################
    # find projections where different class values are well separated
    def optimizeSeparation(self):
        if self.data == None: return
        if not self.data.domain.classVar:
            QMessageBox.critical( None, "VizRank Dialog", 'Projections can be evaluated only in datasets with a class value.', QMessageBox.Ok)
            return
        
        listOfAttributes = self.optimizationDlg.getEvaluatedAttributes(self.data)

        text = str(self.optimizationDlg.attributeCountCombo.currentText())
        if text == "ALL": maxLen = len(listOfAttributes)
        else:             maxLen = int(text)
        
        if self.optimizationDlg.getOptimizationType() == self.optimizationDlg.EXACT_NUMBER_OF_ATTRS: minLen = maxLen
        else: minLen = 3

        self.optimizationDlg.clearResults()

        possibilities = 0
        for i in range(minLen, maxLen+1):
            possibilities += combinations(i, len(listOfAttributes))*fact(i-1)/2
            
        self.graph.totalPossibilities = possibilities
        self.graph.triedPossibilities = 0
    
        if self.graph.totalPossibilities > 20000:
            self.warning("There are %s possible radviz projections with this set of attributes"% (createStringFromNumber(self.graph.totalPossibilities)))
        
        self.optimizationDlg.disableControls()
        
        try:
            self.graph.getOptimalSeparation(listOfAttributes, minLen, maxLen, self.optimizationDlg.addResult)
        except:
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # print the exception

        self.optimizationDlg.enableControls()
        self.optimizationDlg.finishedAddingResults()
    

    # ################################################################################################
    # find projections that have tight clusters of points that belong to the same class value
    def optimizeClusters(self):
        if self.data == None: return
        if not self.data.domain.classVar or not self.data.domain.classVar.varType == orange.VarTypes.Discrete:
            QMessageBox.critical( None, "Cluster Detection Dialog", 'Clusters can be detected only in data sets with a discrete class value', QMessageBox.Ok)
            return

        listOfAttributes = self.optimizationDlg.getEvaluatedAttributes(self.data)

        text = str(self.optimizationDlg.attributeCountCombo.currentText())
        if text == "ALL": maxLen = len(listOfAttributes)
        else:             maxLen = int(text)
        
        if self.clusterDlg.getOptimizationType() == self.clusterDlg.EXACT_NUMBER_OF_ATTRS: minLen = maxLen
        else: minLen = 3

        self.clusterDlg.clearResults()
        self.clusterDlg.clusterStabilityButton.setOn(0)
        self.clusterDlg.pointStability = None
        
        possibilities = 0
        for i in range(minLen, maxLen+1): possibilities += combinations(i, len(listOfAttributes))*fact(i-1)/2
            
        self.graph.totalPossibilities = possibilities
        self.graph.triedPossibilities = 0
    
        if self.graph.totalPossibilities > 20000:
            proj = str(self.graph.totalPossibilities)
            l = len(proj)
            for i in range(len(proj)-2, 0, -1):
                if (l-i)%3 == 0: proj = proj[:i] + "," + proj[i:]
            self.warning("There are %s possible radviz projections using currently visualized attributes"% (proj))
        
        self.clusterDlg.disableControls()
        try:
            self.graph.getOptimalClusters(listOfAttributes, minLen, maxLen, self.clusterDlg.addResult)
        except:
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # print the exception

        self.clusterDlg.enableControls()
        self.clusterDlg.finishedAddingResults()
   

    # ################################################################################################
    # try to find a better projection than the currently shown projection by adding other attributes to the projection and evaluating projections
    def optimizeGivenProjectionClick(self, numOfBestAttrs = -1, maxProjLen = -1):
        if numOfBestAttrs == -1:
            if self.data and len(self.data.domain.attributes) > 1000:
                (text, ok) = QInputDialog.getText('Qt Optimize Current Projection', 'How many of the best ranked attributes do you wish to test?')
                if not ok: return
                numOfBestAttrs = int(str(text))
            else: numOfBestAttrs = 10000
        self.optimizationDlg.disableControls()
        acc = self.graph.getProjectionQuality(self.getShownAttributeList())[0]
        # try to find a better separation than the one that is currently shown
        self.graph.optimizeGivenProjection(self.getShownAttributeList(), acc, self.optimizationDlg.getEvaluatedAttributes(self.data)[:numOfBestAttrs], self.optimizationDlg.addResult, restartWhenImproved = 1, maxProjectionLen = maxProjLen)
        self.optimizationDlg.enableControls()
        self.optimizationDlg.finishedAddingResults()


    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        if not self.data: return
        (selected, unselected, merged) = self.graph.getSelectionsAsExampleTables(self.getShownAttributeList())
    
        self.send("Selected Examples",selected)
        self.send("Unselected Examples",unselected)
        self.send("Example Distribution", merged)

    def sendShownAttributes(self):
        self.send("Attribute Selection List", [str(self.shownAttribsLB.text(i)) for i in range(self.shownAttribsLB.count())])


    # ####################################
    # free attribute anchors

    def radialAnchors(self):
        attrList = self.getShownAttributeList()
        phi = 2*math.pi/len(attrList)
        self.graph.anchorData = [(cos(i*phi), sin(i*phi), a) for i, a in enumerate(attrList)]
        self.graph.updateData(attrList)
        self.graph.repaint()
        
    def ranch(self, label):
        import random
        r = self.lockToCircle and 1.0 or 0.3+0.7*random.random()
        #print r
        phi = 2*pi*random.random()
        return (r*math.cos(phi), r*math.sin(phi), label)

    def randomAnchors(self):
        import random
        attrList = self.getShownAttributeList()
        self.graph.anchorData = [self.ranch(a) for a in attrList]
        if not self.lockToCircle:
            self.singleStep() # this won't do much, it's just for normalization
        else:
            self.graph.updateData(self.getShownAttributeList())
            self.graph.repaint()
            self.recomputeEnergy()

    def freeAttributes(self, iterations, steps, singleStep = False):
        attrList = self.getShownAttributeList()
        classes = [int(x.getclass()) for x in self.graph.rawdata]
        optimizer = self.lockToCircle and orangeom.optimizeAnchorsRadial or orangeom.optimizeAnchors
        ai = self.graph.attributeNameIndex
        attrIndices = [ai[label] for label in self.getShownAttributeList()]
    
        if not singleStep:
            minE = orangeom.computeEnergy(Numeric.transpose(self.graph.scaledData).tolist(), classes, self.graph.anchorData, attrIndices, self.attractG, -self.repelG)
        
        # repeat until less than 1% energy decrease in 5 consecutive iterations*steps steps
        noChange = 0
        while noChange < 5:
            for i in range(iterations):
                self.graph.anchorData, E = optimizer(Numeric.transpose(self.graph.scaledData).tolist(), classes, self.graph.anchorData, attrIndices, self.attractG, -self.repelG, steps)
                self.energyLabel.setText("Energy: %.3f" % E)
                self.energyLabel.repaint()
                self.graph.updateData(attrList)
                self.graph.repaint()
                if singleStep:
                    noChange = 5
                else:
                    if E > minE*0.99:
                        noChange += 1
                    else:
                        minE = E
                        noChange = 0

    def singleStep(self): self.freeAttributes(1, 1, True)
    def optimize(self):   self.freeAttributes(1, 100)
    def animate(self):   self.freeAttributes(10, 10)
    def slowAnimate(self):    self.freeAttributes(100, 1)

    def removeHidden(self):
        rad2 = (self.hideRadius/10)**2
        rem = 0
        newAnchorData = []
        for i, t in enumerate(self.graph.anchorData):
            if t[0]**2 + t[1]**2 < rad2:
                self.shownAttribsLB.removeItem(i-rem)
                self.hiddenAttribsLB.insertItem(t[2])
                rem += 1
            else:
                newAnchorData.append(t)
        self.graph.anchorData = newAnchorData
        attrList = self.getShownAttributeList()
        self.graph.updateData(attrList, 0)
        self.graph.repaint()
        self.recomputeEnergy()

    def setLockToCircle(self):
        if self.lockToCircle:
            anchorData = self.graph.anchorData
            for i, anchor in enumerate(anchorData):
                rad = math.sqrt(anchor[0]**2 + anchor[1]**2)
                anchorData[i] = (anchor[0]/rad, anchor[1]/rad) + anchor[2:]
            self.graph.updateData(self.getShownAttributeList(), 0)
            self.graph.repaint()
        self.recomputeEnergy()

    def recomputeEnergy(self):
        classes = [int(x.getclass()) for x in self.graph.rawdata]
        ai = self.graph.attributeNameIndex
        attrIndices = [ai[label] for label in self.getShownAttributeList()]
        E = orangeom.computeEnergy(Numeric.transpose(self.graph.scaledData).tolist(), classes, self.graph.anchorData, attrIndices, self.attractG, -self.repelG)
        #print E
        self.energyLabel.setText("Energy: %.3f" % E)
        self.energyLabel.repaint()

    def createPopulation(self):
        l = len(self.data.domain.attributes)
        self.DERadvizSolver = RadvizSolver(self, l * 2 , self.differentialEvolutionPopSize)
        Min = [0.0] * 2* l
        Max = [1.0] * 2* l

        self.DERadvizSolver.Setup(Min, Max, 0, 0.95, 1)
        
        
    def evolvePopulation(self):
        if not self.DERadvizSolver:
            QMessageBox.critical( None, "Differential evolution", 'To evolve a population you first have to create one by pressing "Create population" button', QMessageBox.Ok)

        self.DERadvizSolver.Solve(5)
        solution = self.DERadvizSolver.Solution()
        self.graph.anchorData = [(solution[2*i], solution[2*i+1], self.data.domain.attributes[i].name) for i in range(len(self.data.domain.attributes))]
        self.graph.updateData([attr.name for attr in self.data.domain.attributes], 0)

    # ####################################
    # show selected interesting projection
    def showSelectedAttributes(self):
        self.graph.removeAllSelections()
        val = self.optimizationDlg.getSelectedProjection()
        if not val: return
        (accuracy, other_results, tableLen, attrList, tryIndex, strList) = val
        
        values = self.classificationResults
        if self.optimizationDlg.showKNNCorrectButton.isOn() or self.optimizationDlg.showKNNWrongButton.isOn():
            shortData = self.graph.createProjectionAsExampleTable([self.graph.attributeNames.index(attr) for attr in attrList])
            values = self.optimizationDlg.kNNClassifyData(shortData)
            if self.optimizationDlg.showKNNCorrectButton.isOn(): values = [1.0 - val for val in values]
            clusterClosure = self.graph.clusterClosure
        else: clusterClosure = None

        self.showAttributes(attrList, values, clusterClosure)


    def showSelectedCluster(self):
        self.graph.removeAllSelections()
        val = self.clusterDlg.getSelectedCluster()
        if not val: return
        (value, closure, vertices, attrList, classValue, enlargedClosure, other, strList) = val

        if self.clusterDlg.clusterStabilityButton.isOn():
            validData = self.graph.getValidList([self.graph.attributeNames.index(attr) for attr in attrList])
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
        self.setShownAttributes(attrList)
        self.graph.updateData(attrList, setAnchors = 1, insideColors = insideColors, clusterClosure = clusterClosure)
        self.graph.repaint()
        self.sendShownAttributes()


        
    # ####################
    # LIST BOX FUNCTIONS
    # ####################
    def getShownAttributeList(self):
        return [str(self.shownAttribsLB.text(i)) for i in range(self.shownAttribsLB.count())]        
    
    def setShownAttributes(self, attributes):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()
        if self.data == None: return 0
        
        for attr in attributes: self.shownAttribsLB.insertItem(attr)
        for attr in self.data.domain:
            if attr.name not in attributes: self.hiddenAttribsLB.insertItem(attr.name)
        return 1

    # move selected attribute in "Attribute Order" list one place up
    def moveAttrUP(self):
        self.graph.removeAllSelections()
        self.graph.insideColors = None; self.graph.clusterClosure = None
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
        self.graph.insideColors = None; self.graph.clusterClosure = None
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
        self.graph.insideColors = None; self.graph.clusterClosure = None
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
        self.updateGraph(1)
        self.graph.replot()

    def removeAttribute(self):
        self.graph.removeAllSelections()
        self.graph.insideColors = None; self.graph.clusterClosure = None
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
        self.updateGraph(1)
        self.graph.replot()

    # #####################

    def updateGraph(self, setAnchors = 0, *args):
        self.graph.updateData(self.getShownAttributeList(), setAnchors)
        self.graph.update()
        self.repaint()

    # #########################
    # RADVIZ SIGNALS
    # #########################    
    
    # ###### CDATA signal ################################
    # receive new data and update all fields
    def cdata(self, data, clearResults = 1, keepMinMaxVals = 0):
        if data:
            name = ""
            if hasattr(data, "name"): name = data.name
            data = orange.Preprocessor_dropMissingClasses(data)
            data.name = name
        if self.data != None and data != None and self.data.checksum() == data.checksum(): return    # check if the new data set is the same as the old one
        exData = self.data
        self.data = data
        self.graph.setData(self.data, keepMinMaxVals)
        self.optimizationDlg.setData(data)  
        self.clusterDlg.setData(data, clearResults)
        self.graph.insideColors = None; self.graph.clusterClosure = None
        
        if not (data and exData and str(exData.domain.attributes) == str(data.domain.attributes)): # preserve attribute choice if the domain is the same                
            self.shownAttribsLB.clear()
            self.hiddenAttribsLB.clear()
            if data:
                for attr in data.domain.attributes[:10]: self.shownAttribsLB.insertItem(attr.name)
                if len(data.domain.attributes) > 10:
                    for attr in data.domain.attributes[10:]: self.hiddenAttribsLB.insertItem(attr.name)
                if data.domain.classVar: self.hiddenAttribsLB.insertItem(data.domain.classVar.name)
                
        self.updateGraph(1)
        self.sendSelections()
        self.sendShownAttributes()

    def subsetdata(self, data, update = 1):
        if self.graph.subsetData != None and data != None and self.graph.subsetData.checksum() == data.checksum(): return    # check if the new data set is the same as the old one
        self.graph.subsetData = data
        if update: self.updateGraph()
        self.optimizationDlg.setSubsetData(data)
        self.clusterDlg.setSubsetData(data)
       

    # ###### SELECTION signal ################################
    # receive info about which attributes to show
    def selection(self, list):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if self.data == None or list == None: return

        for attr in self.data.domain:
            if attr.name in list: self.shownAttribsLB.insertItem(attr.name)
            else:                 self.hiddenAttribsLB.insertItem(attr.name)

        self.updateGraph(1)


    def test_results(self, results):
        if results == None:
            self.classificationResults = None
        elif isinstance(results, orngTest.ExperimentResults):
            self.classificationResults = [results.results[i].probabilities[0][results.results[i].actualClass] for i in range(len(results.results))]
                
        self.showAttributes(self.getShownAttributeList(), self.classificationResults, clusterClosure)
        
        
    # ################################################

    # #########################
    # RADVIZ EVENTS
    # #########################
    def updateValues(self):
        self.graph.showClusters = self.showClusters
        self.graph.updateSettings(optimizedDrawing = self.optimizedDrawing, useDifferentSymbols = self.useDifferentSymbols, useDifferentColors = self.useDifferentColors)
        self.graph.updateSettings(showFilledSymbols = self.showFilledSymbols, tooltipKind = self.tooltipKind, tooltipValue = self.tooltipValue)
        self.graph.updateSettings(showLegend = self.showLegend, pointWidth = self.pointWidth, scaleFactor = self.scaleFactor)
        self.graph.updateSettings(hideRadius = self.hideRadius, showAnchors = self.showAnchors)
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
        self.graph.insideColors = None; self.graph.clusterClosure = None
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


    # ######################################################
    # functions used by OWClusterOptimization class
    # ######################################################
    def setMinimalGraphProperties(self):
        attrs = ["pointWidth", "showLegend", "showClusters", "autoSendSelection"]
        self.oldSettings = dict([(attr, getattr(self, attr)) for attr in attrs])
        self.pointWidth = 4
        self.showLegend = 0
        self.showClusters = 0
        self.autoSendSelection = 0
        self.graph.showAttributeNames = 0
        self.graph.setAxisScale(QwtPlot.xBottom, -1.05, 1.05, 1)
        self.graph.setAxisScale(QwtPlot.yLeft, -1.05, 1.05, 1)


    def restoreGraphProperties(self):
        if hasattr(self, "oldSettings"):
            for key in self.oldSettings:
                self.__setattr__(key, self.oldSettings[key])
        self.graph.showAttributeNames = 1
        self.graph.setAxisScale(QwtPlot.xBottom, -1.22, 1.22, 1)
        self.graph.setAxisScale(QwtPlot.yLeft, -1.13, 1.13, 1)



    
#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRadviz()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
