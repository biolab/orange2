from OWBaseWidget import *
from OWWidget import OWWidget
import os
import OWGUI, orngVisFuncts, OWQCanvasFuncts
from qtcanvas import QCanvas, QCanvasView
from orngMosaic import *
from orngScaleData import getVariableValuesSorted

mosaicMeasures = [("Pearson's Chi Square", CHI_SQUARE),
                  ("Cramer's Phi (correlation with class)", CRAMERS_PHI),
                  ("Information Gain (in % of class entropy removed)", INFORMATION_GAIN),
                  ("Distance Measure", DISTANCE_MEASURE),
                  ("Minimum Description Length", MDL),
                  ("Interaction Gain (in % of class entropy removed)", INTERACTION_GAIN),
                  ("Average Probability Of Correct Classification", AVERAGE_PROBABILITY_OF_CORRECT_CLASSIFICATION),
                  ("Gini index", GINI_INDEX),
                  ("CN2 Rules", CN2_RULES)]

allExamplesText = "<All examples>"

class OWMosaicOptimization(OWBaseWidget, orngMosaic):
    resultsListLenNums = [ 100 ,  250 ,  500 ,  1000 ,  5000 ,  10000, 20000, 50000, 100000]
    resultsListLenList = [str(x) for x in resultsListLenNums]
    settingsList = ["optimizationType", "attributeCount", "attrDisc", "qualityMeasure", "percentDataUsed", "ignoreTooSmallCells",
                    "timeLimit", "useTimeLimit", "VizRankClassifierName", "mValue", "probabilityEstimation",
                    "optimizeAttributeOrder", "optimizeAttributeValueOrder", "attributeOrderTestingMethod",
                    "classificationMethod", "classConfidence", "lastSaveDirName",
                    "projectionLimit", "useProjectionLimit"]

    percentDataNums = [ 5 ,  10 ,  15 ,  20 ,  30 ,  40 ,  50 ,  60 ,  70 ,  80 ,  90 ,  100 ]
    #evaluationTimeNums = [0.5, 1, 2, 5, 10, 20, 30, 40, 60, 80, 120]

    def __init__(self, mosaicWidget = None, signalManager = None):
        OWBaseWidget.__init__(self, None, signalManager, "Mosaic Evaluation Dialog", savePosition = True)
        orngMosaic.__init__(self)

        self.resize(390,620)

        if (int(qVersion()[0]) >= 3):
            self.setCaption("Mosaic Evaluation Dialog")
        else:
            self.setCaption("Qt Mosaic Evaluation Dialog")
        self.controlArea = QVBoxLayout(self)

        # loaded variables
        self.mosaicWidget = mosaicWidget
        self.showConfidence = 1
        self.optimizeAttributeOrder = 0
        self.optimizeAttributeValueOrder = 0
        self.VizRankClassifierName = "Mosaic Learner"
        self.useTimeLimit = 0
        self.useProjectionLimit = 0

        self.lastSaveDirName = os.getcwd()
        self.selectedClasses = []
        self.cancelArgumentation = 0
        self.useTimeLimit = 0

        # explorer variables
        self.wholeDataSet = None
        self.processingSubsetData = 0       # this is a flag that we set when we call mosaicWidget.setData function
        self.showDataSubset = 1
        self.invertSelection = 0
        self.mosaicSize = 300

        self.attrLenDict = {}
        self.shownResults = []

        self.loadSettings()

        self.tabs = QTabWidget(self, 'tabWidget')
        self.controlArea.addWidget(self.tabs)

        self.MainTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self)
        self.TreeTab = QVGroupBox(self)
        self.ManageTab = QVGroupBox(self)
        self.ArgumentationTab = QVGroupBox(self)
        self.ClassificationTab = QVGroupBox(self)

        self.tabs.insertTab(self.MainTab, "Main")
        self.tabs.insertTab(self.SettingsTab, "Settings")
        self.tabs.insertTab(self.ArgumentationTab, "Argumentation")
        self.tabs.insertTab(self.ClassificationTab, "Classification")
        self.tabs.insertTab(self.TreeTab, "Tree")
        self.tabs.insertTab(self.ManageTab, "Manage")

        # ###########################
        # MAIN TAB
        self.optimizationBox = OWGUI.widgetBox(self.MainTab, "Evaluate")
        self.buttonBox = OWGUI.widgetBox(self.optimizationBox, orientation = "horizontal")
        self.resultsBox = OWGUI.widgetBox(self.MainTab, "Projection list, most interesting projections first")
        self.optimizeOrderBox = OWGUI.widgetBox(self.MainTab, "Attribute and value order")
        self.optimizeOrderSubBox = OWGUI.widgetBox(self.optimizeOrderBox, orientation = "horizontal")
        self.buttonsBox = OWGUI.widgetBox(self.MainTab, box = 1)

        self.label1 = QLabel('Projections with ', self.buttonBox)
        self.optimizationTypeCombo = OWGUI.comboBox(self.buttonBox, self, "optimizationType", items = ["    exactly    ", "  maximum  "] )
        self.attributeCountCombo = OWGUI.comboBox(self.buttonBox, self, "attributeCount", items = range(1, 5), tooltip = "Evaluate only projections with exactly (or maximum) this number of attributes", sendSelectedValue = 1, valueType = int)
        self.attributeLabel = QLabel(' attributes', self.buttonBox)

        self.startOptimizationButton = OWGUI.button(self.optimizationBox, self, "Start Evaluating Projections", callback = self.evaluateProjections)
        f = self.startOptimizationButton.font(); f.setBold(1);   self.startOptimizationButton.setFont(f)
        self.stopOptimizationButton = OWGUI.button(self.optimizationBox, self, "Stop Evaluation", callback = self.stopEvaluationClick)
        self.stopOptimizationButton.setFont(f)
        self.stopOptimizationButton.hide()

        self.resultList = QListBox(self.resultsBox)
        self.resultList.setMinimumSize(200,200)
        self.connect(self.resultList, SIGNAL("selectionChanged()"), self.showSelectedAttributes)

        OWGUI.checkBox(self.optimizeOrderSubBox, self, "optimizeAttributeOrder", "Optimize order of attributes", callback = self.optimizeCurrentAttributeOrder, tooltip = "Order the visualized attributes so that it will enhance class separation")
        OWGUI.checkBox(self.optimizeOrderSubBox, self, "optimizeAttributeValueOrder", "Optimize order of attribute values", callback = self.optimizeCurrentAttributeOrder, tooltip = "Order also the values of visualized attributes so that it will enhance class separation.\nWARNING: This can take a lot of time when visualizing attributes with many values.")

        self.optimizeOrderButton = OWGUI.button(self.buttonsBox, self, "Optimize Current Attribute Order", callback = self.optimizeCurrentAttributeOrder, tooltip = "Optimize the order of currently visualized attributes", debuggingEnabled=0)


        # ##########################
        # SETTINGS TAB
        self.measureCombo = OWGUI.comboBox(self.SettingsTab, self, "qualityMeasure", box = "Measure of projection interestingness", items = [item[0] for item in mosaicMeasures], tooltip = "What is interesting?", callback = self.updateGUI)

        self.ignoreSmallCellsBox = OWGUI.widgetBox(self.SettingsTab, "Ignore small cells" )
        self.ignoreSmallCellsCombo = OWGUI.checkBox(self.ignoreSmallCellsBox, self, "ignoreTooSmallCells", "Ignore cells where expected number of cases is less than 5", tooltip = "Statisticians advise that in cases when the number of expected examples is less than 5 we ignore the cell \nsince it can significantly influence the chi-square value.")

        self.testingBox = OWGUI.widgetBox(self.SettingsTab, "Testing method")
        self.testingCombo = OWGUI.comboBox(self.testingBox, self, "testingMethod", items = ["10 fold cross validation", "70/30 separation 10 times "], tooltip = "Method for evaluating the class separation in the projection.")
        self.percentDataUsedCombo= OWGUI.comboBoxWithCaption(self.testingBox, self, "percentDataUsed", "Percent of data used: ", items = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], sendSelectedValue = 1, valueType = int, tooltip = "In case that we have a large dataset the evaluation of each projection can take a lot of time.\nWe can therefore use only a subset of randomly selected examples, evaluate projection on them and thus make evaluation faster.")

        OWGUI.comboBox(self.SettingsTab, self, "attrDisc", box = "Measure for ranking attributes", items = [val for (val, m) in discMeasures], callback = self.removeEvaluatedAttributes)

        self.testingCombo2 = OWGUI.comboBox(self.SettingsTab, self, "attributeOrderTestingMethod", box = "Testing method used for optimizing attribute orders", items = ["10 fold cross validation", "Learn and test on learn data"], tooltip = "Method used when evaluating different attribute orders.")

        self.stopOptimizationBox = OWGUI.widgetBox(self.SettingsTab, "When to stop evaluation or optimization?")
        OWGUI.checkWithSpin(self.stopOptimizationBox, self, "Time limit:                     ", 1, 1000, "useTimeLimit", "timeLimit", "  (minutes)", debuggingEnabled = 0)      # disable debugging. we always set this to 1 minute
        OWGUI.checkWithSpin(self.stopOptimizationBox, self, "Use projection count limit:  ", 1, 1000000, "useProjectionLimit", "projectionLimit", "  (projections)", debuggingEnabled = 0)

        # ##########################
        # ARGUMENTATION TAB
        self.argumentationBox = OWGUI.widgetBox(self.ArgumentationTab, "Arguments")
        self.findArgumentsButton = OWGUI.button(self.argumentationBox, self, "Find Arguments", callback = self.findArguments, tooltip = "Evaluate arguments for each possible class value using settings in the Classification tab.", debuggingEnabled = 0)
        f = self.findArgumentsButton.font(); f.setBold(1);  self.findArgumentsButton.setFont(f)
        self.stopArgumentationButton = OWGUI.button(self.argumentationBox, self, "Stop Searching", callback = self.stopArgumentationClick)
        self.stopArgumentationButton.setFont(f)
        self.stopArgumentationButton.hide()

        self.argumentsClassBox = OWGUI.widgetBox(self.ArgumentationTab, "Show arguments for class:", orientation = "horizontal")
        self.classValueList = OWGUI.comboBox(self.argumentsClassBox, self, "argumentationClassValue", tooltip = "Select the class value that you wish to see arguments for", callback = self.updateShownArguments)
        self.logitLabel = OWGUI.widgetLabel(self.argumentsClassBox, " ", labelWidth = 100)

        self.argumentBox = OWGUI.widgetBox(self.ArgumentationTab, "Arguments/Odds ratios for the selected class value")
        self.argumentList = QListBox(self.argumentBox)
        self.argumentList.setMinimumSize(200,200)
        self.connect(self.argumentList, SIGNAL("selectionChanged()"),self.argumentSelected)
        self.resultsDetailsBox = OWGUI.widgetBox(self.ArgumentationTab, "Shown details in arguments list" , orientation = "horizontal")
        self.showConfidenceCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showConfidence', '95% confidence interval', callback = self.updateShownArguments, tooltip = "Show confidence interval of the argument.")

        # ##########################
        # CLASSIFICATION TAB
        self.classifierNameEdit = OWGUI.lineEdit(self.ClassificationTab, self, 'VizRankClassifierName', box = ' Learner / Classifier Name ', tooltip='Name to be used by other widgets to identify your learner/classifier.')

        #self.argumentValueFormulaIndex = OWGUI.comboBox(self.ClassificationTab, self, "argumentValueFormula", box="Argument Value is Computed As ...", items=["1.0 x Projection Value", "0.5 x Projection Value + 0.5 x Predicted Example Probability", "1.0 x Predicted Example Probability"], tooltip=None)
        probBox = OWGUI.widgetBox(self.ClassificationTab, box = "Probability estimation")
        self.probCombo = OWGUI.comboBox(probBox, self, "probabilityEstimation", items = ["Relative Frequency", "Laplace", "m-Estimate"], callback = self.updateMestimateComboState)

        mValid = QDoubleValidator(self)
        mValid.setRange(0,10000,1)
        self.mEditBox = OWGUI.lineEdit(probBox, self, 'mValue', label='              Parameter for m-estimate:   ', orientation='horizontal', valueType = float, validator = mValid)

        b = OWGUI.widgetBox(self.ClassificationTab, "Evaluation time")
        OWGUI.checkWithSpin(b, self, "Use time limit:    ", 1, 1000, "useTimeLimit", "timeLimit", "(minutes)", debuggingEnabled = 0)      # disable debugging. we always set this to 1 minute
        classBox = OWGUI.widgetBox(self.ClassificationTab, "Class prediction settings")
        classMethodsCombo = OWGUI.comboBox(classBox, self, "classificationMethod", items = ["Top-ranked projections", "Semi-naive Bayes", "Naive Bayes with combining attribute values"], callback = self.updateClassMethodsCombo)

        # top projection settings
        self.classTopProjCount = OWGUI.widgetBox(classBox, orientation="horizontal")
        OWGUI.comboBoxWithCaption(self.classTopProjCount, self, "clsTopProjCount", "Number of top projections used:", tooltip = "How many of the top projections do you want to consider in class prediction?", items = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100], sendSelectedValue = 1, valueType = int)

        # semi naive bayes parameters
        self.classTau = OWGUI.widgetBox(classBox, orientation="horizontal")
        OWGUI.comboBoxWithCaption(self.classTau, self, "clsTau", "Treshold value (tau):", tooltip = "Value above which we join attribute values.", items = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], sendSelectedValue = 1, valueType = float)

        # combining attribute values
        self.classConfidenceBox = OWGUI.widgetBox(classBox, orientation="horizontal")
        OWGUI.separator(self.classConfidenceBox, 20, 0)
        OWGUI.spin(self.classConfidenceBox, self, "classConfidence", 0, 99, 1, label = 'Confidence Interval (%):    ', tooltip = 'Confidence interval used in deciding whether to use a set of attributes independently or dependently')

        OWGUI.button(self.ClassificationTab, self, "Resend Learner", callback = self.resendLearner, tooltip = "Resend learner with new settings. You need to press this \nonly when you are sending mosaic learner signal to other widgets.")

        # ##########################
        # TREE TAB
        subsetBox = OWGUI.widgetBox(self.TreeTab, "Example subset analysis")
        self.splitter = QSplitter(Qt.Vertical, subsetBox, "main")
        self.subsetTree = QListView(self.splitter)
        self.subsetTree.setRootIsDecorated(1)
        self.subsetTree.setAllColumnsShowFocus(1)
        self.subsetTree.addColumn('Visualized Attributes')
        self.subsetTree.addColumn('# inst.')
        self.subsetTree.setColumnWidth(0, 300)
        self.subsetTree.setColumnWidthMode(0, QListView.Maximum)
        self.subsetTree.setColumnAlignment(0, QListView.AlignLeft)
        self.subsetTree.setColumnWidth(1, 50)
        self.subsetTree.setColumnWidthMode(1, QListView.Manual)
        self.subsetTree.setColumnAlignment(1, QListView.AlignRight)
        self.connect(self.subsetTree, SIGNAL("selectionChanged(QListViewItem *)"), self.mtSelectedTreeItemChanged)
        self.connect(self.subsetTree, SIGNAL("rightButtonPressed(QListViewItem *, const QPoint &, int )"), self.mtSubsetTreeRemoveItemPopup)

        self.selectionsList = QListBox(self.splitter)
        self.connect(self.selectionsList, SIGNAL("selectionChanged()"), self.mtSelectedListItemChanged)
        self.connect(self.selectionsList, SIGNAL('doubleClicked(QListBoxItem *)'), self.mtSelectedListItemDoubleClicked)

        self.subsetItems = {}
        self.subsetUpdateInProgress = 0
        self.treeRoot = None

        explorerBox = OWGUI.widgetBox(self.TreeTab, 1)
        OWGUI.button(explorerBox, self, "Explore Currently Selected Examples", callback = self.mtEploreCurrentSelection, tooltip = "Visualize only selected examples and find interesting projections of them", debuggingEnabled=0)
        OWGUI.checkBox(explorerBox, self, 'showDataSubset', 'Show unselected data as example subset', tooltip = "This option determines what to do with the examples that are not selected in the projection.\nIf checked then unselected examples will be visualized in the same way as examples that are received through the 'Example Subset' signal.")

        self.mosaic = orngMosaic()
        autoBuildTreeBox = OWGUI.widgetBox(self.TreeTab, "Mosaic tree", orientation = "vertical")
        autoBuildTreeButtonBox = OWGUI.widgetBox(autoBuildTreeBox, orientation = "horizontal")
        self.autoBuildTreeButton = OWGUI.button(autoBuildTreeButtonBox, self, "Build Tree", callback = self.mtMosaicAutoBuildTree, tooltip = "Evaluate different mosaic diagrams and automatically build a tree of mosaic diagrams with clear class separation", debuggingEnabled = 0)
        OWGUI.button(autoBuildTreeButtonBox, self, "Visualize Tree", callback = self.mtVisualizeMosaicTree, tooltip = "Visualize a tree where each node is a mosaic diagram", debuggingEnabled = 0)
        OWGUI.lineEdit(autoBuildTreeBox, self, "mosaicSize", "Size of individual mosaic diagrams: ", orientation = "horizontal", tooltip = "What are the X and Y dimensions of individual mosaics in the tree?", valueType = int, validator = QIntValidator(self))

        loadSaveBox = OWGUI.widgetBox(self.TreeTab, "Load/Save mosaic tree", orientation = "horizontal")
        OWGUI.button(loadSaveBox, self, "Load", callback = self.mtLoadTree, tooltip = "Load a tree from a file", debuggingEnabled = 0)
        OWGUI.button(loadSaveBox, self, "Save", callback = self.mtSaveTree, tooltip = "Save tree to a file", debuggingEnabled = 0)

        self.subsetPopupMenu = QPopupMenu(self)
        self.subsetPopupMenu.insertItem("Explore currently selected examples", self.mtEploreCurrentSelection)
        self.subsetPopupMenu.insertItem("Find interesting projection", self.evaluateProjections)
        self.subsetPopupMenu.insertSeparator()
        self.subsetPopupMenu.insertItem("Remove node", self.mtRemoveSelectedItem)
        self.subsetPopupMenu.insertItem("Clear tree", self.mtInitSubsetTree)



        # ##########################
        # SAVE TAB
        self.visualizedAttributesBox = OWGUI.widgetBox(self.ManageTab, "Number of concurrently visualized attributes")
        self.dialogsBox = OWGUI.widgetBox(self.ManageTab, "Dialogs")
        self.manageResultsBox = OWGUI.widgetBox(self.ManageTab, "Manage projections")

        self.attrLenList = QListBox(self.visualizedAttributesBox)
        self.attrLenList.setSelectionMode(QListBox.Multi)
        self.attrLenList.setMinimumSize(60,60)
        self.connect(self.attrLenList, SIGNAL("selectionChanged()"), self.attrLenListChanged)

        self.buttonBox7 = OWGUI.widgetBox(self.dialogsBox, orientation = "horizontal")
        OWGUI.button(self.buttonBox7, self, "Attribute Ranking", self.attributeAnalysis, debuggingEnabled = 0)
        OWGUI.button(self.buttonBox7, self, "Attribute Interactions", self.interactionAnalysis, debuggingEnabled = 0)

        self.buttonBox8 = OWGUI.widgetBox(self.dialogsBox, orientation = "horizontal")
        OWGUI.button(self.buttonBox8, self, "Graph Projection Scores", self.graphProjectionQuality, debuggingEnabled = 0)
        OWGUI.button(self.buttonBox8, self, "Outlier Identification", self.identifyOutliers, debuggingEnabled = 0)

        self.buttonBox6 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.loadButton = OWGUI.button(self.buttonBox6, self, "Load", self.load, debuggingEnabled = 0)
        self.saveButton = OWGUI.button(self.buttonBox6, self, "Save", self.save, debuggingEnabled = 0)

        self.buttonBox5 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.clearButton = OWGUI.button(self.buttonBox5, self, "Clear results", self.clearResults)

        # ###########################
        self.statusBar = QStatusBar(self)
        self.controlArea.addWidget(self.statusBar)
        self.controlArea.activate()
        
        # reset some parameters if we are debugging so that it won't take too much time
        if orngDebugging.orngDebuggingEnabled:
            self.useTimeLimit = 1
            self.timeLimit = 0.3
            self.useProjectionLimit = 1
            self.projectionLimit = 100

        self.updateMestimateComboState()
        self.updateClassMethodsCombo()
        self.updateGUI()



    # ##############################################################
    # EVENTS
    def showSelectedAttributes(self, attrs = None):
        if not self.mosaicWidget: return
        if not attrs:
            projection = self.getSelectedProjection()
            if not projection: return
            (score, attrs, index, extraInfo) = projection
            if extraInfo:
                ruleVals = []   # for which values of the attributes do we have a rule
                for (q, a, vals) in extraInfo:
                    ruleVals.append([vals[a.index(attr)] for attr in attrs])
                self.mosaicWidget.activeRule = (attrs, ruleVals)
        valueOrder = None
        if self.optimizeAttributeOrder:
            self.resultList.setEnabled(0)
            self.optimizeCurrentAttributeOrder(attrs)
            self.resultList.setEnabled(1)
        else:
            self.mosaicWidget.setShownAttributes(attrs)
        self.resultList.setFocus()


    def optimizeCurrentAttributeOrder(self, attrs = None, updateGraph = 1):
        if str(self.optimizeOrderButton.text()) == "Optimize Current Attribute Order":
            self.cancelOptimization = 0
            self.optimizeOrderButton.setText("Stop Optimization")

            if not attrs:
                attrs = self.mosaicWidget.getShownAttributeList()

            bestPlacements = self.findOptimalAttributeOrder(attrs, self.optimizeAttributeValueOrder)
            if updateGraph:
                self.mosaicWidget.bestPlacements = bestPlacements
                if bestPlacements:
                    attrList, valueOrder = bestPlacements[0][1], bestPlacements[0][2]
                    self.mosaicWidget.setShownAttributes(attrList, customValueOrderDict = dict([(attrList[i], tuple(valueOrder[i])) for i in range(len(attrList))]) )

            self.optimizeOrderButton.setText("Optimize Current Attribute Order")
            return bestPlacements
        else:
            self.cancelOptimization = 1
            return []


    def updateGUI(self):
        if self.qualityMeasure in [CHI_SQUARE, CRAMERS_PHI]: self.ignoreSmallCellsBox.show()
        else:                                                self.ignoreSmallCellsBox.hide()
        if self.qualityMeasure == AVERAGE_PROBABILITY_OF_CORRECT_CLASSIFICATION: self.testingBox.show()
        else:   self.testingBox.hide()

    def updateMestimateComboState(self):
        self.mEditBox.setEnabled(self.probabilityEstimation == M_ESTIMATE)

    # based on selected classification method show or hide specific controls
    def updateClassMethodsCombo(self):
        self.classTopProjCount.hide()
        self.classTau.hide()
        self.classConfidenceBox.hide()

        if self.classificationMethod == MOS_TOPPROJ:
            self.classTopProjCount.show()
        elif self.classificationMethod == MOS_SEMINAIVE:
            self.classTau.show()
        elif self.classificationMethod == MOS_COMBINING:
            self.classConfidenceBox.show()


    # selected measure for attribute ranking has changed. recompute attribute importances
    def removeEvaluatedAttributes(self):
        self.evaluatedAttributes = None

    # result list can contain projections with different number of attributes
    # user clicked in the listbox that shows possible number of attributes of result list
    # result list must be updated accordingly
    def attrLenListChanged(self):
        # check which attribute lengths do we want to show
        if hasattr(self, "skipUpdate"): return

        self.attrLenDict = {}
        for i in range(self.attrLenList.count()):
            self.attrLenDict[int(str(self.attrLenList.text(i)))] = self.attrLenList.isSelected(i)
        self.updateShownProjections()

    def clearResults(self):
        orngMosaic.clearResults(self)
        self.resultList.clear()
        self.attrLenDict = {}
        self.attrLenList.clear()

    # ##############################################################
    # ##############################################################

    def updateShownProjections(self, *args):
        self.resultList.clear()
        self.resultListIndices = []
        self.shownResults = []

        for i in range(len(self.results)):
            if self.attrLenDict.has_key(len(self.results[i][ATTR_LIST])) and self.attrLenDict[len(self.results[i][ATTR_LIST])] == 1:
                self.resultList.insertItem("%.3f : %s" % (self.results[i][SCORE], self.buildAttrString(self.results[i][ATTR_LIST])))
                self.resultListIndices.append(i)
                self.shownResults.append(self.results[i])
        qApp.processEvents()

        if self.resultList.count() > 0: self.resultList.setCurrentItem(0)

    def setData(self, data, removeUnusedValues = 0):
        orngMosaic.setData(self, data, removeUnusedValues)

        self.setStatusBarText("")
        self.classValueList.clear()
        self.argumentList.clear()
        self.selectedClasses = []

        # for mosaic tree
        if self.processingSubsetData == 0:
            self.wholeDataSet = self.data        # we have to use self.data and not data, since in self.data we already have discretized attributes
            self.mtInitSubsetTree()

        if not self.data: return

        if hasattr(self.data, "name"): self.datasetName = data.name
        else: self.datasetName = ""

        if not (self.data.domain.classVar and self.data.domain.classVar.varType == orange.VarTypes.Discrete): return

        # add class values
        for val in getVariableValuesSorted(self.data, self.data.domain.classVar.name):
            self.classValueList.insertItem(val)
        self.updateShownArguments()

        if len(self.data.domain.classVar.values) > 0:
            self.classValueList.setCurrentItem(0)
        return self.data


    # ######################################################
    # Argumentation functions
    def findArguments(self, example = None, selectBest = 1, showClassification = 1):
        self.cancelArgumentation = 0

        self.argumentList.clear()
        self.arguments = [[] for i in range(self.classValueList.count())]

        if not example and not self.mosaicWidget.subsetData:
            QMessageBox.information( None, "Argumentation", 'To find arguments you first have to provide an example that you wish to classify. \nYou can do this by sending the example to the Mosaic display widget through the "Example Subset" signal.', QMessageBox.Ok + QMessageBox.Default)
            return None, None
        if len(self.results) == 0:
            QMessageBox.information( None, "Argumentation", 'To find arguments you first have to evaluate some projections by clicking "Start evaluating projections" in the Main tab.', QMessageBox.Ok + QMessageBox.Default)
            return None, None

        if not self.data:
            QMessageBox.critical(None,'No data','There is no data or no class value is selected in the Manage tab.',QMessageBox.Ok)
            return None, None

        if example == None: example = self.mosaicWidget.subsetData[0]

        self.findArgumentsButton.hide()
        self.stopArgumentationButton.show()

        classValue, dist = orngMosaic.findArguments(self, example)

        self.stopArgumentationButton.hide()
        self.findArgumentsButton.show()

        values = getVariableValuesSorted(self.data, self.data.domain.classVar.name)
        self.argumentationClassValue = values.index(classValue)     # activate the class that has the highest probability
        self.updateShownArguments()
        if self.argumentList.count() > 0 and selectBest: self.argumentList.setCurrentItem(0)

        if showClassification:
            s = '<nobr>Based on current classification settings, the example would be classified </nobr><br><nobr>to class <b>%(cls)s</b> with probability <b>%(prob).2f%%</b>.</nobr><br><nobr>Predicted class distribution is:</nobr><br>' % {"cls": str(classValue), "prob": max(dist)*100. / float(sum(dist))}
            for key in values:
                s += "<nobr>&nbsp &nbsp &nbsp &nbsp %s : %.2f%%</nobr><br>" % (key, dist[key]*100)
            QMessageBox.information(None, "Classification results", s, QMessageBox.Ok + QMessageBox.Default)

        return (classValue, dist)


    def finishedAddingResults(self):
        self.skipUpdate = 1

        self.attrLenDict = dict([(i,1) for i in range(self.attributeCount+1)])

        self.attrLenList.clear()
        for i in range(1,5):
            if self.attrLenDict.has_key(i):
                self.attrLenList.insertItem(str(i))

        self.attrLenList.selectAll(1)
        delattr(self, "skipUpdate")
        self.updateShownProjections()
        self.resultList.setCurrentItem(0)


    # ##############################################################
    # Loading and saving projection files
    # ##############################################################

    # save the list into a file - filename can be set if you want to call this function without showing the dialog
    def save(self, filename = None):
        if filename == None:
            # get file name
            if self.datasetName != "":
                filename = "%s - %s" % (os.path.splitext(os.path.split(self.datasetName)[1])[0], "Mosaic plots")
            else:
                filename = "%s" % (self.parentName)
            qname = QFileDialog.getSaveFileName( os.path.join(self.lastSaveDirName, filename), "Interesting mosaic visualizations (*.mproj)", self, "", "Save List of Visualizations")
            if qname.isEmpty(): return
            name = str(qname)
        else:
            name = filename
        self.setStatusBarText("Saving visualizations")

        # take care of extension
        if os.path.splitext(name)[1] != ".mproj":
            name = name + ".mproj"

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        orngMosaic.save(self, name)

        self.setStatusBarText("Saved %d visualizations" % (len(self.results)))


    # load projections from a file
    def load(self, name = None, ignoreCheckSum = 0):
        self.setStatusBarText("Loading visualizations")
        if self.data == None:
            QMessageBox.critical(None,'Load','There is no data. First load a data set and then load projection file',QMessageBox.Ok)
            return

        if name == None:
            name = QFileDialog.getOpenFileName( self.lastSaveDirName, "Interesting mosaic visualizations (*.mproj)", self, "", "Open List of Visualizations")
            if name.isEmpty(): return
            name = str(name)

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        count = orngMosaic.load(self, name, ignoreCheckSum)

        # update loaded results
        self.finishedAddingResults()
        self.setStatusBarText("Loaded %d visualizations" % (count))


    # disable all controls while evaluating projections
    def disableControls(self):
        self.startOptimizationButton.hide()
        self.stopOptimizationButton.show()
        self.SettingsTab.setEnabled(0)
        self.ManageTab.setEnabled(0)

    def enableControls(self):
        self.startOptimizationButton.show()
        self.stopOptimizationButton.hide()
        self.SettingsTab.setEnabled(1)
        self.ManageTab.setEnabled(1)


    def attrsToString(self, attrList):
        return ", ".join(attrList)

    # ######################################################
    # Mosaic tree functions
    # ######################################################
    # clear subset tree and create a new root
    def mtInitSubsetTree(self):
        self.subsetItems = {}
        self.subsetTree.clear()
        self.selectionsList.clear()
        self.treeRoot = None
        self.subsetTree.setColumnWidth(0, self.subsetTree.width() - self.subsetTree.columnWidth(1)-4)

        if self.wholeDataSet:
            root = QListViewItem(self.subsetTree, allExamplesText, str(len(self.wholeDataSet)))
            root.details = {"data": self.wholeDataSet, "exampleCount": len(self.wholeDataSet)}
            root.selections = {}
            self.treeRoot = root
            root.setOpen(1)
            self.subsetTree.insertItem(root)
            self.processingSubsetData = 1
            self.subsetTree.setSelected(root, 1)
            self.processingSubsetData = 0

    # find out which attributes are currently visualized, which examples are selected and what is the additional info
    def mtGetProjectionState(self, getSelectionIndices = 1):
        selectedIndices = None
        attrList = self.mosaicWidget.getShownAttributeList()
        exampleCount = self.mosaicWidget.data and len(self.mosaicWidget.data) or 0
        projDict = {"attrs": list(attrList), "exampleCount": exampleCount}
        selectionDict = {"selectionConditions": list(self.mosaicWidget.selectionConditions), "selectionConditionsHistorically": list(self.mosaicWidget.selectionConditionsHistorically)}
        if getSelectionIndices:
            selectedIndices = self.mosaicWidget.getSelectedExamples(asExampleTable = 0)
            selectionDict["selectedIndices"] = selectedIndices
        return attrList, selectedIndices, projDict, selectionDict

    # new element is added into the subsetTree
    def mtEploreCurrentSelection(self):
        if not self.wholeDataSet:
            return

        attrList, selectedIndices, projDict, selectionDict = self.mtGetProjectionState()

        if sum(selectedIndices) == 0:
            QMessageBox.information(self, "No data selection", "To explore a subset of examples you first have to select them in the projection.", QMessageBox.Ok)
            return

        selectedData = self.mosaicWidget.data.selectref(selectedIndices)
        unselectedData = self.mosaicWidget.data.selectref(selectedIndices, negate = 1)
        selectedTreeItem = self.subsetTree.selectedItem()     # current selection

        # add a new item into the list box
        newListItem = QListBoxText(self.selectionsList, self.mtSelectionsToString(selectionDict))
        newListItem.selections = selectionDict
        self.selectionsList.setSelected(newListItem, 1)

        # add a child into the tree view
        attrListStr = self.attrsToString(attrList)
        newTreeItem = QListViewItem(selectedTreeItem, attrListStr)
        newTreeItem.details = {"attrs": list(attrList), "exampleCount": len(selectedData)}
        newTreeItem.selections = selectionDict
        newTreeItem.setText(1, str(len(selectedData)))
        newTreeItem.setOpen(1)


    # a different attribute set was selected in mosaic. update the attributes in the selected node
    def mtUpdateState(self):
        if not self.wholeDataSet: return
        if self.processingSubsetData: return

        selectedTreeItem = self.subsetTree.selectedItem()
        selectedListItem = self.selectionsList.currentItem() != -1 and self.selectionsList.item(self.selectionsList.currentItem()) or None
        attrList, selectionIndices, projDict, selectionDict = self.mtGetProjectionState(getSelectionIndices = 0)
        if not selectedTreeItem: return

        # if this is the last element in the tree, then update the element's values
        if selectedTreeItem.firstChild() == None:
            selectedTreeItem.setText(0, self.attrsToString(attrList))
            selectedTreeItem.details.update(projDict)
            if selectedListItem:
                selectedListItem.selections = selectionDict
                selectedListItem.setText(self.mtSelectionsToString(selectionDict))
        # add a sibling if we changed any value
        else:
            # did we change the visualized attributes. If yes then we have to add a new node into the tree
            if 0 in [selectedTreeItem.details[key] == projDict[key] for key in projDict.keys()]:
                newTreeItem = QListViewItem(selectedTreeItem.parent() or self.subsetTree, self.attrsToString(attrList), str(selectedTreeItem.text(1)))
                newTreeItem.setOpen(1)
                newTreeItem.details = projDict
                newTreeItem.selections = {}
                self.subsetTree.setSelected(newTreeItem, 1)
                self.selectionsList.clear()


    # we selected a different item in the tree
    def mtSelectedTreeItemChanged(self, newSelection):
        if self.processingSubsetData:
            return
        self.processingSubsetData = 1

        indices = self.mtGetItemIndices(newSelection)
        selectedData = self.wholeDataSet
        unselectedData = orange.ExampleTable(self.wholeDataSet.domain)
        for ind in indices:
            unselectedData.extend(selectedData.selectref(ind, negate = 1))
            selectedData = selectedData.selectref(ind)

        # set data
        if self.invertSelection:
            temp = selectedData
            selectedData = unselectedData
            unselectedData = temp
        self.mosaicWidget.setData(selectedData)  #self.mosaicWidget.setData(selectedData, onlyDrilling = 1)
        if self.showDataSubset and len(unselectedData) > 0:
            self.mosaicWidget.setSubsetData(unselectedData)      #self.mosaicWidget.subsetData = unselectedData
        else:
            self.mosaicWidget.setSubsetData(None)
        self.mosaicWidget.handleNewSignals()

        self.selectionsList.clear()
        child = newSelection.firstChild()
        while child:
            selectionDict = child.selections
            newListItem = QListBoxText(self.selectionsList, self.mtSelectionsToString(selectionDict))
            newListItem.selections = selectionDict
            child = child.nextSibling()

        self.mosaicWidget.setShownAttributes(newSelection.details.get("attrs", None))
        self.mosaicWidget.updateGraph()
        self.processingSubsetData = 0

    # a new selection was selected in the selection list. update the graph
    def mtSelectedListItemChanged(self):
        selectedListItem = self.selectionsList.currentItem() != -1 and self.selectionsList.item(self.selectionsList.currentItem())
        if not selectedListItem:
            return

        selectionDict = selectedListItem.selections
        self.mosaicWidget.selectionConditions = list(selectionDict.get("selectionConditions", []))
        self.mosaicWidget.selectionConditionsHistorically = list(selectionDict.get("selectionConditionsHistorically", []))
        self.mosaicWidget.updateGraph()

    def mtSelectedListItemDoubleClicked(self, item):
        pos = self.selectionsList.currentItem()
        treeItem = self.subsetTree.selectedItem().firstChild()
        for i in range(pos):
            treeItem = treeItem.nextSibling()
        self.subsetTree.setSelected(treeItem, 1)
        self.mtSelectedTreeItemChanged(treeItem)

    def mtGetItemIndices(self, item):
        indices = []
        while item:
            ind = item.selections.get("selectedIndices", None)
            if ind:
                indices.insert(0, ind)        # insert indices in reverse order
            item = item.parent()
        return indices

    def mtGetData(self, indices):
        data = self.wholeDataSet
        unselectedData = orange.ExampleTable(data.domain)
        for ind in indices:
            unselectedData.extend(data.selectref(ind, negate = 1))
            data = data.selectref(ind)
        return data, unselectedData

    # popup menu items
    def mtRemoveSelectedItem(self):
        item = self.subsetTree.selectedItem()
        if not item:
            return
        parent = item.parent()
        if parent == None:
            self.mtInitSubsetTree()
        else:
            self.mtRemoveTreeItem(item)
            self.subsetTree.setSelected(parent, 1)
            self.mtSelectedTreeItemChanged(parent)

    def mtSubsetTreeRemoveItemPopup(self, item, point, i):
        self.subsetPopupMenu.popup(point, 0)

    def resizeEvent(self, ev):
        OWBaseWidget.resizeEvent(self, ev)
        self.subsetTree.setColumnWidth(0, self.subsetTree.width()-self.subsetTree.columnWidth(1)-4 - 20)


    def mtSelectionsToString(self, settings):
        attrCombs = ["-".join(sel) for sel in settings.get("selectionConditions", [])]
        return "+".join(attrCombs)


    # return actual item in the tree to that str(item) == strItem
    def mtStrToItem(self, strItem, currItem = -1):
        if currItem == -1:
            currItem = self.treeRoot
        if currItem == None:
            return None
        if str(currItem) == strItem:
            return currItem
        child = currItem.firstChild()
        if child:
            item = self.mtStrToItem(strItem, child)
            if item:
                return item
        return self.mtStrToItem(strItem, currItem.nextSibling())


    # save tree to a file
    def mtSaveTree(self, name = None):
        if name == None:
            qname = QFileDialog.getSaveFileName( os.path.join(self.lastSaveDirName, "explorer tree.tree"), "Explorer tree (*.tree)", self, "", "Save tree")
            if qname.isEmpty():
                return
            name = str(qname)
        self.lastSaveDirName = os.path.split(name)[0]

        tree = {}
        self.mtTreeToDict(self.treeRoot, tree)
        import cPickle
        f = open(name, "w")
        cPickle.dump(tree, f)
        f.close()

    # load tree from a file
    def mtLoadTree(self, name = None):
        self.subsetItems = {}
        self.subsetTree.clear()
        self.treeRoot = None

        if name == None:
            name = QFileDialog.getOpenFileName( self.lastSaveDirName, "Explorer tree (*.tree)", self, "", "Load tree")
            if name.isEmpty(): return
            name = str(name)

        self.lastSaveDirName = os.path.split(name)[0]
        import cPickle
        f = open(name, "r")
        tree = cPickle.load(f)
        self.mtDictToTree(tree, "None", self.subsetTree)
        root = self.subsetTree.firstChild()
        if root:
            self.treeRoot = root
            self.subsetTree.setSelected(root, 1)
            self.selectedTreeItemChanged(root)

    # generate a dictionary from the tree that can be pickled
    def mtTreeToDict(self, node, tree):
        if not node: return

        child = node.firstChild()
        if child:
            self.mtTreeToDict(child, tree)

        tree[str(node.parent())] = tree.get(str(node.parent()), []) + [(str(node), node.details, node.selections)]
        self.mtTreeToDict(node.nextSibling(), tree)

    # create a tree from a dictionary
    def mtDictToTree(self, tree, currItemKey, parentItem):
        if tree.has_key(currItemKey):
            children = tree[currItemKey]
            for (strChildNode, details, selections) in children:
                strAttrs = self.attrsToString(details["attrs"])
                exampleCount = details["exampleCount"]
                item = QListViewItem(parentItem, strAttrs, str(exampleCount))
                item.details = details
                item.selections = selections
                item.setOpen(1)
                self.mtDictToTree(tree, strChildNode, item)

    #################################################
    # build mosaic tree methods
    def mtMosaicAutoBuildTree(self):
        if str(self.autoBuildTreeButton.text()) != "Build Tree":
            self.mosaic.cancelTreeBuilding = 1
            self.mosaic.cancelEvaluation = 1
        else:
            try:
                self.mosaic.cancelTreeBuilding = 0
                self.mosaic.cancelEvaluation = 0
                self.autoBuildTreeButton.setText("Stop Building")
                qApp.processEvents()

                examples = self.mosaicWidget.data
                selectedItem = self.subsetTree.selectedItem()
                if selectedItem and selectedItem.parent() != None:
                    res = QMessageBox.information(self, "Tree Building", "Currently you are visualizing only a subset of examples. Do you want to build the tree\nonly for these examples or for all examples?", "Only for These", "For All Examples", "", 0, 1)
                    if res == 1:
                        examples = self.wholeDataSet
                        parent = self.subsetTree
                    else:
                        parent = selectedItem.parent()
                else:
                    parent = self.subsetTree
                    selections = selectedItem and selectedItem.selections or {}

                 #create a mosaic and use a classifier to generate a mosaic tree so that we don't set data to the main mosaic (which would mean that we would have to prevent the user from clicking the current tree)
                for setting in self.settingsList:
                    setattr(self.mosaic, setting, getattr(self, setting, None))
                if self.qualityMeasure == CN2_RULES:
                    self.mosaic.qualityMeasure == MDL
                self.mosaic.qApp = qApp
                root = MosaicTreeClassifier(self.mosaic, examples, self.setStatusBarText).mosaicTree

                # create tree items in the listview based on the tree in classifier.mosaicTree
                if root:
                    # if the selected item doesn't have any children we remove it and it will be replaced with the root of the tree that we generate
                    if not selectedItem:
                        self.subsetTree.clear()
                    elif selectedItem.firstChild() == None:
                        self.mtRemoveTreeItem(selectedItem)

                    item = QListViewItem(parent, self.attrsToString(root.attrs), str(len(root.branchSelector.data)))
                    item.details = {"attrs": root.attrs, "exampleCount": len(root.branchSelector.data)}
                    item.selections = selections
                    item.setOpen(1)
                    if parent == self.subsetTree:
                        self.treeRoot = item
                    self.mtGenerateTreeFromClassifier(root, item)
            except:
                import sys
                type, val, traceback = sys.exc_info()
                sys.excepthook(type, val, traceback)  # print the exception
            self.autoBuildTreeButton.setText("Build Tree")

    def mtGenerateTreeFromClassifier(self, treeNode, parentTreeItem):
        for key in treeNode.branches.keys():
            branch = treeNode.branches[key]
            strAttrs = self.attrsToString(branch.attrs)
            selections = treeNode.branchSelector.values[key]
            exampleCount = len(branch.branchSelector.data)
            item = QListViewItem(parentTreeItem, strAttrs, str(exampleCount))
            item.details = {"attrs": branch.attrs, "exampleCount": exampleCount}
            item.selections = {'selectionConditions': selections, 'selectionConditionsHistorically': [selections], "selectedIndices": self.mosaicWidget.getSelectedExamples(asExampleTable = 0, selectionConditions = selections, data = treeNode.branchSelector.data, attrs = treeNode.attrs)}
            item.setOpen(1)
            self.mtGenerateTreeFromClassifier(branch, item)

    # remove a tree item and also remove selections dict from its parent
    def mtRemoveTreeItem(self, item):
        parent = item.parent()
        if parent == None:
            parent = self.subsetTree
        parent.takeItem(item)

    def mtVisualizeMosaicTree(self):
        tree = {}
        self.mtTreeToDict(self.treeRoot, tree)
        #dialog = MosaicTreeDialog(self, self.mosaicWidget, self.signalManager)
        #dialog.visualizeTree(tree)
        #dialog.show()
        treeDialog = OWBaseWidget(self, self.signalManager, "Mosaic Tree")
        treeDialog.canvasLayout = QVBoxLayout(treeDialog)
        treeDialog.canvasWidget = QWidget(treeDialog)

        treeDialog.canvas = QCanvas(10000, 10000)
        treeDialog.canvasView = QCanvasView(treeDialog.canvas, treeDialog)
        treeDialog.canvasLayout.addWidget(treeDialog.canvasView)
        treeDialog.canvasLayout.activate()
        treeDialog.canvasView.show()
        treeDialog.resize(800, 800)
        treeDialog.move(0,0)

        xMosOffset = 80
        xMosaicSize = self.mosaicSize + 2 * 50     # we need some space also for text labels
        yMosaicSize = self.mosaicSize + 2 * 25

        mosaicCanvas = self.mosaicWidget.canvas
        mosaicCanvasView = self.mosaicWidget.canvasView
        cellSpace = self.mosaicWidget.cellspace
        self.mosaicWidget.canvas = treeDialog.canvas
        self.mosaicWidget.canvasView = treeDialog.canvasView
        self.mosaicWidget.cellspace = 5
        oldMosaicSelectionConditions = self.mosaicWidget.selectionConditions;                         self.mosaicWidget.selectionConditions = []
        oldMosaicSelectionConditionsHistorically = self.mosaicWidget.selectionConditionsHistorically; self.mosaicWidget.selectionConditionsHistorically = []

        nodeDict = {}
        rootNode = {"treeNode": tree["None"][0][0], "parentNode": None, "childNodes": []}
        rootNode.update(tree["None"][0][1])
        nodeDict[tree["None"][0][0]] = rootNode
        itemsToDraw = {0: [(rootNode,)]}
        treeDepth = 0
        canvasItems = {}

        # generate the basic structure of the tree
        while itemsToDraw.has_key(treeDepth):
            groups = itemsToDraw[treeDepth]
            xPos = 0
            for group in groups:
                for node in group:
                    node["currXPos"] = xPos
                    xPos += xMosaicSize        # next mosaic will be to the right

                    toDraw = []
                    children = tree.get(node["treeNode"], [])
                    for (strNode, details, selections) in children:
                        childNode = {"treeNode":strNode, "parentNode":node["treeNode"]}
                        childNode.update(details)
                        childNode.update(selections)
                        childNode["childNodes"] = []
                        node["childNodes"].append(childNode)
                        nodeDict[strNode] = childNode
                        toDraw.append(childNode)
                    if toDraw != []:
                        itemsToDraw[treeDepth+1] = itemsToDraw.get(treeDepth+1, []) + [toDraw]
            treeDepth += 1

        # fix positions of mosaic so that child nodes will be under parent
        changedPosition = 1
        while changedPosition:
            changedPosition = 0
            treeDepth = max(itemsToDraw.keys())
            while treeDepth > 0:
                groups = itemsToDraw[treeDepth]
                xPos = 0
                for group in groups:
                    # the current XPositions of the group might not be valid if we moved items in the previous groups. We therefore have to move the items if their xpos is smaller than xPos
                    if xPos > group[0]["currXPos"]:
                        for i in range(len(group)):
                            group[i]["currXPos"] = xPos + i* xMosaicSize

                    groupMidXPos = (group[0]["currXPos"] + group[-1]["currXPos"]) / 2
                    parentXPos = nodeDict[group[0]["parentNode"]]["currXPos"]
                    if abs(parentXPos - groupMidXPos) < 5:
                        xPos = group[-1]["currXPos"] + xMosaicSize
                        continue
                    changedPosition = 1        # we obviously have to move the parent or its children
                    if parentXPos < groupMidXPos:    # move the parent forward
                        self.mtRepositionNode(itemsToDraw[treeDepth-1], group[0]["parentNode"], groupMidXPos, xMosaicSize)
                    elif parentXPos > groupMidXPos:    # move the children backwards
                        xPos = self.mtRepositionNode(itemsToDraw[treeDepth], group[0]["treeNode"], parentXPos - (group[-1]["currXPos"] - group[0]["currXPos"])/2, xMosaicSize)
                treeDepth -= 1

        # visualize each mosaic diagram
        colors = self.mosaicWidget.selectionColorPalette

        maxX = 0
        maxY = 100 + (max(itemsToDraw.keys())+1) * yMosaicSize
        for depth in range(max(itemsToDraw.keys())+1):
            groups = itemsToDraw[depth]
            yPos = 50 + (depth > 0) * 50 + depth * yMosaicSize

            for group in groups:
                for node in group:
                    # create a dict with colors to be used to mark the selected rectangles
                    selectionDict = {}
                    for ind, selections in enumerate([child["selectionConditions"] for child in node["childNodes"]]):
                        for selection in selections:
                            selectionDict[tuple(selection)] = colors[ind]
                    data, unselectedData = self.mtGetData(self.mtGetItemIndices(self.mtStrToItem(node["treeNode"])))
                    # draw the mosaic
                    self.mosaicWidget.updateGraph(data, unselectedData, node["attrs"], erasePrevious = 0, positions = (node["currXPos"]+xMosOffset, yPos, self.mosaicSize), drawLegend = (depth == 0), drillUpdateSelection = 0, selectionDict = selectionDict)
                    maxX = max(maxX, node["currXPos"])

                    # draw a line between the parent and this node
                    if node["parentNode"]:
                        parent = nodeDict[node["parentNode"]]
                        nodeIndex = parent["childNodes"].index(node)
                        parentXPos = parent["currXPos"] + xMosaicSize/2 + 10*(-(len(parent["childNodes"])-1)/2 + nodeIndex)
                        OWQCanvasFuncts.OWCanvasLine(treeDialog.canvas, parentXPos, yPos - 30, node["currXPos"] + xMosaicSize/2, yPos - 10, penWidth = 4, penColor = colors[nodeIndex])

        treeDialog.canvas.resize(maxX + self.mosaicSize + 200, maxY)

        # restore the original canvas and canvas view
        self.mosaicWidget.canvas = mosaicCanvas
        self.mosaicWidget.canvasView = mosaicCanvasView
        self.mosaicWidget.cellspace = cellSpace
        self.mosaicWidget.selectionConditions = oldMosaicSelectionConditions
        self.mosaicWidget.selectionConditionsHistorically = oldMosaicSelectionConditionsHistorically
        treeDialog.show()

    # find the node nodeToMove in the groups and move it to newPos. reposition also all nodes that follow this node.
    def mtRepositionNode(self, groups, nodeToMove, newPos, xMosaicSize):
        found = 0
        for group in groups:
            for node in group:
                if node["treeNode"] == nodeToMove:        # we found the node to move
                    node["currXPos"] = newPos
                    found = 1
                    xPos = newPos + xMosaicSize
                elif found == 1:
                    node["currXPos"] = xPos
                    xPos += xMosaicSize
        return xPos     # return next valid position where to put a mosaic


    # ######################################################
    # Auxiliary functions
    # ######################################################

    def getSelectedProjection(self):
        currentItem = self.resultList.currentItem()
        if self.resultList.count() == 0 or currentItem == -1:
            return None
        return self.results[self.resultListIndices[currentItem]]      # we have to look into resultListIndices, since perhaps not all projections from the self.results are shown

    def stopEvaluationClick(self):
        self.cancelEvaluation = 1

    def isEvaluationCanceled(self):
        if self.cancelEvaluation:   return 1
        if self.useTimeLimit:       return orngMosaic.isEvaluationCanceled(self)

    def destroy(self, dw = 1, dsw = 1):
        self.saveSettings()

    def setStatusBarText(self, text):
        self.statusBar.message(text)
        qApp.processEvents()

    def insertArgument(self, argScore, error, attrList, index):
        s = "%.3f " % argScore
        if self.showConfidence and type(error) != tuple: s += "+-%.2f " % error
        s += "- " + self.buildAttrString(attrList)
        self.argumentList.insertItem(s, index)

    def updateShownArguments(self):
        self.argumentList.clear()
        if len(self.arguments) == 0: return
        classVal = str(self.classValueList.currentText())
        self.logitLabel.setText("log odds = %.2f" % self.logits.get(classVal, -1))

        if self.classificationMethod == MOS_COMBINING:
            self.logitLabel.show()
        else:
            self.logitLabel.hide()

        if not self.arguments.has_key(classVal): return
        for i in range(len(self.arguments[classVal])):
            (argScore, accuracy, attrList, index, error) = self.arguments[classVal][i]
            self.insertArgument(argScore, error, attrList, i)


    def argumentSelected(self):
        ind = self.argumentList.currentItem()
        classVal = str(self.classValueList.currentText())
        self.showSelectedAttributes(self.arguments[classVal][ind][2])


    def resendLearner(self):
        self.mosaicWidget.send("Learner", self.mosaicWidget.VizRankLearner)

    def stopArgumentationClick(self):
        self.cancelArgumentation = 1

    # ##############################################################
    # create different dialogs
    def interactionAnalysis(self):
        import OWkNNOptimization
        dialog = OWkNNOptimization.OWInteractionAnalysis(self, signalManager = self.signalManager)
        dialog.setResults(self.shownResults, OWkNNOptimization.VIZRANK_MOSAIC)
        dialog.show()


    def attributeAnalysis(self):
        import OWkNNOptimization
        dialog = OWkNNOptimization.OWGraphAttributeHistogram(self, signalManager = self.signalManager)
        dialog.setResults(self.shownResults, OWkNNOptimization.VIZRANK_MOSAIC)
        dialog.show()

    def graphProjectionQuality(self):
        import OWkNNOptimization
        dialog = OWkNNOptimization.OWGraphProjectionQuality(self, signalManager = self.signalManager)
        dialog.setResults(self.results, OWkNNOptimization.VIZRANK_MOSAIC)
        dialog.show()

    def identifyOutliers(self):
        import OWkNNOptimization
        dialog = OWkNNOptimization.OWGraphIdentifyOutliers(self, signalManager = self.signalManager, widget = self.mosaicWidget)
        dialog.setData(self.results, self.data, OWkNNOptimization.VIZRANK_MOSAIC)
        dialog.show()


#test widget appearance
if __name__=="__main__":
    import sys
    a = QApplication(sys.argv)
    ow = OWMosaicOptimization()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
