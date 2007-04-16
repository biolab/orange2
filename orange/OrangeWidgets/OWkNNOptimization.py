from OWBaseWidget import *
from OWWidget import OWWidget
import OWGUI, OWDlgs, OWGraphTools, numpy
from OWGraph import *
from orngVizRank import *
from orngScaleData import getVariableValuesSorted

class OWVizRank(VizRank, OWBaseWidget):
    settingsList = ["kValue", "resultListLen", "percentDataUsed", "qualityMeasure", "testingMethod",
                    "lastSaveDirName", "attrCont", "attrDisc", "showRank", "showAccuracy", "showInstances",
                    "evaluationAlgorithm", "evaluationTime", "learnerName",
                    "argumentCount", "optimizeBestProjection", "optimizeBestProjectionTime",
                    "locOptMaxAttrsInProj", "locOptAttrsToTry", "locOptProjCount", "locOptAllowAddingAttributes",
                    "useExampleWeighting", "useSupervisedPCA", "attrSubsetSelection", "optimizationType", "attributeCount",
                    "locOptOptimizeProjectionByPermutingAttributes", "timeLimit", "projectionLimit", "storeEachPermutation",
                    "boxLocalOptimization", "boxStopOptimization"]
    resultsListLenNums = [ 10, 100 ,  500 ,  1000 ,  5000 ,  10000, 20000, 50000, 100000, 500000 ]
    percentDataNums = [ 5 ,  10 ,  15 ,  20 ,  30 ,  40 ,  50 ,  60 ,  70 ,  80 ,  90 ,  100 ]
    kNeighboursNums = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 30,35, 40, 50, 60, 70, 80, 100, 120, 150, 200]
    argumentCounts = [1, 3, 5, 10, 15, 20, 30, 50, 100, 200]
    evaluationTimeNums = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 60, 80, 120]
    moreArgumentsNums = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

    def __init__(self, parentWidget = None, signalManager = None, graph = None, visualizationMethod = SCATTERPLOT, parentName = "Visualization widget"):
        VizRank.__init__(self, visualizationMethod, graph)
        OWBaseWidget.__init__(self, None, signalManager, "Optimization Dialog", savePosition = True)

        self.parentWidget = parentWidget
        self.parentName = parentName
        self.visualizationMethod = visualizationMethod
        self.setCaption("Qt VizRank Optimization Dialog")
        self.controlArea = QVBoxLayout(self)

        self.resultListLen = 1000
        self.cancelOptimization = 0
        self.cancelEvaluation = 0
        self.learnerName = "VizRank Learner"

        self.useTimeLimit = 0
        self.useProjectionLimit = 0
        self.evaluationTime = 10
        self.optimizeBestProjection = 0                     # do we want to try to locally improve the best projections
        self.optimizeBestProjectionTime = 10                 # how many minutes do we want to try to locally optimize the best projections

        self.boxLocalOptimization = 1
        self.boxStopOptimization = 1

        self.maxResultListLen = self.resultsListLenNums[len(self.resultsListLenNums)-1]
        #self.widgetDir = os.path.realpath(os.path.dirname(__file__)) + "/"
        self.lastSaveDirName = os.getcwd()

        self.evaluatedAttributes = None   # save last evaluated attributes
        self.evaluatedAttributesByClass = None

        self.showRank = 0
        self.showAccuracy = 1
        self.showInstances = 0
        self.shownResults = []
        self.attrLenDict = {}

        self.interactionAnalysisDlg = None
        self.identifyOutliersDlg = None
        self.attributeHistogramDlg = None

        self.loadSettings()
        self.attrCont = min(self.attrCont, 3)

        self.tabs = QTabWidget(self, 'tabWidget')
        self.controlArea.addWidget(self.tabs)

        self.MainTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self)
        self.ArgumentationTab = QVGroupBox(self)
##        self.ClassificationTab = QVGroupBox(self)
##        self.ExplorerTab = QVGroupBox(self)
        self.ManageTab = QVGroupBox(self)

        self.tabs.insertTab(self.MainTab, "Main")
        self.tabs.insertTab(self.SettingsTab, "Settings")
        self.tabs.insertTab(self.ArgumentationTab, "Argumentation")
##        self.tabs.insertTab(self.ClassificationTab, "Classification")
##        self.tabs.insertTab(self.ExplorerTab, "Explorer")
        self.tabs.insertTab(self.ManageTab, "Manage & Save")

        # ###########################
        # MAIN TAB
        self.optimizationBox = OWGUI.widgetBox(self.MainTab, "Evaluate")
        self.resultsBox = OWGUI.widgetBox(self.MainTab, "Projection List, Most Interesting Projections First")
        self.resultsDetailsBox = OWGUI.widgetBox(self.MainTab, "Shown Details in Projections List" , orientation = "horizontal")
        self.buttonBox = OWGUI.widgetBox(self.optimizationBox, orientation = "horizontal")

        if visualizationMethod != SCATTERPLOT:
            self.label1 = QLabel('Projections with ', self.buttonBox)
            self.optimizationTypeCombo = OWGUI.comboBox(self.buttonBox, self, "optimizationType", items = ["    exactly    ", "  maximum  "] )
            self.attributeCountCombo = OWGUI.comboBox(self.buttonBox, self, "attributeCount", items = range(3, 20), tooltip = "Evaluate only projections with exactly (or maximum) this number of attributes", sendSelectedValue = 1, valueType = int, debuggingEnabled = 0)
            self.attributeLabel = QLabel(' attributes', self.buttonBox)

        self.startOptimizationButton = OWGUI.button(self.optimizationBox, self, "Start Evaluating Projections", callback = self.evaluateProjections)
        f = self.startOptimizationButton.font(); f.setBold(1);   self.startOptimizationButton.setFont(f)
        self.optimizeGivenProjectionButton = OWGUI.button(self.optimizationBox, self, "Locally Optimize Best Projections", callback = self.optimizeBestProjections)

        self.resultList = QListBox(self.resultsBox)
        #self.resultList.setSelectionMode(QListBox.Extended)   # this would be nice if could be enabled, but it has a bug - currentItem doesn't return the correct value if this is on
        self.resultList.setMinimumSize(200,200)
        if self.parentWidget:
            self.connect(self.resultList, SIGNAL("selectionChanged()"), self.parentWidget.showSelectedAttributes)
            self._guiElements = getattr(self, "_guiElements", []) + [("listBox", self.resultList, None, self.parentWidget.showSelectedAttributes)]      # we want to debug this control

        self.showRankCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showRank', 'Rank', callback = self.updateShownProjections, tooltip = "Show projection ranks")
        self.showAccuracyCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showAccuracy', 'Score', callback = self.updateShownProjections, tooltip = "Show prediction accuracy of a k-NN classifier on the projection")
        self.showInstancesCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showInstances', '# Instances', callback = self.updateShownProjections, tooltip = "Show number of instances in the projection")

        # ##########################
        # SETTINGS TAB
        self.optimizationSettingsBox = OWGUI.widgetBox(self.SettingsTab, "VizRank Evaluation Settings")
        self.methodTypeCombo = OWGUI.comboBoxWithCaption(self.optimizationSettingsBox, self, "evaluationAlgorithm", "Projection evaluation method: ", tooltip = "Which learning method to use to use to evaluate given projections.", items = ["k-Nearest Neighbor", "Heuristic (very fast)"])
        self.attrKNeighboursEdit = OWGUI.lineEdit(self.optimizationSettingsBox, self, "kValue", "Number of neighbors (k):                 ", orientation = "horizontal", tooltip = "Number of neighbors used in k-NN algorithm to evaluate the projection", valueType = int, validator = QIntValidator(self))
        self.percentDataUsedCombo= OWGUI.comboBoxWithCaption(self.optimizationSettingsBox, self, "percentDataUsed", "Percent of data used: ", items = self.percentDataNums, sendSelectedValue = 1, valueType = int, tooltip = "In case that we have a large dataset the evaluation of each projection can take a lot of time.\nWe can therefore use only a subset of randomly selected examples, evaluate projection on them and thus make evaluation faster.")
        self.testingCombo = OWGUI.comboBox(self.optimizationSettingsBox, self, "testingMethod", label = "Testing method:                             ", orientation = "horizontal", items = ["Leave one out (slowest)", "10 fold cross validation", "Test on learning set (fastest)"], tooltip = "Method for evaluating the classifier. Slower are more accurate while faster give only a rough approximation.")
        OWGUI.checkBox(self.optimizationSettingsBox, self, 'useExampleWeighting', 'Use example weighting (in case of uneven class distribution)', tooltip = "Don't try all possible permutations of an attribute subset but only those,\nthat will most likely produce interesting projections.")
        if visualizationMethod != SCATTERPLOT:
            OWGUI.checkBox(self.optimizationSettingsBox, self, 'storeEachPermutation', 'Save all projections for each permutation of attributes', tooltip = "Do you want to see in the projection list all evaluated projections or only the best projection for each attribute permutation.\nUsually this value is unchecked.")

        if visualizationMethod == LINEAR_PROJECTION:
            OWGUI.checkBox(self.SettingsTab, self, 'useSupervisedPCA', 'Optimize class separation using supervised PCA', box = " Supervised PCA ")
        else: self.useSupervisedPCA = 0

        self.measureCombo = OWGUI.comboBox(self.SettingsTab, self, "qualityMeasure", box = " Measure of Classification Success ", items = ["Classification accuracy", "Average probability assigned to the correct class", "Brier score", "Area under curve (AUC)"], tooltip = "Measure to evaluate prediction accuracy of k-NN method on the projected data set.")

        self.attributeSelectionBox = OWGUI.widgetBox(self.SettingsTab, "Attribute Subset Selection")
        OWGUI.comboBox(self.attributeSelectionBox, self, "attrSubsetSelection", items = ["Deterministically using the selected attribute ranking measures", "Use gamma distribution and test all possible placements", "Use gamma distribution and test only one possible placement"])

        self.heuristicsSettingsBox = OWGUI.widgetBox(self.SettingsTab, "Measures for Attribute Ranking")
        OWGUI.comboBoxWithCaption(self.heuristicsSettingsBox, self, "attrCont", "For continuous attributes:", items = [val for (val, m) in contMeasures], callback = self.removeEvaluatedAttributes)
        OWGUI.comboBoxWithCaption(self.heuristicsSettingsBox, self, "attrDisc", "For discrete attributes:", items = [val for (val, m) in discMeasures], callback = self.removeEvaluatedAttributes)

        self.stopOptimizationBox = OWGUI.collapsableWidgetBox(self.SettingsTab, "When to Stop Evaluation?", self, "boxStopOptimization")
        OWGUI.checkWithSpin(self.stopOptimizationBox, self, "Use time limit:                     ", 1, 1000, "useTimeLimit", "timeLimit", "  (minutes)", debuggingEnabled = 0)      # disable debugging. we always set this to 1 minute
        OWGUI.checkWithSpin(self.stopOptimizationBox, self, "Use projection count limit:  ", 1, 1000000, "useProjectionLimit", "projectionLimit", "  (projections)", debuggingEnabled = 0)

        self.localOptimizationSettingsBox = OWGUI.collapsableWidgetBox(self.SettingsTab, "Local Optimization Settings", self, "boxLocalOptimization")
        bbb = OWGUI.checkBox(self.localOptimizationSettingsBox, self, 'locOptOptimizeProjectionByPermutingAttributes', 'Try improving projection by permuting attributes in projection')
        self.localOptimizationProjCountCombo = OWGUI.comboBoxWithCaption(self.localOptimizationSettingsBox , self, "locOptProjCount", "Number of best projections to optimize:           ", items = range(1,30), tooltip = "Specify the number of best projections in the list that you want to try to locally optimize.\nIf you select 1 only the currently selected projection will be optimized.", sendSelectedValue = 1, valueType = int)
        self.localOptimizationAttrsCount = OWGUI.lineEdit(self.localOptimizationSettingsBox, self, "locOptAttrsToTry", "Number of best attributes to try:                       ", orientation = "horizontal", tooltip = "How many of the top ranked attributes do you want to try in the projections?", valueType = int, validator = QIntValidator(self))
        locOptBox = OWGUI.widgetBox(self.localOptimizationSettingsBox, orientation = "horizontal")
        self.localOptimizationAddAttrsCheck  = OWGUI.checkBox(locOptBox, self, 'locOptAllowAddingAttributes', 'Allow adding attributes. Max attrs in proj:', tooltip = "Should local optimization only try to replace some attributes in a projection or is it also allowed to add new attributes?")
        self.localOptimizationProjMaxAttr    = OWGUI.comboBox(locOptBox, self, "locOptMaxAttrsInProj", items = range(3,50), tooltip = "What is the maximum number of attributes in a projection?", sendSelectedValue = 1, valueType = int)

        self.miscSettingsBox = OWGUI.widgetBox(self.SettingsTab, "Length of the Projection List")
        self.resultListCombo = OWGUI.comboBoxWithCaption(self.miscSettingsBox, self, "resultListLen", "Maximum length of projection list:   ", tooltip = "Maximum length of the list of interesting projections. This is also the number of projections that will be saved if you click Save button.", items = self.resultsListLenNums, callback = self.updateShownProjections, sendSelectedValue = 1, valueType = int)

        # ##########################
        # ARGUMENTATION TAB
        self.argumentationBox = OWGUI.widgetBox(self.ArgumentationTab, "Arguments")
        self.findArgumentsButton = OWGUI.button(self.argumentationBox, self, "Find Arguments", callback = self.findArguments, debuggingEnabled = 0)
        f = self.findArgumentsButton.font(); f.setBold(1);  self.findArgumentsButton.setFont(f)
        self.classValueList = OWGUI.comboBox(self.ArgumentationTab, self, "argumentationClassValue", box = "Arguments for Class:", tooltip = "Select the class value that you wish to see arguments for", callback = self.argumentationClassChanged, debuggingEnabled = 0)
        self.argumentBox = OWGUI.widgetBox(self.ArgumentationTab, "Arguments for the Selected Class Value")
        self.argumentList = QListBox(self.argumentBox)
        self.argumentList.setMinimumSize(200,200)
        self.connect(self.argumentList, SIGNAL("selectionChanged()"),self.argumentSelected)

##        # ##########################
##        # CLASSIFICATION TAB
##        if self.parentWidget:
##            self.classifierNameEdit = OWGUI.lineEdit(self.ClassificationTab, self, 'parentWidget.VizRankLearnerName', box = 'Learner / Classifier Name', tooltip='Name to be used by other widgets to identify your learner/classifier.')
##
##        #self.argumentValueFormulaIndex = OWGUI.comboBox(self.ClassificationTab, self, "argumentValueFormula", box="Argument Value is Computed As ...", items=["1.0 x Projection Value", "0.5 x Projection Value + 0.5 x Predicted Example Probability", "1.0 x Predicted Example Probability"], tooltip=None)
##
##        b = OWGUI.widgetBox(self.ClassificationTab, "Evaluating Time")
##        self.evaluationTimeEdit = OWGUI.comboBoxWithCaption(b, self, "evaluationTime", "Time for evaluating projections (minutes):", tooltip = "What is the maximum time that the classifier is allowed for evaluating projections (learning)", items = self.evaluationTimeNums, sendSelectedValue = 1, valueType = float, debuggingEnabled = 0)
##        b2 = OWGUI.widgetBox(b, orientation = "horizontal")
##        self.optimizeBestProjectionCheck = OWGUI.checkBox(b2, self, "optimizeBestProjection", "Afterwards use local optimization for (minutes):", tooltip = "Do you want to try to locally optimize the best projection when the time for evaluating projections runs out?", debuggingEnabled = 0)
##        self.optimizeBestProjectionCombo = OWGUI.comboBox(b2, self, "optimizeBestProjectionTime", items = self.evaluationTimeNums, sendSelectedValue = 1, valueType = float, debuggingEnabled = 0)
##        projCountBox = OWGUI.widgetBox(self.ClassificationTab, "Projection Count")
##        self.argumentCountEdit = OWGUI.comboBoxWithCaption(projCountBox, self, "argumentCount", "Number of projections used when classifying:", tooltip = "What is the maximum number of projections (arguments) that will be used when classifying an example.", items = self.argumentCounts, sendSelectedValue = 1, valueType = int, debuggingEnabled = 0)


        # ##########################
        # SAVE & MANAGE TAB
        self.classesBox = OWGUI.widgetBox(self.ManageTab, "Select Class Values You Wish to Separate")
        self.classesBox.setFixedHeight(130)
        self.visualizedAttributesBox = OWGUI.widgetBox(self.ManageTab, "Number of Concurrently Visualized Attributes")
        self.dialogsBox = OWGUI.widgetBox(self.ManageTab, "Dialogs")
        self.manageResultsBox = OWGUI.widgetBox(self.ManageTab, "Manage Projections")

        self.classesList = QListBox(self.classesBox)
        self.classesList.setSelectionMode(QListBox.Multi)
        self.classesList.setMinimumSize(60,60)
        self.connect(self.classesList, SIGNAL("selectionChanged()"), self.classesListChanged)
        self._guiElements = getattr(self, "_guiElements", []) + [("listBox", self.classesList, None, self.classesListChanged)]      # we want to debug this control

        self.attrLenList = QListBox(self.visualizedAttributesBox)
        self.attrLenList.setSelectionMode(QListBox.Multi)
        self.attrLenList.setMinimumSize(60,60)
        self.connect(self.attrLenList, SIGNAL("selectionChanged()"), self.attrLenListChanged)
        self._guiElements = getattr(self, "_guiElements", []) + [("listBox", self.attrLenList, None, self.attrLenListChanged)]      # we want to debug this control

        #self.removeSelectedButton = OWGUI.button(self.buttonBox5, self, "Remove selection", self.removeSelected)
        #self.filterButton = OWGUI.button(self.buttonBox5, self, "Save best graphs", self.exportMultipleGraphs)

        self.buttonBox7 = OWGUI.widgetBox(self.dialogsBox, orientation = "horizontal")
        self.attributeRankingButton = OWGUI.button(self.buttonBox7, self, "Attribute Ranking", self.attributeAnalysis, debuggingEnabled = 0)
        self.attributeInteractionsButton = OWGUI.button(self.buttonBox7, self, "Attribute Interactions", self.interactionAnalysis, debuggingEnabled = 0)

        self.buttonBox8 = OWGUI.widgetBox(self.dialogsBox, orientation = "horizontal")
        self.projectionScoresButton = OWGUI.button(self.buttonBox8, self, "Graph Projection Scores", self.graphProjectionQuality, debuggingEnabled = 0)
        self.outlierIdentificationButton = OWGUI.button(self.buttonBox8, self, "Outlier Identification", self.identifyOutliers, debuggingEnabled = 0)

        self.buttonBox6 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.loadButton = OWGUI.button(self.buttonBox6, self, "Load", self.loadProjections, debuggingEnabled = 0)
        self.saveButton = OWGUI.button(self.buttonBox6, self, "Save", self.saveProjections, debuggingEnabled = 0)

        self.buttonBox9 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.saveBestButton = OWGUI.button(self.buttonBox9, self, "Save Best Graphs", self.exportMultipleGraphs, debuggingEnabled = 0)
        OWGUI.button(self.buttonBox9, self, "Remove Similar Projections", callback = self.removeTooSimilarProjections, debuggingEnabled = 0)

        self.buttonBox3 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        if self.parentWidget:
            self.evaluateProjectionButton = OWGUI.button(self.buttonBox3, self, 'Evaluate Projection', callback = self.evaluateCurrentProjection, debuggingEnabled = 0)
        self.reevaluateResults = OWGUI.button(self.buttonBox3, self, "Reevaluate Projections", callback = self.reevaluateAllProjections)

        self.buttonBox4 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.showKNNCorrectButton = OWGUI.button(self.buttonBox4, self, 'Show k-NN Correct', self.showKNNCorect)
        self.showKNNWrongButton = OWGUI.button(self.buttonBox4, self, 'Show k-NN Wrong', self.showKNNWrong)
        self.showKNNCorrectButton.setToggleButton(1); self.showKNNWrongButton.setToggleButton(1)

        self.buttonBox5 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.clearButton = OWGUI.button(self.buttonBox5, self, "Clear Results", self.clearResults)

        # ###########################
        self.statusBox = OWGUI.widgetBox(self, orientation = "horizontal")
        self.statusBar = QStatusBar(self.statusBox)
        self.controlArea.addWidget(self.statusBox)
        self.controlArea.activate()

        self.stopOptimizationBox.syncControls()     # show open or closed
        self.localOptimizationSettingsBox.syncControls()

        self.removeEvaluatedAttributes()

        self.setMinimumWidth(375)
        self.tabs.setMinimumWidth(375)
        self.resize(375, 700)


    # ##############################################################
    # EVENTS
    # ##############################################################

    # the heuristic checkbox is enabled only if the signal to noise OVA measure is selected
    def removeEvaluatedAttributes(self):
        # clear the list of evaluated attributes
        self.evaluatedAttributes = None
        self.evaluatedAttributesByClass = None


    # result list can contain projections with different number of attributes
    # user clicked in the listbox that shows possible number of attributes of result list
    # result list must be updated accordingly
    def attrLenListChanged(self):
        # check which attribute lengths do we want to show
        self.attrLenDict = {}
        for i in range(self.attrLenList.count()):
            intVal = int(str(self.attrLenList.text(i)))
            selected = self.attrLenList.isSelected(i)
            self.attrLenDict[intVal] = selected
        self.updateShownProjections()

    def classesListChanged(self):
        results = self.results
        self.clearResults()

        self.selectedClasses = self.getSelectedClassValues()
        if len(self.selectedClasses) in [self.classesList.count(), 0]:
            for i in range(len(results)):
                VizRank.insertItem(self, i, results[i][OTHER_RESULTS][0], results[i][OTHER_RESULTS], results[i][LEN_TABLE], results[i][ATTR_LIST], results[i][TRY_INDEX], results[i][GENERAL_DICT])
        else:
            funct = self.qualityMeasure != BRIER_SCORE and max or min
            for result in results:
                acc = 0.0; sum = 0.0
                for index in self.selectedClasses:
                    acc += result[OTHER_RESULTS][OTHER_PREDICTIONS][index] * result[OTHER_RESULTS][OTHER_DISTRIBUTION][index]
                    sum += result[OTHER_RESULTS][OTHER_DISTRIBUTION][index]
                VizRank.insertItem(self, self.findTargetIndex(acc/max(sum,1.), funct), acc/max(sum,1.), result[OTHER_RESULTS], result[LEN_TABLE], result[ATTR_LIST], result[TRY_INDEX], result[GENERAL_DICT])

        self.finishedAddingResults()

    def clearResults(self):
        VizRank.clearResults(self)
        self.shownResults = []
        self.resultList.clear()
        self.attrLenDict = {}
        self.attrLenList.clear()

    def clearArguments(self):
        VizRank.clearArguments(self)
        self.argumentList.clear()

    # remove projections that are selected
    def removeSelected(self):
        for i in range(self.resultList.count()-1, -1, -1):
            if self.resultList.isSelected(i):
                # remove from listbox and original list of results
                self.resultList.removeItem(i)
                self.shownResults.remove(self.shownResults[i])

    # ##############################################################
    # ##############################################################

    def getSelectedClassValues(self):
        selectedClasses = []
        for i in range(self.classesList.count()):
            if self.classesList.isSelected(i): selectedClasses.append(i)
        return selectedClasses


    # a function that is meaningful when visualizing using radviz or polyviz
    # it removes projections that don't have different at least two attributes in comparison with some better ranked projection
    def removeTooSimilarProjections(self, allowedPercentOfEqualAttributes = -1):
        if allowedPercentOfEqualAttributes == -1:
            (text, ok) = QInputDialog.getText('Qt Allowed Similarity', 'How many attributes can be present in some better projection for a projection to be still considered as different (in pecents. Default = 70)?')
            if not ok: return
            allowedPercentOfEqualAttributes = int(str(text))

        qApp.setOverrideCursor(QWidget.waitCursor)
        self.setStatusBarText("Removing similar projections")
        i=0
        while i < self.resultList.count():
            qApp.processEvents()
            if self.existsABetterSimilarProjection(i, allowedPercentOfEqualAttributes = allowedPercentOfEqualAttributes):
                self.results.pop(i)
                self.shownResults.pop(i)
                self.resultList.removeItem(i)
            else:
                i += 1

        self.setStatusBarText("")
        qApp.restoreOverrideCursor()


    def updateShownProjections(self, *args):
        if hasattr(self, "dontUpdate"): return

        self.resultList.clear()
        self.shownResults = []
        i = 0
        qApp.setOverrideCursor(QWidget.waitCursor)

        while self.resultList.count() < self.resultListLen and i < len(self.results):
            if self.attrLenDict[len(self.results[i][ATTR_LIST])] == 1:
                string = ""
                if self.showRank: string += str(i+1) + ". "
                if self.showAccuracy: string += "%.2f" % (self.results[i][ACCURACY])
                if not self.showInstances and self.showAccuracy: string += " : "
                elif self.showInstances: string += " (%d) : " % (self.results[i][LEN_TABLE])
                string += self.buildAttrString(self.results[i][ATTR_LIST], self.results[i][GENERAL_DICT].get("reverse", []))

                self.resultList.insertItem(string)
                self.shownResults.append(self.results[i])
            i+=1
        qApp.processEvents()
        qApp.restoreOverrideCursor()

        if self.resultList.count() > 0: self.resultList.setCurrentItem(0)

    # set value of k to sqrt(n)
    def setData(self, data):
        self.setStatusBarText("")

        VizRank.setData(self, data)   # set the k value, remove results, arguments, ...
        self.removeEvaluatedAttributes()
        self.classesList.clear()
        self.classValueList.clear()
        self.selectedClasses = []

        hasDiscreteClass = self.data != None and len(self.data) > 0 and self.data.domain.classVar != None and self.data.domain.classVar.varType == orange.VarTypes.Discrete
        self.startOptimizationButton.setEnabled(hasDiscreteClass)
        self.optimizeGivenProjectionButton.setEnabled(hasDiscreteClass)
        self.evaluateProjectionButton.setEnabled(hasDiscreteClass)
        self.showKNNCorrectButton.setEnabled(hasDiscreteClass)
        self.showKNNWrongButton.setEnabled(hasDiscreteClass)
        self.attributeRankingButton.setEnabled(hasDiscreteClass)
        self.attributeInteractionsButton.setEnabled(hasDiscreteClass)
        self.projectionScoresButton.setEnabled(hasDiscreteClass)
        self.outlierIdentificationButton.setEnabled(hasDiscreteClass)
        self.findArgumentsButton.setEnabled(hasDiscreteClass)

        if not hasDiscreteClass:
            return

        # add class values
        for i in range(len(data.domain.classVar.values)):
            self.classesList.insertItem(data.domain.classVar.values[i])
            self.classValueList.insertItem(data.domain.classVar.values[i])
        self.classesList.selectAll(1)
        self.datasetName = getattr(data, "name", "")


    # given a dataset return a list of attributes where attribute are sorted by their decreasing importance for class discrimination
    def getEvaluatedAttributes(self, data):
        if self.evaluatedAttributes: return self.evaluatedAttributes

        self.setStatusBarText("Evaluating attributes...")
        qApp.setOverrideCursor(QWidget.waitCursor)

        try:
            if data.domain.classVar.varType == orange.VarTypes.Discrete:
                selectedClassesStr = [data.domain.classVar.values[i] for i in self.selectedClasses]
                nonSelectedClassesStr = []
                for val in data.domain.classVar.values:
                    if val not in selectedClassesStr: nonSelectedClassesStr.append(val)

                if len(nonSelectedClassesStr) > 0:
                    selection = orange.EnumVariable("Selection", values = selectedClassesStr + ["nonSelectedClass"])

                    shortData1 = data.select({data.domain.classVar.name: selectedClassesStr})
                    shortData2 = data.select({data.domain.classVar.name: nonSelectedClassesStr})

                    selection.getValueFrom = lambda ex, what: ex[data.domain.classVar]
                    d1 = orange.Domain(shortData1.domain.attributes + [selection])
                    data1 = orange.ExampleTable(d1, shortData1)

                    selection.getValueFrom = lambda ex, what: orange.Value(selection, "nonSelectedClass")
                    data2 = orange.ExampleTable(d1, shortData2)
                    data1.extend(data2)
                    data = data1

            # evaluate attributes using the selected attribute measure
            self.evaluatedAttributes = orngVisFuncts.evaluateAttributes(data, contMeasures[self.attrCont][1], discMeasures[self.attrDisc][1])
        except:
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # print the exception

        self.setStatusBarText("")
        qApp.restoreOverrideCursor()

        if self.evaluatedAttributes == None: return []
        else:             return self.evaluatedAttributes


    # insert new result - give parameters: accuracy of projection, number of examples in projection and list of attributes.
    def insertItem(self, index, accuracy, other_results, lenTable, attrList, tryIndex, generalDict = {}, updateStatusBar = 0):
        if index < self.maxResultListLen:
            self.results.insert(index, (accuracy, other_results, lenTable, attrList, tryIndex, generalDict))

        if index < self.resultListLen:
            string = ""
            if self.showRank: string += str(index+1) + ". "
            if self.showAccuracy: string += "%.2f" % (accuracy)
            if not self.showInstances and self.showAccuracy: string += " : "
            elif self.showInstances: string += " (%d) : " % (lenTable)

            string += self.buildAttrString(attrList, generalDict.get("reverse", []))

            self.resultList.insertItem(string, index)
            self.shownResults.insert(index, (accuracy, lenTable, other_results, attrList, tryIndex, generalDict))

            # remove worst projection if list is too long
            if self.resultList.count() > self.resultListLen:
                self.resultList.removeItem(self.resultList.count()-1)
                self.shownResults.pop()
            if updateStatusBar: self.setStatusBarText("Inserted %s projections" % (orngVisFuncts.createStringFromNumber(index)))
            qApp.processEvents()        # allow processing of other events


    def finishedAddingResults(self):
        self.cancelOptimization = 0
        self.cancelEvaluation = 0

        self.attrLenList.clear()
        self.attrLenDict = {}
        maxLen = -1
        for i in range(len(self.results)):
            if len(self.results[i][ATTR_LIST]) > maxLen:
                maxLen = len(self.results[i][ATTR_LIST])
        if maxLen == -1: return
        if maxLen == 2: vals = [2]
        else: vals = range(3, maxLen+1)
        for val in vals:
            self.attrLenList.insertItem(str(val))
            self.attrLenDict[val] = 1
        self.attrLenList.selectAll(1)
        self.resultList.setCurrentItem(0)

        # make sure that if the dialogs are shown we show the updated results
        if self.attributeHistogramDlg and self.attributeHistogramDlg.isVisible():
            self.attributeHistogramDlg.setResults(self.shownResults, VIZRANK_POINT)
        if self.interactionAnalysisDlg and self.interactionAnalysisDlg.isVisible():
            self.interactionAnalysisDlg.setResults(self.shownResults, VIZRANK_POINT)
        if self.identifyOutliersDlg and self.identifyOutliersDlg.isVisible():
            self.identifyOutliersDlg.setData(self.results, self.data, VIZRANK_POINT)


    # ##############################################################
    # kNNClassifyData - compute classification error for every example in table
    def kNNClassifyData(self, table):
        if len(table) == 0:
            return [], []

        # check if we have a discrete class
        if not table.domain.classVar or not table.domain.classVar.varType == orange.VarTypes.Discrete:
            return [], []

        if self.externalLearner: learner = self.externalLearner
        else:                    learner = self.createkNNLearner()
        results = apply(testingMethods[self.testingMethod], [[learner], table])

        returnTable = []

        if table.domain.classVar.varType == orange.VarTypes.Discrete:
            probabilities = numpy.zeros((len(table), len(table.domain.classVar.values)), numpy.float)
            lenClassValues = len(list(table.domain.classVar.values))
            if self.qualityMeasure in [AVERAGE_CORRECT, AUC]:       # for AUC we have no way of computing the prediction accuracy for each example
                for i in range(len(results.results)):
                    res = results.results[i]
                    returnTable.append(res.probabilities[0][res.actualClass])
                    probabilities[i] = res.probabilities[0]
            elif self.qualityMeasure == BRIER_SCORE:
                for i in range(len(results.results)):
                    res = results.results[i]
                    s = sum([val*val for val in res.probabilities[0]])
                    returnTable.append((s + 1 - 2*res.probabilities[0][res.actualClass])/float(lenClassValues))
                    probabilities[i] = res.probabilities[0]
            elif self.qualityMeasure == CLASS_ACCURACY:
                for i in range(len(results.results)):
                    res = results.results[i]
                    returnTable.append(res.probabilities[0][res.actualClass] == max(res.probabilities[0]))
                    probabilities[i] = res.probabilities[0]
            else:
                print "unknown quality measure for kNNClassifyData"
        else:
            probabilities = None
            # for continuous class we can't compute brier score and classification accuracy
            for res in results.results:
                if not res.probabilities[0]: returnTable.append(0)
                else:                        returnTable.append(res.probabilities[0].density(res.actualClass))

        return returnTable, probabilities


    # reevaluate projections in result list with the current VizRank settings (different k value, different measure of classification succes, ...)
    def reevaluateAllProjections(self):
        results = list(self.getShownResults())
        self.clearResults()

        self.parentWidget.progressBarInit()
        self.disableControls()

        testIndex = 0
        strTotal = orngVisFuncts.createStringFromNumber(len(results))
        for (acc, other, tableLen, attrList, tryIndex, generalDict) in results:
            if self.isOptimizationCanceled(): break
            testIndex += 1
            self.parentWidget.progressBarSet(100.0*testIndex/float(len(results)))

            table = self.graph.createProjectionAsExampleTable([self.attributeNameIndex[attr] for attr in attrList], settingsDict = generalDict)
            accuracy, other_results = self.kNNComputeAccuracy(table)
            self.addResult(accuracy, other_results, tableLen, attrList, tryIndex, generalDict)
            self.setStatusBarText("Reevaluated %s/%s projections..." % (orngVisFuncts.createStringFromNumber(testIndex), strTotal))

        self.setStatusBarText("")
        self.parentWidget.progressBarFinished()
        self.enableControls()
        self.finishedAddingResults()

    # evaluate knn accuracy on current projection
    def evaluateCurrentProjection(self):
        acc, other_results = self.getProjectionQuality(self.parentWidget.getShownAttributeList(), useAnchorData = 1)

        if self.data.domain.classVar.varType == orange.VarTypes.Continuous:
            QMessageBox.information( None, self.parentName, 'Mean square error of kNN model is %.2f'%(acc), QMessageBox.Ok + QMessageBox.Default)
        else:
            if self.qualityMeasure == CLASS_ACCURACY:
                QMessageBox.information( None, self.parentName, 'Classification accuracy of kNN model is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
            elif self.qualityMeasure == AVERAGE_CORRECT:
                QMessageBox.information( None, self.parentName, 'Average probability of correct classification is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
            elif self.qualityMeasure == AUC:
                QMessageBox.information( None, self.parentName, 'AUC is %.3f'%(acc), QMessageBox.Ok + QMessageBox.Default)
            elif self.qualityMeasure == BRIER_SCORE:
                QMessageBox.information( None, self.parentName, 'Brier score of kNN model is %.3f' % (acc), QMessageBox.Ok + QMessageBox.Default)
            else:
                QMessageBox.information( None, self.parentName, 'Accuracy of the model is is %.3f' % (acc), QMessageBox.Ok + QMessageBox.Default)



    # ##############################################################
    # Loading and saving projection files
    # ##############################################################
    def abortOperation(self):
        self.abortCurrentOperation = 1

    # save the list into a file - filename can be set if you want to call this function without showing the dialog
    def saveProjections(self):
        self.setStatusBarText("Saving projections")

        # get file name
        if self.datasetName != "":
            filename = "%s - %s" % (os.path.splitext(os.path.split(self.datasetName)[1])[0], self.parentName)
        else:
            filename = "%s" % (self.parentName)
        qname = QFileDialog.getSaveFileName( os.path.join(self.lastSaveDirName, filename), "Interesting projections (*.proj)", self, "", "Save Projections")
        if qname.isEmpty(): return
        name = str(qname)

        self.lastSaveDirName = os.path.split(name)[0]

        # show button to stop saving
        butt = OWGUI.button(self.statusBox, self, "Stop Saving", callback = self.abortOperation); butt.show()

        VizRank.save(self, name, self.shownResults, len(self.shownResults))

        self.setStatusBarText("Saved %s projections" % (orngVisFuncts.createStringFromNumber(len(self.shownResults))))
        butt.hide()


    # load projections from a file
    def loadProjections(self, name = None, ignoreCheckSum = 0, maxCount = -1):
        self.setStatusBarText("Loading projections")
        if self.data == None:
            QMessageBox.critical(None,'Load','There is no data. First load a data set and then load projection file',QMessageBox.Ok)
            return

        if name == None:
            name = QFileDialog.getOpenFileName( self.lastSaveDirName, "Interesting projections (*.proj)", self, "", "Open Projections")
            if name.isEmpty(): return
            name = str(name)

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        # show button to stop loading
        butt = OWGUI.button(self.statusBox, self, "Stop Loading", callback = self.abortOperation); butt.show()

        selectedClasses, count = VizRank.load(self, name, ignoreCheckSum, maxCount)

        self.dontUpdate = 1
        if self.data and self.data.domain.classVar and self.data.domain.classVar.varType == orange.VarTypes.Discrete:
            for i in range(len(self.data.domain.classVar.values)): self.classesList.setSelected(i, i in selectedClasses)
        del self.dontUpdate
        self.finishedAddingResults()

        self.setStatusBarText("Loaded %s projections" % (orngVisFuncts.createStringFromNumber(count)))
        butt.hide()

    def showKNNCorect(self):
        self.showKNNWrongButton.setOn(0)
        if self.parentWidget: self.parentWidget.updateGraph()

    # show quality of knn model by coloring accurate predictions with lighter color and bad predictions with dark color
    def showKNNWrong(self):
        self.showKNNCorrectButton.setOn(0)
        if self.parentWidget: self.parentWidget.updateGraph()


    # disable all controls while evaluating projections
    def disableControls(self):
        for control in [self.buttonBox, self.resultsDetailsBox, self.optimizeGivenProjectionButton, self.SettingsTab, self.ManageTab, self.ArgumentationTab]:
            control.setEnabled(0)

    def enableControls(self):
        for control in [self.buttonBox, self.resultsDetailsBox, self.optimizeGivenProjectionButton, self.SettingsTab, self.ManageTab, self.ArgumentationTab]:
            control.setEnabled(1)


    # ##############################################################
    # exporting multiple pictures
    def exportMultipleGraphs(self):
        (text, ok) = QInputDialog.getText('Qt Graph count', 'How many of the best projections do you wish to save?')
        if not ok: return
        self.bestGraphsCount = int(str(text))

        self.sizeDlg = OWDlgs.OWChooseImageSizeDlg(self.graph)
        self.sizeDlg.printButton.setEnabled(0)
        self.sizeDlg.saveMatplotlibButton.setEnabled(0)
        self.sizeDlg.disconnect(self.sizeDlg.saveImageButton, SIGNAL("clicked()"), self.sizeDlg.saveImage)
        self.sizeDlg.connect(self.sizeDlg.saveImageButton, SIGNAL("clicked()"), self.saveToFileAccept)
        self.sizeDlg.exec_loop()

    def saveToFileAccept(self):
        fileName = self.sizeDlg.getFileName("Graph", "Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)", ".png")
        if not fileName: return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        if ext == "":
        	ext = "PNG"  	# if no format was specified, we choose png
        	fileName = fileName + ".png"
        ext = ext.upper()

        (fil, extension) = os.path.splitext(fileName)
        for i in range(0, min(self.resultList.count(), self.bestGraphsCount)):
            self.resultList.setSelected(i, 1)
            self.graph.replot()
            name = fil + " (%02d, %.2f, %d)" % (i+1, self.shownResults[i][ACCURACY], self.shownResults[i][LEN_TABLE]) + extension
            self.sizeDlg.saveImage(name, closeDialog = 0)
        QDialog.accept(self.sizeDlg)

    # ##############################################################
    # create different dialogs
    def interactionAnalysis(self):
        if not self.interactionAnalysisDlg:
            self.interactionAnalysisDlg = OWInteractionAnalysis(self, signalManager = self.signalManager)
        self.interactionAnalysisDlg.setResults(self.shownResults, VIZRANK_POINT)
        self.interactionAnalysisDlg.show()


    def attributeAnalysis(self):
        if not self.attributeHistogramDlg:
            self.attributeHistogramDlg = OWGraphAttributeHistogram(self, signalManager = self.signalManager)
        self.attributeHistogramDlg.show()
        self.attributeHistogramDlg.setResults(self.shownResults, VIZRANK_POINT)

    def graphProjectionQuality(self):
        dialog = OWGraphProjectionQuality(self, signalManager = self.signalManager)
        dialog.show()
        dialog.setResults(self.results, VIZRANK_POINT)

    def identifyOutliers(self):
        if not self.identifyOutliersDlg:
            self.identifyOutliersDlg = OWGraphIdentifyOutliers(self, signalManager = self.signalManager, widget = self.parentWidget, graph = self.graph)
        self.identifyOutliersDlg.show()
        self.identifyOutliersDlg.setData(self.results, self.data, VIZRANK_POINT)

    # ######################################################
    # Auxiliary functions

    # from a list of attributes build a nice string with attribute names
    def buildAttrString(self, attrList, attrReverseList = []):
        if len(attrList) == 0: return ""

        if attrReverseList != []:
            strList = ""
            for i in range(len(attrList)):
                strList += attrList[i]
                if attrReverseList[i]: strList += "-"
                strList += ", "
            strList = strList[:-2]
        else:
            strList = reduce(lambda x,y: x+', '+y, attrList)
        return strList

    def getOptimizationType(self):
        return self.optimizationType

    def getQualityMeasure(self):
        return self.qualityMeasure

    def getQualityMeasureStr(self):
        if self.qualityMeasure ==0: return "Classification accuracy"
        elif self.qualityMeasure==1: return "Average probability of correct classification"
        else: return "Brier score"

    def getAllResults(self):
        return self.results

    def getShownResults(self):
        return self.shownResults

    def getSelectedProjection(self):
        if self.resultList.count() == 0: return None
        return self.shownResults[self.resultList.currentItem()]

    def evaluateProjections(self):
        if str(self.startOptimizationButton.text()) == "Start Evaluating Projections":
            if self.attributeCount >= 10 and not (self.useSupervisedPCA) and self.visualizationMethod != SCATTERPLOT and self.attrSubsetSelection != GAMMA_SINGLE and QMessageBox.critical(self, 'VizRank', 'You chose to evaluate projections with a high number of attributes. Since VizRank has to evaluate different placements\nof these attributes there will be a high number of projections to evaluate. Do you still want to proceed?','Continue','Cancel', '', 0,1):
                return
            if not self.data.domain.classVar or not self.data.domain.classVar.varType == orange.VarTypes.Discrete:
                QMessageBox.information( None, self.parentName, "Projections can be evaluated only for data with a discrete class.", QMessageBox.Ok + QMessageBox.Default)
                return
            self.startOptimizationButton.setText("Stop Evaluation")

            self.parentWidget.progressBarInit()
            self.disableControls()
            evaluatedProjections = VizRank.evaluateProjections(self)
            self.enableControls()
            self.parentWidget.progressBarFinished()

            secs = time.time() - self.startTime
            self.setStatusBarText("Finished evaluation (evaluated %s projections in %d min, %d sec)" % (orngVisFuncts.createStringFromNumber(evaluatedProjections), secs/60, secs%60))
            self.finishedAddingResults()
            qApp.processEvents()
            if self.parentWidget:
                self.parentWidget.showSelectedAttributes()
            self.startOptimizationButton.setText("Start Evaluating Projections")
        else:
            self.cancelEvaluation = 1
            self.cancelOptimization = 1


    def optimizeBestProjections(self, restartWhenImproved = 1):
        self.startOptimizationButton.setText("Stop Optimization")

        self.disableControls()
        evaluatedProjections = VizRank.optimizeBestProjections(self, restartWhenImproved)
        self.enableControls()

        secs = time.time() - self.startTime
        self.setStatusBarText("Finished evaluation (evaluated %s projections in %d min, %d sec)" % (orngVisFuncts.createStringFromNumber(evaluatedProjections), secs/60, secs%60))
        self.finishedAddingResults()
        qApp.processEvents()
        if self.parentWidget:
            self.parentWidget.showSelectedAttributes()
        self.startOptimizationButton.setText("Start Evaluating Projections")


    def isEvaluationCanceled(self):
        stop = self.cancelEvaluation
        if self.useTimeLimit:       stop = stop or (time.time() - self.startTime) / 60 >= self.timeLimit
        if self.useProjectionLimit: stop = stop or self.evaluatedProjectionsCount >= self.projectionLimit
        return stop

    def isOptimizationCanceled(self):
        stop = self.cancelOptimization
        if self.useTimeLimit:       stop = stop or (time.time() - self.startTime) / 60 >= self.timeLimit
        if self.useProjectionLimit: stop = stop or self.optimizedProjectionsCount >= self.projectionLimit
        return stop

    def destroy(self, dw = 1, dsw = 1):
        self.saveSettings()
        OWBaseWidget.destroy(self, dw, dsw)


    # ######################################################
    # Argumentation functions
    def findArguments(self, example = None, selectBest = 1, showClassification = 1):
        self.clearArguments()
        self.arguments = [[] for i in range(len(self.data.domain.classVar.values))]

        if not example and self.subsetData == None:
            QMessageBox.information( None, "VizRank Argumentation", 'To find arguments you first have to provide a new example that you wish to classify. \nYou can do this by sending the example to the visualization widget through the "Example Subset" signal.', QMessageBox.Ok + QMessageBox.Default)
            return (None,None)
        if len(self.shownResults) == 0:
            QMessageBox.information( None, "VizRank Argumentation", 'To find arguments you first have to evaluate some projections by clicking "Start evaluating projections" in the Main tab.', QMessageBox.Ok + QMessageBox.Default)
            return (None,None)
        if not example:
            example = self.subsetData[0]

        # call VizRank's function for finding arguments
        classValue, dist = VizRank.findArguments(self, example)

        if not self.arguments: return
        classIndex = self.classValueList.currentItem()
        for i in range(len(self.arguments[0])):
            prob, d, attrList, index = self.arguments[classIndex][i]
            self.argumentList.insertItem("%.3f - %s" %(prob, attrList), i)

        if self.argumentList.count() > 0 and selectBest:
            values = getVariableValuesSorted(self.data, self.data.domain.classVar.name)
            self.argumentationClassValue = values.index(classValue)     # activate the class that has the highest probability
            self.argumentationClassChanged()
            self.argumentList.setCurrentItem(0)
            self.argumentSelected()

        if showClassification or (example.getclass() and example.getclass().value != classValue):
            s = '<nobr>Based on current classification settings, the example would be classified </nobr><br><nobr>to class <b>%s</b> with probability <b>%.2f%%</b>.</nobr><br><nobr>Predicted class distribution is:</nobr><br>' % (classValue, dist[classValue]*100)
            for key in dist.keys(): s += "<nobr>&nbsp &nbsp &nbsp &nbsp %s : %.2f%%</nobr><br>" % (key, dist[key]*100)
            QMessageBox.information(None, "Classification results", s[:-4], QMessageBox.Ok + QMessageBox.Default)

        #qApp.processEvents()
        return classValue, dist


    def argumentationClassChanged(self):
        self.argumentList.clear()
        if len(self.arguments) == 0: return
        ind = self.classValueList.currentItem()
        for i in range(len(self.arguments[ind])):
            val = self.arguments[ind][i]
            self.argumentList.insertItem("%.2f - %s" %(val[0], val[2]))

    def argumentSelected(self):
        ind = self.argumentList.currentItem()
        classInd = self.classValueList.currentItem()
        generalDict = self.results[self.arguments[classInd][ind][3]][GENERAL_DICT]
        self.graph.updateData(self.arguments[classInd][ind][2], setAnchors = 1, XAnchors = generalDict.get("XAnchors"), YAnchors = generalDict.get("YAnchors"))

    def setStatusBarText(self, text):
        self.statusBar.message(text)
##        qApp.processEvents()


VIZRANK_POINT = 0
CLUSTER_POINT = 1
VIZRANK_MOSAIC = 2

# #############################################################################
# analyse the attributes that appear in the top projections. show how often do they appear also in other top projections
class OWInteractionAnalysis(OWWidget):
    settingsList = ["onlyLower", "rectColoring", "sortAttributesByQuality", "useGeneSets", "recentGeneSets"]
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "VizRank's Interaction Analysis", wantGraph = 1, savePosition = True)

        self.parent = parent
        self.attributeCount = 15
        self.projectionCount = 100
        self.rotateXAttributes = 1
        self.onlyLower = 0
        self.results = None
        self.dialogType = -1
        self.sortAttributesByQuality = 0
        self.rectColoring = 1

        self.recentGeneSets = []
        self.geneToSet, self.setToGenes = None, None
        self.useGeneSets = 0

        self.graph = OWGraph(self.mainArea)
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)
        self.box.activate()

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        b1 = OWGUI.widgetBox(self.controlArea, 'Number Of Attributes')
        b2 = OWGUI.widgetBox(self.controlArea, 'Number Of Projections')
        b3 = OWGUI.widgetBox(self.controlArea, "Settings")
        b4 = OWGUI.widgetBox(self.controlArea, "Use color to represent ...")
        b5 = OWGUI.widgetBox(self.controlArea, "Gene Sets")

        OWGUI.qwtHSlider(b1, self, 'attributeCount', minValue = 5, maxValue = 100, step=0, callback = self.updateGraph, precision = 0, maxWidth = 170)
        self.projectionCountSlider = OWGUI.qwtHSlider(b2, self, 'projectionCount', minValue = 5, maxValue = 1000, step = 5, callback = self.updateGraph, precision = 0, maxWidth = 170)
        OWGUI.checkBox(b3, self, 'rotateXAttributes', label = "Rotate X labels", callback = self.updateGraph)
        OWGUI.checkBox(b3, self, 'onlyLower', label = "Show only lower diagonal", callback = self.updateGraph)
        OWGUI.checkBox(b3, self, 'sortAttributesByQuality', 'Sort attributes according to quality', callback = self.updateGraph, tooltip = "Do you want to show the attributes as they are ranked according to some quality measure\nor as they appear in the top ranked projections?")

        OWGUI.comboBox(b4, self, "rectColoring", tooltip = "What should darkness of color of rectangles represent?", items = ["(don't use color)", "projection quality", "frequency of occurence in projections"], callback = self.updateGraph)

        OWGUI.checkBox(b5, self, "useGeneSets", label = "Use gene sets", callback = self.updateGraph)
        bb5 = OWGUI.widgetBox(b5, orientation  = "horizontal")
        self.fileCombo = OWGUI.comboBox(bb5, self, "geneSets")
        OWGUI.button(bb5, self, '...', callback = self.browseGeneFile, disabled=0, width=25)
        self.connect(self.fileCombo, SIGNAL('activated(int)'), self.selectGeneFile)

        b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b2.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b3.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b4.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b5.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))

        box = OWGUI.widgetBox(self.controlArea, "")
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.MinimumExpanding ))

        qApp.processEvents()
        self.updateGraph()
        self.updateGeneCombo()
        self.loadGeneSet()

    # ------- gene set functions ------------- #
    def selectGeneFile(self,n):
        name = self.recentGeneSets[n]
        self.recentGeneSets.remove(name)
        self.recentGeneSets.insert(0, name)
        if len(self.recentGeneSets) > 0:
            self.updateGeneCombo()
            self.loadGeneSet()

    def browseGeneFile(self):
        d = os.getcwd()
        if len(self.recentGeneSets) == 0 or self.recentGeneSets[0] == "(none)":
            startfile = "."
        else:
            startfile = self.recentGeneSets[0]
        filename = str(QFileDialog.getOpenFileName(startfile, 'Gene set files (*.gmt)\nAll files(*.*)', None,'Open Gene Set File'))
        if filename == "": return
        if filename in self.recentGeneSets: self.recentGeneSets.remove(filename)
        self.recentGeneSets.insert(0, filename)
        self.updateGeneCombo()
        self.loadGeneSet()

    def updateGeneCombo(self):
        self.fileCombo.clear()
        for file in self.recentGeneSets:
            self.fileCombo.insertItem(os.path.split(file)[1])

    def loadGeneSet(self):
        if len(self.recentGeneSets) == 0: return
        self.geneToSet, self.setToGenes = loadGeneSetFile(self.recentGeneSets[0])
        self.updateGraph()
    # ----------------------------------- #

    def setResults(self, results, dialogType):
        self.results = results
        self.dialogType = dialogType
        if results:
            self.projectionCountSlider.setScale(0, (len(results)/50) * 50, 0) # the third parameter for logaritmic scale
        if dialogType in [VIZRANK_POINT, CLUSTER_POINT]:
            if self.parent.attrCont == CONT_MEAS_S2NMIX:
                self.attributes, attrsByClass = orngVisFuncts.findAttributeGroupsForRadviz(self.parent.data, orngVisFuncts.S2NMeasureMix())
            else:
                self.attributes = orngVisFuncts.evaluateAttributes(self.parent.data, contMeasures[self.parent.attrCont][1], discMeasures[self.parent.attrDisc][1])
            self.ATTR_LIST = ATTR_LIST
            self.ACCURACY = ACCURACY
        elif dialogType == VIZRANK_MOSAIC:
            relieff = orange.MeasureAttribute_relief(k=10, m=50)
            self.attributes = orngVisFuncts.evaluateAttributes(self.parent.data, relieff, relieff)
            import orngMosaic
            self.ATTR_LIST = orngMosaic.ATTR_LIST
            self.ACCURACY = orngMosaic.SCORE
        self.updateGraph()

    def updateGraph(self):
        black = QColor(0,0,0)
        white = QColor(255,255,255)
        self.graph.clear()
        self.graph.removeMarkers()
        self.graph.tips.removeAll()

        if not self.results or self.dialogType not in [VIZRANK_POINT, CLUSTER_POINT, VIZRANK_MOSAIC]: return

        self.projectionCount = int(self.projectionCount)
        self.attributeCount = int(self.attributeCount)

        attributes = []
        attrDict = {}

        if self.sortAttributesByQuality:
            attributes = self.attributes[:self.attributeCount]
        else:
            attrCountDict = {}
            for index in range(min(self.projectionCount, len(self.results))):
                for attr in self.results[index][self.ATTR_LIST]:
                    attrCountDict[attr] = attrCountDict.get(attr, 0) + 1
            attrCounts = [(attrCountDict[attr], attr) for attr in attrCountDict.keys()]
            attrCounts.sort()
            attrCounts.reverse()
            attributes = [attr[1] for attr in attrCounts[:self.attributeCount]]

        for index in range(min(len(self.results), self.projectionCount)):
            attrs = self.results[index][self.ATTR_LIST]

            for i in range(len(attrs)):
                for j in range(i+1, len(attrs)):
                    if attrs[i] not in attributes or attrs[j] not in attributes: continue

                    Min = min(attrs[i], attrs[j])
                    Max = max(attrs[i], attrs[j])

                    # frequency of occurence
                    if self.rectColoring == 2:
                        attrDict[(Min, Max)] = attrDict.get((Min, Max), 0) + 1
                    # projection quality
                    elif not attrDict.has_key((Min, Max)):
                        attrDict[(Min, Max)] = self.results[index][self.ACCURACY]
            index += 1

        if self.rectColoring == 2:
            if attrDict.values() != []: best = max(attrDict.values())
            else: best = 1
            worst = -1  # we could use 0 but those with 1 would be barely visible
        else:
            best = self.results[0][self.ACCURACY]
            worst= self.results[min(len(self.results)-1, self.projectionCount)][self.ACCURACY]

        eps = 0.05 + 0.15 * self.useGeneSets
        eps2 = 0.05
        num = len(attributes)

        for x in range(num):
            for y in range(num-x):
                yy = num-y-1

                if attrDict.has_key((attributes[x], attributes[yy])):
                    val = attrDict[(attributes[x], attributes[yy])]
                elif attrDict.has_key((attributes[yy], attributes[x])):
                    val = attrDict[(attributes[yy], attributes[x])]
                else:
                    continue

                if not self.rectColoring:
                    color = black
                else:
                    v = int(255 - 255*((val-worst)/float(best - worst)))
                    color = QColor(v,v,v)

                s = ""
                if self.rectColoring == 1:  # projection quality
                    s = "Pair: <b>%s</b> and <b>%s</b><br>Best projection quality: <b>%.3f</b>" % (attributes[x], attributes[yy], val)
                elif self.rectColoring == 2:
                    s = "Pair: <b>%s</b> and <b>%s</b><br>Number of appearances: <b>%d</b>" % (attributes[x], attributes[yy], val)

                sharedGeneSets = []
                if self.useGeneSets and self.geneToSet:
                    set1 = getGeneSet(self.geneToSet, attributes[x])
                    set2 = getGeneSet(self.geneToSet, attributes[yy])
                    for s1 in set2:
                        if s1 in set1: sharedGeneSets.append(s1)

                if self.useGeneSets and self.geneToSet:
                    if sharedGeneSets != []:
                        s += "<hr>Shared gene sets: %s<br>" % (sharedGeneSets)
                    s += "<hr>Gene sets for individual genes:<br>&nbsp; &nbsp; <b>%s</b>: %s<br>&nbsp; &nbsp; <b>%s</b>: %s" % (attributes[x], getGeneSet(self.geneToSet, attributes[x]), attributes[yy], getGeneSet(self.geneToSet, attributes[yy]))

                key = self.graph.insertCurve(PolygonCurve(self.graph, QPen(color, 1), QBrush(color), [x-0.5+eps, x+0.5-eps, x+0.5-eps, x-0.5+eps, x-0.5+eps], [y+1-0.5+eps, y+1-0.5+eps, y+1+0.5-eps, y+1+0.5-eps, y+1-0.5+eps]))

                if self.useGeneSets and self.geneToSet and sharedGeneSets != []:
                    key = self.graph.insertCurve(PolygonCurve(self.graph, QPen(color, 1), QBrush(Qt.NoBrush), [x-0.5+eps2, x+0.5-eps2, x+0.5-eps2, x-0.5+eps2, x-0.5+eps2], [y+1-0.5+eps2, y+1-0.5+eps2, y+1+0.5-eps2, y+1+0.5-eps2, y+1-0.5+eps2]))
                if s != "":
                    self.graph.tips.addToolTip(x, y+1, s, 0.5, 0.5)

                if not self.onlyLower:
                    key = self.graph.insertCurve(PolygonCurve(self.graph, QPen(color, 1), QBrush(color), [num-1-0.5-y+eps, num-1-0.5-y+eps, num-1+0.5-y-eps, num-1+0.5-y-eps, num-1-0.5-y+eps], [num-0.5-x+eps, num+0.5-x-eps, num+0.5-x-eps, num-0.5-x+eps, num-0.5-x+eps]))

                    if self.useGeneSets and self.geneToSet and sharedGeneSets != []:
                        key = self.graph.insertCurve(PolygonCurve(self.graph, QPen(color, 1), QBrush(Qt.NoBrush), [num-1-0.5-y+eps2, num-1-0.5-y+eps2, num-1+0.5-y-eps2, num-1+0.5-y-eps2, num-1-0.5-y+eps2], [num-0.5-x+eps2, num+0.5-x-eps2, num+0.5-x-eps2, num-0.5-x+eps2, num-0.5-x+eps2]))
                    if s != "":
                        self.graph.tips.addToolTip(num-1-y, num-x, s, 0.5, 0.5)

        # draw empty boxes at the diagonal
        for x in range(num):
            key = self.graph.insertCurve(PolygonCurve(self.graph, QPen(black), QBrush(Qt.NoBrush), [x-0.5+2*eps2, x+0.5-2*eps2, x+0.5-2*eps2, x-0.5+2*eps2, x-0.5+2*eps2], [num-x-0.5+2*eps2, num-x-0.5+2*eps2, num-x+0.5-2*eps2, num-x+0.5-2*eps2, num-x-0.5+2*eps2]))

        """
        # draw x markers
        for x in range(num):
            marker = RotatedMarker(self.graph, attributes[x], x + 0.5, -0.3, 90*self.rotateXAttributes)
            mkey = self.graph.insertMarker(marker)
            if self.rotateXAttributes: self.graph.marker(mkey).setLabelAlignment(Qt.AlignLeft + Qt.AlignHCenter)
            else: self.graph.marker(mkey).setLabelAlignment(Qt.AlignCenter)

        # draw y markers
        for y in range(num):
            mkey = self.graph.insertMarker(attributes[num-y-1])
            self.graph.marker(mkey).setXValue(-0.3)
            self.graph.marker(mkey).setYValue(y + 0.5)
            self.graph.marker(mkey).setLabelAlignment(Qt.AlignLeft + Qt.AlignHCenter)

        self.graph.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
        self.graph.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        self.graph.axisScaleDraw(QwtPlot.xBottom).setTickLength(0, 0, 0)
        self.graph.axisScaleDraw(QwtPlot.yLeft).setTickLength(0, 0, 0)
        self.graph.axisScaleDraw(QwtPlot.xBottom).setOptions(0)
        self.graph.axisScaleDraw(QwtPlot.yLeft).setOptions(0)
        self.graph.setAxisScale(QwtPlot.xBottom, - 1.2 - 0.1*len(attributes) , num, 1)
        self.graph.setAxisScale(QwtPlot.yLeft, - 0.9 - 0.1*len(attributes) , num, 1)
        """

        self.graph.setAxisScaleDraw(QwtPlot.xBottom, OWGraphTools.DiscreteAxisScaleDraw(attributes))
        self.graph.axisScaleDraw(QwtPlot.xBottom).setTickLength(0, 0, 0)  # hide ticks
        self.graph.axisScaleDraw(QwtPlot.xBottom).setOptions(0)           # hide horizontal line representing x axis
        self.graph.setAxisMaxMajor(QwtPlot.xBottom, len(attributes))
        self.graph.setAxisMaxMinor(QwtPlot.xBottom, 0)
        self.graph.setAxisScale(QwtPlot.xBottom, -1, len(attributes), 1)
        if self.rotateXAttributes:
            self.graph.axisScaleDraw(QwtPlot.xBottom).setLabelRotation(-90)
            self.graph.axisScaleDraw(QwtPlot.xBottom).setLabelAlignment(Qt.AlignLeft)

        self.graph.setAxisScaleDraw(QwtPlot.yLeft, OWGraphTools.DiscreteAxisScaleDraw([""] + attributes[::-1]))
        self.graph.axisScaleDraw(QwtPlot.yLeft).setTickLength(0, 0, 0)  # hide ticks
        self.graph.axisScaleDraw(QwtPlot.yLeft).setOptions(0)           # hide horizontal line representing x axis
        self.graph.setAxisMaxMajor(QwtPlot.yLeft, len(attributes))
        self.graph.setAxisMaxMinor(QwtPlot.yLeft, 0)
        self.graph.setAxisScale(QwtPlot.yLeft, 0, len(attributes)+1, 1)


        self.graph.update()  # don't know if this is necessary
        self.graph.repaint()

    def hideEvent(self, ev):
        self.saveSettings()
        OWWidget.hideEvent(self, ev)


class OWGraphAttributeHistogram(OWWidget):
    settingsList = ["attributeCount", "projectionCount", "rotateXAttributes", "colorAttributes", "progressLines", "useGeneSets", "recentGeneSets"]
    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Attribute Histogram", wantGraph = 1, savePosition = True)

        self.results = None
        self.dialogType = -1

        self.graph = OWGraph(self.mainArea)
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)
##        self.box.activate()
        self.graph.showYLaxisTitle = 1

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        self.attributeCount = 10
        self.projectionCount = 100
        self.rotateXAttributes = 1
        self.colorAttributes = 1
        self.progressLines = 1
        self.useGeneSets = 0
        self.recentGeneSets = []
        self.useProjectionWeighting = 1

        b1 = OWGUI.widgetBox(self.controlArea, box = 1)
        b2 = OWGUI.widgetBox(self.controlArea, 'Number Of Attributes')
        b3 = OWGUI.widgetBox(self.controlArea, 'Number Of Projections')
        b4 = OWGUI.widgetBox(self.controlArea, "Gene Sets")
        box = OWGUI.widgetBox(self.controlArea)

        OWGUI.checkBox(b1, self, 'useProjectionWeighting', label = "Weight projections according to rank", callback = self.updateGraph, tooltip = "Projections contribute to attribute ranking according to their rank in the list of projections.")
        OWGUI.checkBox(b1, self, 'colorAttributes', label = "Color attributes according to class vote", callback = self.updateGraph)
        OWGUI.checkBox(b1, self, 'progressLines', label = "Show intermediate lines", callback = self.updateGraph)
        OWGUI.checkBox(b1, self, 'rotateXAttributes', label = "Show attribute names vertically", callback = self.updateGraph)
        OWGUI.qwtHSlider(b2, self, 'attributeCount', minValue = 5, maxValue = 100, step = 1, callback = self.updateGraph, precision = 0, maxWidth = 170)
        OWGUI.qwtHSlider(b3, self, 'projectionCount', minValue = 10, maxValue = 5000, step=10, callback = self.updateGraph, precision = 0, maxWidth = 170)
        OWGUI.checkBox(b4, self, "useGeneSets", label = "Use gene sets", callback = self.updateGraph)
        bb4 = OWGUI.widgetBox(b4, orientation  = "horizontal")
        self.fileCombo = OWGUI.comboBox(bb4, self, "geneSets")
        OWGUI.button(bb4, self, '...', callback = self.browseGeneFile, disabled=0, width=25)
        self.connect(self.fileCombo, SIGNAL('activated(int)'), self.selectGeneFile)

        b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b2.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b3.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b4.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.MinimumExpanding ))

        self.parent = parent
        self.results = None

        self.evaluatedAttributes = None
        self.evaluatedAttributesByClass = None
        qApp.processEvents()
        self.updateGeneCombo()
        self.loadGeneSet()

    # ------- gene set functions ------------- #
    def selectGeneFile(self,n):
        name = self.recentGeneSets[n]
        self.recentGeneSets.remove(name)
        self.recentGeneSets.insert(0, name)
        if len(self.recentGeneSets) > 0:
            self.updateGeneCombo()
            self.loadGeneSet()

    def browseGeneFile(self):
        d = os.getcwd()
        if len(self.recentGeneSets) == 0 or self.recentGeneSets[0] == "(none)":
            startfile = "."
        else:
            startfile = self.recentGeneSets[0]
        filename = str(QFileDialog.getOpenFileName(startfile, 'Gene set files (*.gmt)\nAll files(*.*)', None,'Open Gene Set File'))
        if filename == "": return
        if filename in self.recentGeneSets: self.recentGeneSets.remove(filename)
        self.recentGeneSets.insert(0, filename)
        self.updateGeneCombo()
        self.loadGeneSet()

    def updateGeneCombo(self):
        self.fileCombo.clear()
        for file in self.recentGeneSets:
            self.fileCombo.insertItem(os.path.split(file)[1])

    def loadGeneSet(self):
        if len(self.recentGeneSets) == 0: return
        self.geneToSet, self.setToGenes = loadGeneSetFile(self.recentGeneSets[0])
        self.updateGraph()
    # ----------------------------------- #

    def setResults(self, results, dialogType):
        self.results = results
        self.dialogType = dialogType
        if dialogType in [VIZRANK_POINT, CLUSTER_POINT]:
            self.ATTR_LIST = ATTR_LIST
        elif dialogType == VIZRANK_MOSAIC:
            import orngMosaic
            self.ATTR_LIST = orngMosaic.ATTR_LIST

        self.evaluatedAttributes = None
        self.evaluatedAttributesByClass = None
        self.updateGraph()

    def updateGraph(self):
        black = QColor(0,0,0)
        white = QColor(255,255,255)
        self.graph.clear()
        #self.graph.removeMarkers()
        if self.results == None: return
        eps = 0.1 + self.progressLines * 0.1
        self.projectionCount = int(self.projectionCount)
        self.attributeCount = int(self.attributeCount)

        attrCountDict = {}
        count = min(self.projectionCount, len(self.results))
        part = 0
        diff = count / 5
        import math
        s = math.sqrt(-count**2 / math.log(0.001))      # normalizing factor

        for index in range(count):
            if index > (part+1) * diff+1:
                part += 1
            attrs = self.results[index][self.ATTR_LIST]
            if self.useGeneSets and self.geneToSet:    # replace genes with the sets in which the genes appear
                newAttrs = []
                for attr in attrs:
                    newAttrs += getGeneSet(self.geneToSet, attr)
                attrs = newAttrs

            if self.useProjectionWeighting: val = math.e ** (-index*index/(s*s))    # top ranked projections have greater val than lower ranked
            else:                           val = 1         # all projections have the same influence
            for attr in attrs:
                if not attrCountDict.has_key(attr):
                    attrCountDict[attr] = [0, {}]
                attrCountDict[attr][0] += val
                attrCountDict[attr][1][part] = attrCountDict[attr][1].get(part, 0) + val

        attrs = [(attrCountDict[key][0], attrCountDict[key][1], key) for key in attrCountDict.keys()]
        attrs.sort()
        attrs.reverse()
        attrs = attrs[:self.attributeCount]
        if not attrs: return

        if self.colorAttributes and self.evaluatedAttributes == None and self.dialogType in [VIZRANK_POINT, CLUSTER_POINT]:
            evalAttrs, attrsByClass = orngVisFuncts.findAttributeGroupsForRadviz(self.parent.data, orngVisFuncts.S2NMeasureMix())

            classVariableValues = getVariableValuesSorted(self.parent.data, self.parent.data.domain.classVar.name)
            classColors = ColorPaletteHSV(len(classVariableValues))
            self.evaluatedAttributes = evalAttrs
            self.evaluatedAttributesByClass = attrsByClass
        else:
            (evalAttrs, attrsByClass) = (self.evaluatedAttributes, self.evaluatedAttributesByClass)
            classVariableValues = getVariableValuesSorted(self.parent.data, self.parent.data.domain.classVar.name)
            classColors = ColorPaletteHSV(len(classVariableValues))


        attrNames = []
        maxProjCount = attrs[0][0]      # the number of appearances of the most frequent attribute. used to determine when to stop drawing the progress lines
        for (ind, (count, progressCountDict, attr)) in enumerate(attrs):
            if self.colorAttributes and self.dialogType in [VIZRANK_POINT, CLUSTER_POINT]:
                if attr in evalAttrs:
                    classIndex = evalAttrs.index(attr) % len(classVariableValues)
                    color = classColors[classVariableValues.index(self.parent.data.domain.classVar.values[classIndex])]
                else:
                    color = black
            else:
                color = black

            curve = PolygonCurve(self.graph, QPen(color, 1), QBrush(color), [ind-0.5+eps, ind+0.5-eps, ind+0.5-eps, ind-0.5+eps, ind-0.5+eps], [0, 0, count, count, 0])
            key = self.graph.insertCurve(curve)

            if self.progressLines and count*8 > maxProjCount:
                curr = 0
                for i in range(4):
                    c = progressCountDict.get(i, 0)
                    curr += c
                    self.graph.addCurve("", black, black, 2, QwtCurve.Lines, QwtSymbol.None, xData = [ind-0.5+0.5*eps,ind+0.5-0.5*eps], yData = [curr, curr], lineWidth = 3)

            attrNames.append(attr)
            """
            y = -attrs[0][0] * 0.03
            if self.rotateXAttributes: marker = RotatedMarker(self.graph, attr, ind + 0.5, y, 90)
            else: marker = RotatedMarker(self.graph, attr, ind + 0.5, y, 0)
            mkey = self.graph.insertMarker(marker)
            if self.rotateXAttributes: self.graph.marker(mkey).setLabelAlignment(Qt.AlignLeft+ Qt.AlignVCenter)
            else: self.graph.marker(mkey).setLabelAlignment(Qt.AlignCenter + Qt.AlignBottom)
            """

        # draw attribute names
        self.graph.setAxisScaleDraw(QwtPlot.xBottom, OWGraphTools.DiscreteAxisScaleDraw(attrNames))
        self.graph.axisScaleDraw(QwtPlot.xBottom).setTickLength(0, 0, 0)  # hide ticks
        self.graph.axisScaleDraw(QwtPlot.xBottom).setOptions(0)           # hide horizontal line representing x axis
        self.graph.setAxisMaxMajor(QwtPlot.xBottom, len(attrNames))
        self.graph.setAxisMaxMinor(QwtPlot.xBottom, 0)
        self.graph.setAxisScale(QwtPlot.xBottom, -1, len(attrNames), 1)
        if self.rotateXAttributes:
            self.graph.axisScaleDraw(QwtPlot.xBottom).setLabelRotation(-90)
            self.graph.axisScaleDraw(QwtPlot.xBottom).setLabelAlignment(Qt.AlignLeft)
        else:
            self.graph.axisScaleDraw(QwtPlot.xBottom).setLabelRotation(0)
            self.graph.axisScaleDraw(QwtPlot.xBottom).setLabelAlignment(Qt.AlignHCenter)

        self.graph.setYLaxisTitle("Number of appearances in top projections")

        """
        self.graph.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
        #self.graph.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        self.graph.axisScaleDraw(QwtPlot.xBottom).setTickLength(0, 0, 0)
        #self.graph.axisScaleDraw(QwtPlot.yLeft).setTickLength(0, 0, 0)
        self.graph.axisScaleDraw(QwtPlot.xBottom).setOptions(0)
        #self.graph.axisScaleDraw(QwtPlot.yLeft).setOptions(0)
        self.graph.setAxisScale(QwtPlot.xBottom, - 0.5 , len(attrs), 1)
        #self.graph.setAxisScale(QwtPlot.yLeft, - 0.9 - 0.1*len(attrs) , attrs[0][0], 1)
        """

        if self.colorAttributes:
            classVariableValues = getVariableValuesSorted(self.parent.data, self.parent.data.domain.classVar.name)
            classColors = ColorPaletteHSV(len(classVariableValues))
            self.graph.addCurve("<b>" + self.parent.data.domain.classVar.name + ":</b>", QColor(0,0,0), QColor(0,0,0), 0, symbol = QwtSymbol.None, enableLegend = 1)
            for i,val in enumerate(classVariableValues):
                self.graph.addCurve(val, classColors[i], classColors[i], 15, symbol = QwtSymbol.Rect, enableLegend = 1)

        self.graph.update()  # don't know if this is necessary
        self.graph.repaint()

    def hideEvent(self, ev):
        self.saveSettings()
        OWWidget.hideEvent(self, ev)

# #############################################################################
# draw a graph for all the evaluated projections that shows how is the classification accuracy falling when we are moving from the best to the worst evaluated projections
class OWGraphProjectionQuality(OWWidget):
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Projection Quality", wantGraph = 1)

        self.lineWidth = 1
        self.showDistributions = 0
        self.smoothingParameter = 1

        self.results = None
        self.dialogType = -1

        b1 = OWGUI.widgetBox(self.controlArea, box = "Show...")
        self.smoothingBox = OWGUI.widgetBox(self.controlArea, 'Smoothing parameter')
        b3 = OWGUI.widgetBox(self.controlArea, 'Line Width')

        b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        self.smoothingBox.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b3.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        box = OWGUI.widgetBox(self.controlArea)

        OWGUI.comboBox(b1, self, "showDistributions", items = ["Drop in scores", "Distribution of scores"], callback = self.updateGraph)
        OWGUI.qwtHSlider(self.smoothingBox, self, "smoothingParameter", minValue = 0.0, maxValue = 5, step = 0.1, callback = self.updateGraph)
        OWGUI.comboBox(b3, self, "lineWidth", items = range(1,5), callback = self.updateGraph, sendSelectedValue = 1, valueType = int)

        self.graph = OWGraph(self.mainArea)
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)
        self.box.activate()
        self.graph.showXaxisTitle = 1
        self.graph.showYLaxisTitle = 1

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        self.updateGraph()

    def setResults(self, results, dialogType):
        self.results = results
        self.dialogType = dialogType
        if dialogType in [VIZRANK_POINT, CLUSTER_POINT]:
            self.ACCURACY = ACCURACY
        elif dialogType == VIZRANK_MOSAIC:
            import orngMosaic
            self.ACCURACY = orngMosaic.SCORE
        self.updateGraph()

    def updateGraph(self):
        #colors = ColorPaletteHSV(2)
        #c = colors.getColor(0)
        c = QColor(0,0,0)
        self.graph.clear()
        if self.results == None or self.dialogType not in [VIZRANK_POINT, CLUSTER_POINT, VIZRANK_MOSAIC]: return

        yVals = [result[self.ACCURACY] for result in self.results]
        if not yVals: return

        if self.showDistributions:
            try:
                from numarray.nd_image import gaussian_filter1d
            except:
                self.showDistributions = 0
                QMessageBox.information( None, "Missing library", 'In order to show distibution of scores gaussian smoothing has to be applied and a module called numarray is needed.\nIt can be downloaded at http://sourceforge.net/projects/numpy', QMessageBox.Ok + QMessageBox.Default)

        self.smoothingBox.setEnabled(self.showDistributions)

        if not self.showDistributions:
            xVals = range(len(yVals))
            if len(yVals) > 10:
                fact = len(yVals)/200
                if fact > 0:        # make the array of data smaller
                    pos = 0
                    xTemp = []; yTemp = []
                    while pos < len(yVals):
                        xTemp.append(xVals[pos])
                        yTemp.append(yVals[pos])
                        pos += fact
                    xVals = xTemp; yVals = yTemp
            self.graph.addCurve("", c, c, 1, QwtCurve.Lines, QwtSymbol.None, xData = xVals, yData = yVals, lineWidth = self.lineWidth)
            self.graph.setXaxisTitle("Evaluated projections")
            self.graph.setYLaxisTitle("Projection score")
        else:
            ymax = yVals[0]
            ymin = yVals[-1]
            yVals.reverse()
            diff = (ymax-ymin) / 100.
            xs = [ymin + diff/2.]
            ys = [0]
            x = ymin
            for index in range(len(yVals)):
                if yVals[index] > x + diff:     # if we stepped into another part, we start counting elements in here from 0
                    ys.append(0)
                    x = x + diff
                    xs.append(x + diff/2.)
                ys[-1] += 1
            ys = gaussian_filter1d(ys, self.smoothingParameter).tolist()
            self.graph.addCurve("", c, c, 1, QwtCurve.Lines, QwtSymbol.None, xData = xs, yData = ys, lineWidth = self.lineWidth)
            self.graph.setXaxisTitle("Projection score")
            self.graph.setYLaxisTitle("Number of projections")

        self.graph.update()  # don't know if this is necessary
        self.graph.repaint()


# #############################################################################
# draw a graph for all the evaluated projections that shows how is the classification accuracy falling when we are moving from the best to the worst evaluated projections
class OWGraphIdentifyOutliers(OWWidget):
    settingsList = ["projectionCountList", "showLegend", "showAllClasses", "sortProjections", "showClickedProjection"]
    def __init__(self,parent=None, signalManager = None, widget = None, graph = None):
        OWWidget.__init__(self, parent, signalManager, "Outlier Identification", wantGraph = 1, wantStatusBar = 1, savePosition = True)

        self.projectionCountList = ["5", "10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "Other..."]
        self.projectionCount = "20"
        self.selectedExampleIndex = 0
        self.showPredictionsInProjection = 0
        self.showLegend = 1
        self.showAllClasses = 0
        self.sortProjections = 1
        self.showClickedProjection = 1

        self.projectionIndices = []
        self.matrixOfPredictions = None
        self.projectionGraph = graph
        self.parent = parent
        self.widget = widget
        self.graphMatrix = None
        self.results = None
        self.data = None
        self.dialogType = -1
        self.evaluatedExamples = []

        self.loadSettings()

        b1 = OWGUI.widgetBox(self.controlArea, 'Projection Count')
        self.projectionCountEdit = OWGUI.comboBoxWithCaption(b1, self, "projectionCount", "Best projections to consider:   ", tooltip = "How many projections do you want to consider when computing probabilities of correct classification?", items = self.projectionCountList, callback = self.projectionCountChanged, sendSelectedValue = 1, valueType = str)

        b2 = OWGUI.widgetBox(self.controlArea, 'Example Index', orientation="horizontal")
        self.selectedExampleCombo = OWGUI.comboBox(b2, self, "selectedExampleIndex", tooltip = "Select the index of the example whose predictions you wish to analyse in the graph", callback = self.selectedExampleChanged, sendSelectedValue = 1, valueType = int)
        butt = OWGUI.button(b2, self, "Get from projection", self.updateIndexFromGraph, tooltip = "Use the index of the example that is selected in the projections")
##        butt.setMaximumWidth(60)

        b3 = OWGUI.widgetBox(self.controlArea, 'Graph Settings')
        OWGUI.checkBox(b3, self, 'showAllClasses', 'Show probabilities for all classes', tooltip = "Show predicted probabilities for each class value", callback = self.updateGraph)
        OWGUI.checkBox(b3, self, 'sortProjections', 'Sort projections by decreasing probability', tooltip = "Don't show projections as they are ranked, but by decreasing probability of correct classification (this usually improves perception)", callback = self.updateGraph)
        OWGUI.checkBox(b3, self, 'showLegend', 'Show class legend', callback = self.updateGraph)
        OWGUI.checkBox(b3, self, 'showClickedProjection', 'Show selected projection', tooltip = "Show the corresponding projection by clicking its horizontal bar in the graph", callback = self.updateGraph)

        b6 = OWGUI.widgetBox(self.controlArea, "Show Predictions For All Examples")
        self.showGraphCheck = OWGUI.checkBox(b6, self, 'showPredictionsInProjection', 'Show probabilities in the projection', tooltip = "Color the points in the projection according to the average probability of correct classification over the selected projection count", callback = self.toggleShowPredictions)
        self.exampleList = QListBox(b6)
        QToolTip.add(self.exampleList, "Average probabilities of correct classification and indices of corresponding examples")
        self.connect(self.exampleList, SIGNAL("selectionChanged()"),self.exampleListSelectionChanged)

        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWGraph(self.mainArea)
        self.graph.showXaxisTitle = 1
        self.graph.showYLaxisTitle = 1
        self.graph.setXaxisTitle("Predicted class probabilities")
        self.graph.setYLaxisTitle("Projections")
        self.graph.show()

        self.box.addWidget(self.graph)
        self.box.activate()

        self.graph.disconnect(self.graph, SIGNAL('plotMousePressed(const QMouseEvent&)'), self.graph.onMousePressed)
        self.graph.disconnect(self.graph, SIGNAL('plotMouseReleased(const QMouseEvent&)'),self.graph.onMouseReleased)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        self.connect(self.graph, SIGNAL("plotMouseMoved(const QMouseEvent &)"), self.graphOnMouseMoved)
        self.connect(self.graph, SIGNAL('plotMousePressed(const QMouseEvent&)'), self.graphOnMousePressed)

        self.resize(600, 400)
##        self.show()

    # on escape
    def hideEvent (self, e):
        if self.widget:
            self.widget.outlierValues = None
            self.widget.updateGraph()
        self.saveSettings()
        OWWidget.hideEvent(self, e)

    def setData(self, results, data, dialogType):
        self.results = results
        self.data = data
        self.dialogType = dialogType
        self.matrixOfPredictions = None

        if dialogType == VIZRANK_POINT:
            self.ATTR_LIST = ATTR_LIST
            self.ACCURACY = ACCURACY
        elif dialogType == VIZRANK_MOSAIC:
            import orngMosaic
            self.ATTR_LIST = orngMosaic.ATTR_LIST
            self.ACCURACY = orngMosaic.SCORE

        # example index combo
        self.selectedExampleCombo.clear()
        if self.data:
            for i in range(len(self.data)):
                self.selectedExampleCombo.insertItem(str(i))

        self.evaluateProjections()
        self.selectedExampleChanged()

    def projectionCountChanged(self):
        self.exampleList.clear()
        self.evaluatedExamples = []

        if self.projectionCount == "Other...":
            (text, ok) = QInputDialog.getText('Qt Projection Count', 'How many of the best projections do you wish to consider?')
            if ok and str(text).isdigit():
                text = str(text)
                if text not in self.projectionCountList:
                    i = 0
                    while i < len(self.projectionCountList)-1 and int(self.projectionCountList[i]) < int(text): i+=1
                    self.projectionCountList.insert(i, text)
                    self.projectionCountEdit.insertItem(text, i)
                self.projectionCount = text
            else:
                self.projectionCount = "20"
        self.evaluateProjections()
        self.selectedExampleChanged()


    def evaluateProjections(self):
        if not self.results or not self.data: return

        # compute predictions
        self.widget.progressBarInit()

        projCount = min(int(self.projectionCount), len(self.results))
        classCount = len(self.data.domain.classVar.values)
        existing = 0
        if self.matrixOfPredictions != None:
            existing = numpy.shape(self.matrixOfPredictions)[0]/classCount
            if existing < projCount:
                self.matrixOfPredictions = numpy.resize(self.matrixOfPredictions, (projCount*classCount, len(self.data)))
            elif existing > projCount:
                self.matrixOfPredictions = self.matrixOfPredictions[0:classCount*projCount,:]
        else:
            self.matrixOfPredictions = -1 * numpy.ones((projCount*classCount, len(self.data)), numpy.float)


        # compute the matrix of predictions
        results = self.results[existing:min(len(self.results),projCount)]
        index = 0

        self.widgetStatusArea.show()
        for result in results:
            if self.dialogType == VIZRANK_POINT:
                acc, other, tableLen, attrList, tryIndex, generalDict = result
                attrIndices = [self.projectionGraph.attributeNameIndex[attr] for attr in attrList]
                validDataIndices = self.projectionGraph.getValidIndices(attrIndices)
                table = self.projectionGraph.createProjectionAsExampleTable(attrIndices, settingsDict = generalDict)    # TO DO: this does not work with polyviz!!!
                qApp.processEvents()        # allow processing of other events
                acc, probabilities = self.parent.kNNClassifyData(table)

            elif self.dialogType == VIZRANK_MOSAIC:
                from orngCI import FeatureByCartesianProduct
                acc, attrList, tryIndex, other = result
                probabilities = numpy.zeros((len(self.data), len(self.data.domain.classVar.values)), numpy.float)
                newFeature, quality = FeatureByCartesianProduct(self.data, attrList)
                dist = orange.ContingencyAttrClass(newFeature, self.data)
                data = self.data.select([newFeature, self.data.domain.classVar])     # create a dataset that has only this new feature and class info
                clsVals = len(self.data.domain.classVar.values)
                validDataIndices = range(len(data))
                for i, ex in enumerate(data):
                    try:
                        prob = dist[ex[0]]
                        for j in range(clsVals):
                            probabilities[i][j] = prob[j] / float(sum(prob.values()))
                    except:
                        validDataIndices.remove(i)


            #self.matrixOfPredictions[(existing + index)*classCount:(existing + index +1)*classCount] = numpy.transpose(probabilities)
            probabilities = numpy.transpose(probabilities)
            for i in range(classCount):
                numpy.put(self.matrixOfPredictions[(existing + index)*classCount + i], validDataIndices, probabilities[i])

            index += 1
            self.setStatusBarText("Evaluated %s/%s projections..." % (orngVisFuncts.createStringFromNumber(existing + index), orngVisFuncts.createStringFromNumber(projCount)))
            self.widget.progressBarSet(100.0*(index)/float(projCount-existing))

        # update the list box with probabilities for all examples
        self.exampleList.clear()
        projCount = min(int(self.projectionCount), len(self.results))
        classCount = len(self.data.domain.classVar.values)

        for i in range(len(self.data)):
            matrix = numpy.transpose(numpy.reshape(self.matrixOfPredictions[:, i], (projCount, classCount)))
            valid = numpy.where(matrix[int(self.data[i].getclass())] != -1, 1, 0)
            data = numpy.compress(valid, matrix[int(self.data[i].getclass())])
            if len(data): ave_acc = numpy.sum(data) / float(len(data))
            else:         ave_acc = 0
            self.insertItemToExampleList(ave_acc, i)

        self.widget.progressBarFinished()
        self.widgetStatusArea.hide()


    def toggleShowPredictions(self):
        if not self.widget: return
        if self.showPredictionsInProjection:
            self.evaluateProjections()

            self.widgetStatusArea.show()
            self.setStatusBarText("Computing averages...")

            projCount = min(int(self.projectionCount), len(self.results))
            classCount = len(self.data.domain.classVar.values)

            # compute the average probability of correct classification over the selected number of top projections
            values = [0.0 for i in range(len(self.data))]
            for i in range(len(self.data)):
                corrClass = int(self.data[i].getclass())
                predictions = self.matrixOfPredictions[corrClass::classCount,i]
                predictions = numpy.compress(predictions != -1, predictions)
                predictions = predictions**3
                if len(predictions):    # prevent division by zero!
                    values[i] = numpy.sum(predictions) / float(len(predictions))

            self.widget.outlierValues = (values, "Probability of correct class value = %.2f%%")
            self.setStatusBarText("")
            #self.widgetStatusArea.hide()
        else:
            self.widget.outlierValues = None

        self.widget.updateGraph()
        self.widget.showSelectedAttributes()


    def selectedExampleChanged(self):
        if not self.results or not self.data: return

        projCount = min(int(self.projectionCount), len(self.results))
        classCount = len(self.data.domain.classVar.values)
        self.graphMatrix = numpy.transpose(numpy.reshape(self.matrixOfPredictions[:, self.selectedExampleIndex], (projCount, classCount)))
        self.updateGraph()

        if self.dialogType == VIZRANK_POINT:
            valid = self.projectionGraph.getValidList([self.projectionGraph.attributeNameIndex[attr] for attr in self.widget.getShownAttributeList()])
            insideColors = numpy.zeros(len(self.data))
            insideColors[self.selectedExampleIndex] = 1
            self.widget.updateGraph(insideColors = (numpy.compress(valid, insideColors), "Focused example: %d"))


    # find which examples is selected in the graph and draw its predictions
    def updateIndexFromGraph(self):
        if self.dialogType != VIZRANK_POINT:
            return

        if self.parent.parentName == "Polyviz":
            selected, unselected = self.projectionGraph.getSelectionsAsIndices(self.widget.getShownAttributeList(), self.widget.attributeReverse)
        else:
            selected, unselected = self.projectionGraph.getSelectionsAsIndices(self.widget.getShownAttributeList())

        if len(selected) != 1:
            QMessageBox.information( None, "Outlier Identification", 'Exactly one example must be selected in the graph in order to complete this operation.', QMessageBox.Ok + QMessageBox.Default)
            return
        self.selectedExampleIndex = selected[0]
        self.selectedExampleChanged()


    # insert new result - give parameters: accuracy of projection, number of examples in projection and list of attributes.
    def insertItemToExampleList(self, val, exampleIndex):
        top = 0; bottom = len(self.evaluatedExamples)
        index = 0

        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if min(val, self.evaluatedExamples[mid][0]) == val: bottom = mid
            else: top = mid

        if len(self.evaluatedExamples) == 0: index = 0
        elif min(val, self.evaluatedExamples[top][0]) == val:
            index = top
        else:
            index = bottom

        self.evaluatedExamples.insert(index, (val, exampleIndex))
        self.exampleList.insertItem("%.2f - %d" % (val, exampleIndex), index)


    def exampleListSelectionChanged(self):
        (val, exampleIndex) = self.evaluatedExamples[self.exampleList.currentItem()]
        self.selectedExampleIndex = exampleIndex
        self.selectedExampleChanged()

    # draw the graph of predictions for the selected example
    def updateGraph(self):
        self.graph.clear()
        self.graph.tips.removeAll()
        self.selectedRectangle = None
        if not self.data or self.graphMatrix == None: return

        classColors = ColorPaletteHSV(len(self.data.domain.classVar.values))

        if self.graphMatrix == None: return

        self.graph.setAxisScale(QwtPlot.yLeft, 0, len(self.graphMatrix[0]), len(self.graphMatrix[0])/5)
        self.graph.setAxisScale(QwtPlot.xBottom, 0, 1, 0.2)

        valid = numpy.where(self.graphMatrix[0] != -1, 1, 0)
        allValid = numpy.sum(valid) == len(valid)
        nrOfClasses = len(self.data.domain.classVar.values)

        if self.sortProjections:
            cls = int(self.data[self.selectedExampleIndex].getclass())
            indices = [(self.graphMatrix[cls][i], i) for i in range(len(self.graphMatrix[0]))]
            indices.sort()
            classes = range(nrOfClasses); classes.remove(cls); classes = [cls] + classes
        else:
            indices = [(i,i) for i in range(len(self.graphMatrix[0]))]
            classes = range(nrOfClasses)

        self.projectionIndices = [val[1] for val in indices]
        classVariableValues = getVariableValuesSorted(self.parent.data, self.parent.data.domain.classVar.name)
        classColors = ColorPaletteHSV(len(classVariableValues))

        for i in range(len(self.graphMatrix[0])):
            x = 0
            s = "Predicted class probabilities:<br>"
            for j in classes:
                (prob, index) = indices[i]
                s += "&nbsp; &nbsp; &nbsp; %s: %.2f%%<br>" % (classVariableValues[j], 100*self.graphMatrix[j][index])
                if not self.showAllClasses and int(self.data[self.selectedExampleIndex].getclass()) != j:
                    continue
                xDiff = self.graphMatrix[j][index]
                self.graph.insertCurve(PolygonCurve(self.graph, QPen(classColors.getColor(j)), QBrush(classColors.getColor(j)), [x, x+xDiff, x+xDiff, x, x], [i, i, i+1, i+1, i]))
                x += xDiff
            self.graph.tips.addToolTip(0, i, s[:-4], 1, 1)

        if self.showLegend:
            self.graph.addCurve("<b>" + self.parent.data.domain.classVar.name + ":</b>", QColor(0,0,0), QColor(0,0,0), 0, symbol = QwtSymbol.None, enableLegend = 1)
            for i,val in enumerate(classVariableValues):
                self.graph.addCurve(val, classColors[i], classColors[i], 15, symbol = QwtSymbol.Rect, enableLegend = 1)
        else:
            self.graph.enableLegend(0)

        self.graph.update()  # don't know if this is necessary
        self.graph.repaint()

    def graphOnMouseMoved(self, e):
        import math
        if getattr(self, "selectedRectangle", None) != None:
            self.graph.removeCurve(self.selectedRectangle)
            self.selectedRectangle = None

        y = int(math.floor(self.graph.invTransform(QwtPlot.yLeft, e.y())))
        if self.showClickedProjection and y >= 0 and y < len(self.projectionIndices):
            diff  = 0.005
            self.selectedRectangle = self.graph.insertCurve(PolygonCurve(self.graph, brush = QBrush(Qt.NoBrush), xData = [0-diff, 1+diff, 1+diff, 0-diff, 0-diff], yData = [y-diff, y-diff, y+1+diff, y+1+diff, y-diff]))
            self.graph.replot()


    def graphOnMousePressed(self, e):
        if self.showClickedProjection:
            import math
            y = int(math.floor(self.graph.invTransform(QwtPlot.yLeft, e.y())))
            if y >= len(self.projectionIndices): return
            projIndex = self.projectionIndices[y]
            self.parent.resultList.setSelected(projIndex, 1)

            if self.dialogType == VIZRANK_POINT:
                attrs = self.parent.shownResults[projIndex][self.ATTR_LIST]
                valid = self.projectionGraph.getValidList([self.projectionGraph.attributeNameIndex[attr] for attr in attrs])
                insideColors = numpy.zeros(len(self.data))
                insideColors[self.selectedExampleIndex] = 1
                self.widget.updateGraph(attrs, setAnchors = 1, insideColors = (numpy.compress(valid, insideColors), "Focused example: %d"))


def getGeneSet(geneset, gene):
    if len(gene) > 2 and gene[-2] == "_":
        gene = gene[:-2]        # remove the "_X" suffix
    return geneset.get(gene, [])


# load a .gmt file with gene groups
def loadGeneSetFile(fname):
    f = open(fname, "rt")
    geneToSet = {}
    setToGenes = {}

    for i, line in enumerate(f.xreadlines()):
        items = line[:-1].split("\t")
        setName = items[0]
        genes = []
        for item in items[2:]:
            sub = item.split("///")
            itm = []
            for s in sub:
                gene = s.strip()
                geneToSet[gene] = geneToSet.get(gene, []) + [setName]
                genes.append(gene)
        setToGenes[setName] = (genes, len(items[2:]))
    return geneToSet, setToGenes


#test widget appearance
if __name__=="__main__":
    import sys
    a=QApplication(sys.argv)
    ow=OWVizRank()
##    ow = OWInteractionAnalysis()
##    ow = OWGraphAttributeHistogram()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
