from OWWidget import *
import OWGUI, OWDlgs, OWGraphTools, numpy, user, sys
from OWGraph import *
from orngVizRank import *
from orngScaleData import getVariableValuesSorted

class OWVizRank(VizRank, OWWidget):
    settingsList = ["kValue", "resultListLen", "percentDataUsed", "qualityMeasure", "qualityMeasureCluster", "qualityMeasureContClass", "testingMethod",
                    "lastSaveDirName", "attrCont", "attrDisc", "showRank", "showAccuracy", "showInstances",
                    "evaluationAlgorithm", "evaluationTime", "learnerName", "attrContContClass", "attrDiscContClass", "attrContNoClass", "attrDiscNoClass",
                    "argumentCount", "optimizeBestProjection", "optimizeBestProjectionTime",
                    "locOptMaxAttrsInProj", "locOptAttrsToTry", "locOptProjCount", "locOptAllowAddingAttributes",
                    "useExampleWeighting", "projOptimizationMethod", "attrSubsetSelection", "optimizationType", "attributeCount",
                    "locOptOptimizeProjectionByPermutingAttributes", "timeLimit", "projectionLimit", "storeEachPermutation",
                    "boxLocalOptimization", "boxStopOptimization", "clearPreviousProjections"]
    resultsListLenNums = [ 10, 100 ,  500 ,  1000 ,  5000 ,  10000, 20000, 50000, 100000, 500000 ]
    percentDataNums = [ 5 ,  10 ,  15 ,  20 ,  30 ,  40 ,  50 ,  60 ,  70 ,  80 ,  90 ,  100 ]
    kNeighboursNums = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 30,35, 40, 50, 60, 70, 80, 100, 120, 150, 200]
    argumentCounts = [1, 3, 5, 10, 15, 20, 30, 50, 100, 200]
    evaluationTimeNums = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 60, 80, 120]
    moreArgumentsNums = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

    def __init__(self, parentWidget = None, signalManager = None, graph = None, visualizationMethod = SCATTERPLOT, parentName = "Visualization widget"):
        OWWidget.__init__(self, None, signalManager, "VizRank Dialog", savePosition = True, wantMainArea = 0, wantStatusBar = 1)
        VizRank.__init__(self, visualizationMethod, graph)

        self.parentWidget = parentWidget
        self.parentName = parentName
        self.visualizationMethod = visualizationMethod

        self.resultListLen = 1000
        self.cancelOptimization = 0
        self.cancelEvaluation = 0
        self.learnerName = "VizRank Learner"

        self.useTimeLimit = 0
        self.useProjectionLimit = 0
        self.evaluationTime = 10
        self.optimizeBestProjection = 0                     # do we want to try to locally improve the best projections
        self.optimizeBestProjectionTime = 10                 # how many minutes do we want to try to locally optimize the best projections
        self.clearPreviousProjections = 1

        self.maxResultListLen = self.resultsListLenNums[len(self.resultsListLenNums)-1]
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

        self.layout().setMargin(0)
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.MainTab = OWGUI.createTabPage(self.tabs, "Main")
        self.SettingsTab = OWGUI.createTabPage(self.tabs, "Settings")
        self.ArgumentationTab = OWGUI.createTabPage(self.tabs, "Argumentation")
        self.ManageTab = OWGUI.createTabPage(self.tabs, "Manage && Save")

        # ###########################
        # MAIN TAB
        self.optimizationBox = OWGUI.widgetBox(self.MainTab, "Evaluate")    
        self.buttonBox = OWGUI.widgetBox(self.optimizationBox, orientation = "horizontal")

        if visualizationMethod != SCATTERPLOT:
            self.label1 = OWGUI.widgetLabel(self.buttonBox, 'Projections with ' )
            self.optimizationTypeCombo = OWGUI.comboBox(self.buttonBox, self, "optimizationType", items = ["    exactly    ", "  maximum  "] )
            self.attributeCountCombo = OWGUI.comboBox(self.buttonBox, self, "attributeCount", items = range(3, 20), tooltip = "Evaluate only projections with exactly (or maximum) this number of attributes", sendSelectedValue = 1, valueType = int, debuggingEnabled = 0)
            self.attributeLabel = OWGUI.widgetLabel(self.buttonBox, ' attributes')

        self.startOptimizationButton = OWGUI.button(self.optimizationBox, self, "Start Evaluating Projections", callback = self.evaluateProjections)
        f = self.startOptimizationButton.font(); f.setBold(1);   self.startOptimizationButton.setFont(f)
        self.optimizeGivenProjectionButton = OWGUI.button(self.optimizationBox, self, "Locally Optimize Best Projections", callback = self.optimizeBestProjections)

        box = OWGUI.widgetBox(self.MainTab, "Projection List, Most Interesting Projections First")
        self.resultList = OWGUI.listBox(box, self, callback = self.parentWidget and self.parentWidget.showSelectedAttributes or None)
        #self.resultList.setMinimumSize(200,200)

        self.resultsDetailsBox = OWGUI.widgetBox(self.MainTab, "Shown Details in Projections List" , orientation = "horizontal")
        self.showRankCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showRank', 'Rank', callback = self.updateShownProjections, tooltip = "Show projection ranks")
        self.showAccuracyCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showAccuracy', 'Score', callback = self.updateShownProjections, tooltip = "Show prediction accuracy of a k-NN classifier on the projection")
        self.showInstancesCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showInstances', '# Instances', callback = self.updateShownProjections, tooltip = "Show number of instances in the projection")

        # ##########################
        # SETTINGS TAB
        self.optimizationSettingsDiscClassBox = OWGUI.widgetBox(self.SettingsTab, "VizRank Evaluation Settings")
        self.methodTypeCombo = OWGUI.comboBoxWithCaption(self.optimizationSettingsDiscClassBox, self, "evaluationAlgorithm", "Projection evaluation method: ", tooltip = "Which learning method to use to use to evaluate given projections.", items = ["k-Nearest Neighbor", "Heuristic (very fast)"])
        self.attrKNeighboursEdit = OWGUI.lineEdit(self.optimizationSettingsDiscClassBox, self, "kValue", "Number of neighbors (k):  ", orientation = "horizontal", tooltip = "Number of neighbors used in k-NN algorithm to evaluate the projection", valueType = int, validator = QIntValidator(self))
        self.percentDataUsedCombo= OWGUI.comboBoxWithCaption(self.optimizationSettingsDiscClassBox, self, "percentDataUsed", "Percent of data used: ", items = self.percentDataNums, sendSelectedValue = 1, valueType = int, tooltip = "In case that we have a large dataset the evaluation of each projection can take a lot of time.\nWe can therefore use only a subset of randomly selected examples, evaluate projection on them and thus make evaluation faster.")
        self.testingCombo = OWGUI.comboBox(self.optimizationSettingsDiscClassBox, self, "testingMethod", label = "Testing method:                             ", orientation = "horizontal", items = ["Leave one out (slowest)", "10 fold cross validation", "Test on learning set (fastest)"], tooltip = "Method for evaluating the classifier. Slower are more accurate while faster give only a rough approximation.")
        OWGUI.checkBox(self.optimizationSettingsDiscClassBox, self, 'useExampleWeighting', 'Use example weighting', tooltip = "For datasets where we have uneven class distribution we can weight examples")
        if visualizationMethod != SCATTERPLOT:
            OWGUI.checkBox(self.optimizationSettingsDiscClassBox, self, 'storeEachPermutation', 'Save all projections for each permutation of attributes', tooltip = "Do you want to see in the projection list all evaluated projections or only the best projection for each attribute permutation.\nUsually this value is unchecked.")

        if visualizationMethod == LINEAR_PROJECTION:
            OWGUI.comboBox(self.SettingsTab, self, "projOptimizationMethod", "Projection Optimization Method", items = ["None", "Supervised projection pursuit", "Partial least square"], sendSelectedValue = 0, tooltip = "What method do you want to use to find an interesting projection with good class separation?")
        else:
            self.projOptimizationMethod = 0
            
        self.optimizationSettingsNoClassBox = OWGUI.widgetBox(self.SettingsTab, "VizRank Evaluation Settings")
        
        self.measureComboDiscClassBox = OWGUI.widgetBox(self.SettingsTab, "Measure of Classification Success")
        OWGUI.comboBox(self.measureComboDiscClassBox, self, "qualityMeasure", items = ["Classification accuracy", "Average Probability Assigned to the Correct Class", "Brier Score", "Area under Curve (AUC)"], tooltip = "Measure to evaluate prediction accuracy of k-NN method on the projected data set.")
        
#        self.measureComboContClassBox = OWGUI.widgetBox(self.SettingsTab, "Measure of Regression Accuracy")
#        OWGUI.comboBox(self.measureComboDiscClassBox, self, "qualityMeasureContClass", items = ["Classification accuracy", "Average Probability Assigned to the Correct Class", "Brier Score", "Area under Curve (AUC)"], tooltip = "Measure to evaluate prediction accuracy of k-NN method on the projected data set.")
#        
#        self.measureComboNoClassBox = OWGUI.widgetBox(self.SettingsTab, "Measure of Cluster Interestingness")
#        OWGUI.comboBox(self.measureComboNoClassBox, self, "qualityMeasureCluster", items = ["Example distance"], tooltip = "Measure to evaluate how well are points in the projection separated into clusters.")

        self.attributeSelectionBox = OWGUI.widgetBox(self.SettingsTab, "Attribute Subset Selection")
        OWGUI.comboBox(self.attributeSelectionBox, self, "attrSubsetSelection", items = ["Deterministically Using the Selected Attribute Ranking Measures", "Use Gamma Distribution and Test All Possible Placements", "Use Gamma Distribution and Test Only One Possible Placement"])

        self.heuristicsSettingsDiscClassBox = OWGUI.widgetBox(self.SettingsTab, "Measures for Attribute Ranking")
        OWGUI.comboBoxWithCaption(self.heuristicsSettingsDiscClassBox, self, "attrCont", "For continuous attributes:", items = [val for (val, m) in contMeasuresDiscClass], callback = self.removeEvaluatedAttributes)
        OWGUI.comboBoxWithCaption(self.heuristicsSettingsDiscClassBox, self, "attrDisc", "For discrete attributes:", items = [val for (val, m) in discMeasuresDiscClass], callback = self.removeEvaluatedAttributes)
        
#        self.heuristicsSettingsNoClassBox = OWGUI.widgetBox(self.SettingsTab, "Measures for Attribute Ranking")
#        OWGUI.comboBoxWithCaption(self.heuristicsSettingsNoClassBox, self, "attrContNoClass", "For continuous attributes:", items = [val for (val, m) in contMeasuresNoClass], callback = self.removeEvaluatedAttributes)
#        OWGUI.comboBoxWithCaption(self.heuristicsSettingsNoClassBox, self, "attrDiscNoClass", "For discrete attributes:", items = [val for (val, m) in discMeasuresNoClass], callback = self.removeEvaluatedAttributes)
#        
#        self.heuristicsSettingsContClassBox = OWGUI.widgetBox(self.SettingsTab, "Measures for Attribute Ranking")
#        OWGUI.comboBoxWithCaption(self.heuristicsSettingsContClassBox, self, "attrContContClass", "For continuous attributes:", items = [val for (val, m) in contMeasuresContClass], callback = self.removeEvaluatedAttributes)
#        OWGUI.comboBoxWithCaption(self.heuristicsSettingsContClassBox, self, "attrDiscContClass", "For discrete attributes:", items = [val for (val, m) in discMeasuresContClass], callback = self.removeEvaluatedAttributes)
        
        self.miscSettingsBox = OWGUI.widgetBox(self.SettingsTab, "Projection List")
        self.resultListCombo = OWGUI.comboBoxWithCaption(self.miscSettingsBox, self, "resultListLen", "Maximum length of projection list:   ", tooltip = 'Maximum number of top-ranked projections that are shown in the list box. This is also the number of projections that will be saved if you click "Save" button.', items = self.resultsListLenNums, callback = self.updateShownProjections, sendSelectedValue = 1, valueType = int)
        OWGUI.checkBox(self.miscSettingsBox, self, 'clearPreviousProjections', 'Remove previously evaluated projections', tooltip = 'Do you want to continue projection evaluation from where it was stopped or do \nyou want to evaluate them from the start (by first clearing the list of evaluated projections)?')

        smallWidget = OWGUI.SmallWidgetButton(OWGUI.widgetBox(self.SettingsTab, box = 1), text = "Show advanced settings")
        self.stopOptimizationBox = OWGUI.widgetBox(smallWidget.widget, "When to automatically stop evaluation?", self)
        OWGUI.checkWithSpin(self.stopOptimizationBox, self, "Time limit:                     ", 1, 1000, "useTimeLimit", "timeLimit", "  (minutes)", debuggingEnabled = 0)      # disable debugging. we always set this to 1 minute
        OWGUI.checkWithSpin(self.stopOptimizationBox, self, "Use projection count limit:  ", 1, 1000000, "useProjectionLimit", "projectionLimit", "  (projections)", debuggingEnabled = 0)

        self.localOptimizationSettingsBox = OWGUI.widgetBox(smallWidget.widget, "Local optimization settings", self)
        bbb = OWGUI.checkBox(self.localOptimizationSettingsBox, self, 'locOptOptimizeProjectionByPermutingAttributes', 'Try improving projection by permuting attributes in projection')
        self.localOptimizationProjCountCombo = OWGUI.comboBoxWithCaption(self.localOptimizationSettingsBox , self, "locOptProjCount", "Number of best projections to optimize:           ", items = range(1,30), tooltip = "Specify the number of best projections in the list that you want to try to locally optimize.\nIf you select 1 only the currently selected projection will be optimized.", sendSelectedValue = 1, valueType = int)
        self.localOptimizationAttrsCount = OWGUI.lineEdit(self.localOptimizationSettingsBox, self, "locOptAttrsToTry", "Number of best attributes to try:                       ", orientation = "horizontal", tooltip = "How many of the top ranked attributes do you want to try in the projections?", valueType = int, validator = QIntValidator(self))
        locOptBox = OWGUI.widgetBox(self.localOptimizationSettingsBox, orientation = "horizontal")
        self.localOptimizationAddAttrsCheck  = OWGUI.checkBox(locOptBox, self, 'locOptAllowAddingAttributes', 'Allow adding attributes. Max attrs in proj:', tooltip = "Should local optimization only try to replace some attributes in a projection or is it also allowed to add new attributes?")
        self.localOptimizationProjMaxAttr    = OWGUI.comboBox(locOptBox, self, "locOptMaxAttrsInProj", items = range(3,50), tooltip = "What is the maximum number of attributes in a projection?", sendSelectedValue = 1, valueType = int)

        self.SettingsTab.layout().addStretch(100)

        # ##########################
        # ARGUMENTATION TAB
        self.argumentationBox = OWGUI.widgetBox(self.ArgumentationTab, "Arguments")
        self.findArgumentsButton = OWGUI.button(self.argumentationBox, self, "Find Arguments", callback = self.findArguments, debuggingEnabled = 0)
        f = self.findArgumentsButton.font(); f.setBold(1);  self.findArgumentsButton.setFont(f)
        self.argumentCountEdit = OWGUI.lineEdit(self.argumentationBox , self, "argumentCount", "Number of best projections to consider:     ", orientation = "horizontal", tooltip = "How many of the top ranked projections do you wish to consider?", valueType = int, validator = QIntValidator(self))

        self.classValueCombo = OWGUI.comboBox(self.ArgumentationTab, self, "argumentationClassValue", box = "Arguments for Class:", tooltip = "Select the class value that you wish to see arguments for", callback = self.argumentationClassChanged, debuggingEnabled = 0)
        self.argumentBox = OWGUI.widgetBox(self.ArgumentationTab, "Arguments for the Selected Class Value")
        self.argumentList = OWGUI.listBox(self.argumentBox, self, callback = self.argumentSelected)
        self.argumentList.setMinimumSize(200,200)
        
        
        #Remove and hide the argumentation tab (It does not work)
        self.tabs.removeTab(2)
        self.ArgumentationTab.hide()
        


        # ##########################
        # SAVE & MANAGE TAB
        self.classesBox = OWGUI.widgetBox(self.ManageTab, "Class Values You Wish to See Separated")
        self.classesBox.setFixedHeight(130)
        self.visualizedAttributesBox = OWGUI.widgetBox(self.ManageTab, "Number of Concurrently Visualized Attributes")
        self.dialogsBox = OWGUI.widgetBox(self.ManageTab, "Dialogs")
        self.manageResultsBox = OWGUI.widgetBox(self.ManageTab, "Manage Projections")

        self.classesList = OWGUI.listBox(self.classesBox, self, selectionMode = QListWidget.MultiSelection, callback = self.classesListChanged)
        self.classesList.setMinimumSize(60,60)

        self.attrLenList = OWGUI.listBox(self.visualizedAttributesBox, self, selectionMode = QListWidget.MultiSelection, callback = self.attrLenListChanged)
        self.attrLenList.setMinimumSize(60,60)

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
        self.saveBestButton = OWGUI.button(self.buttonBox9, self, "Save Best", self.exportMultipleGraphs, debuggingEnabled = 0)
        OWGUI.button(self.buttonBox9, self, "Remove Similar", callback = self.removeTooSimilarProjections, debuggingEnabled = 0)

        self.buttonBox3 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        if self.parentWidget:
            self.evaluateProjectionButton = OWGUI.button(self.buttonBox3, self, 'Evaluate Projection', callback = self.evaluateCurrentProjection, debuggingEnabled = 0)
        self.reevaluateResults = OWGUI.button(self.buttonBox3, self, "Reevaluate", callback = self.reevaluateAllProjections)

        self.buttonBox4 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.showKNNCorrectButton = OWGUI.button(self.buttonBox4, self, 'Show k-NN Correct', self.showKNNCorect)
        self.showKNNWrongButton = OWGUI.button(self.buttonBox4, self, 'Show k-NN Wrong', self.showKNNWrong)
        self.showKNNCorrectButton.setCheckable(1); self.showKNNWrongButton.setCheckable(1)

        self.buttonBox5 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.clearButton = OWGUI.button(self.buttonBox5, self, "Clear Results", self.clearResults)

        self.removeEvaluatedAttributes()

        self.setMinimumWidth(375)
        self.tabs.setMinimumWidth(375)
        self.resize(375, 700)
        
        # reset some parameters if we are debugging so that it won't take too much time
        if orngDebugging.orngDebuggingEnabled:
            self.useTimeLimit = 1
            self.useProjectionLimit = 1
            self.timeLimit = 0.3
            self.optimizeTimeLimit = 0.3
            self.projectionLimit = 20
            self.optimizeProjectionLimit = 20
            self.attributeCount = 6
            
        self.subsetData = None


    # ##############################################################
    # EVENTS

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
            intVal = int(str(self.attrLenList.item(i).text()))
            selected = self.attrLenList.item(i).isSelected()
            self.attrLenDict[intVal] = selected
        self.updateShownProjections()

    def classesListChanged(self):
        results = self.results
        self.clearResults()

        self.selectedClasses = self.getSelectedClassValues()
        if len(self.selectedClasses) in [self.classesList.count(), 0]:
            for i in range(len(results)):
                self.insertItem(i, results[i][OTHER_RESULTS][0], results[i][OTHER_RESULTS], results[i][LEN_TABLE], results[i][ATTR_LIST], results[i][TRY_INDEX], results[i][GENERAL_DICT])
        else:
            for result in results:
                acc = 0.0; sum = 0.0
                for index in self.selectedClasses:
                    acc += result[OTHER_RESULTS][OTHER_PREDICTIONS][index] * result[OTHER_RESULTS][OTHER_DISTRIBUTION][index]
                    sum += result[OTHER_RESULTS][OTHER_DISTRIBUTION][index]
                self.insertItem(self.findTargetIndex(acc/max(sum,1.)), acc/max(sum,1.), result[OTHER_RESULTS], result[LEN_TABLE], result[ATTR_LIST], result[TRY_INDEX], result[GENERAL_DICT])

        self.finishedAddingResults()

    def clearResults(self):
        VizRank.clearResults(self)
        self.clearArguments()
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
            if self.resultList.item(i).isSelected():
                # remove from listbox and original list of results
                self.resultList.takeItem(i)
                self.shownResults.remove(self.shownResults[i])
        
    # ##############################################################

    def getSelectedClassValues(self):
        selectedClasses = []
        for i in range(self.classesList.count()):
            if self.classesList.item(i).isSelected(): selectedClasses.append(i)
        return selectedClasses


    # a function that is meaningful when visualizing using radviz or polyviz
    # it removes projections that don't have different at least two attributes in comparison with some better ranked projection
    def removeTooSimilarProjections(self, allowedPercentOfEqualAttributes = -1):
        if allowedPercentOfEqualAttributes == -1:
            (text, ok) = QInputDialog.getText('Allowed Similarity', 'How many attributes can be present in some better projection for a projection to be still considered as different (in pecents. Default = 70)?')
            if not ok: return
            allowedPercentOfEqualAttributes = int(str(text))

        qApp.setOverrideCursor(Qt.WaitCursor)
        self.setStatusBarText("Removing similar projections")
        i=0
        while i < self.resultList.count():
            qApp.processEvents()
            if self.existsABetterSimilarProjection(i, allowedPercentOfEqualAttributes = allowedPercentOfEqualAttributes):
                self.results.pop(i)
                self.shownResults.pop(i)
                self.resultList.takeItem(i)
            else:
                i += 1

        self.setStatusBarText("")
        qApp.restoreOverrideCursor()


    def updateShownProjections(self, *args):
        if hasattr(self, "dontUpdate"): return

        self.resultList.clear()
        self.shownResults = []
        i = 0
        qApp.setOverrideCursor(Qt.WaitCursor)

        while self.resultList.count() < self.resultListLen and i < len(self.results):
            if self.attrLenDict[len(self.results[i][ATTR_LIST])] == 1:
                string = ""
                if self.showRank: string += str(i+1) + ". "
                if self.showAccuracy: string += "%.2f" % (self.results[i][ACCURACY])
                if not self.showInstances and self.showAccuracy: string += " : "
                elif self.showInstances: string += " (%d) : " % (self.results[i][LEN_TABLE])
                string += self.buildAttrString(self.results[i][ATTR_LIST], self.results[i][GENERAL_DICT].get("reverse", []))

                self.resultList.addItem(string)
                self.shownResults.append(self.results[i])
            i+=1
        qApp.processEvents()
        qApp.restoreOverrideCursor()

        if self.resultList.count() > 0: self.resultList.setCurrentRow(0)

    # set value of k to sqrt(n)
    def resetDialog(self):
        self.setStatusBarText("")

        self.removeEvaluatedAttributes()

        #self.startOptimizationButton.setEnabled(self.graph.dataHasDiscreteClass)
        #self.optimizeGivenProjectionButton.setEnabled(self.graph.dataHasDiscreteClass)
        #self.evaluateProjectionButton.setEnabled(self.graph.dataHasDiscreteClass)
        self.showKNNCorrectButton.setEnabled(self.graph.dataHasDiscreteClass)
        self.showKNNWrongButton.setEnabled(self.graph.dataHasDiscreteClass)
        self.attributeRankingButton.setEnabled(self.graph.dataHasDiscreteClass)
        self.attributeInteractionsButton.setEnabled(self.graph.dataHasDiscreteClass)
        self.projectionScoresButton.setEnabled(self.graph.dataHasDiscreteClass)
        self.outlierIdentificationButton.setEnabled(self.graph.dataHasDiscreteClass)
        #self.findArgumentsButton.setEnabled(self.graph.dataHasDiscreteClass)
        
        self.optimizationSettingsDiscClassBox.setVisible(self.graph.dataHasDiscreteClass)
        self.optimizationSettingsNoClassBox.setVisible(not self.graph.dataHasClass)
        self.measureComboDiscClassBox.setVisible(self.graph.dataHasDiscreteClass)
#        self.measureComboNoClassBox.setVisible(not self.graph.dataHasClass)
#        self.measureComboContClassBox.setVisible(self.graph.dataHasContinuousClass)
        self.tabs.setTabEnabled(2, self.graph.dataHasDiscreteClass)
#        self.heuristicsSettingsContClassBox.setVisible(self.graph.dataHasContinuousClass)
        self.heuristicsSettingsDiscClassBox.setVisible(self.graph.dataHasDiscreteClass)
#        self.heuristicsSettingsNoClassBox.setVisible(not self.graph.dataHasClass)
        
        
        if not self.graph.dataHasDiscreteClass:
            self.classesList.clear()
            self.classValueCombo.clear()
            self.selectedClasses = []
            return
        
        classes = [val for val in self.graph.dataDomain.classVar.values]
        existing = [str(self.classesList.item(i).text()) for i in range(self.classesList.count())]
        if classes == existing:
            return

        # set new class values
        self.classesList.clear()
        self.classValueCombo.clear()
        self.selectedClasses = []
        self.classesList.addItems(classes)
        self.classValueCombo.addItems(classes)
        self.classesList.selectAll()
        self.selectedClasses = range(len(self.graph.dataDomain.classVar.values))


    # given a dataset return a list of attributes where attribute are sorted by their decreasing importance for class discrimination
    def getEvaluatedAttributes(self):
        self.setStatusBarText("Evaluating attributes...")
        qApp.setOverrideCursor(Qt.WaitCursor)
        attrs = VizRank.getEvaluatedAttributes(self)
        self.setStatusBarText("")
        qApp.restoreOverrideCursor()
        return attrs


    # insert new result - give parameters: accuracy of projection, number of examples in projection and list of attributes.
    def insertItem(self, index, accuracy, other_results, lenTable, attrList, tryIndex, generalDict = {}, updateStatusBar = 0):
        VizRank.insertItem(self, index, accuracy, other_results, lenTable, attrList, tryIndex, generalDict, updateStatusBar = 0)

        if index < self.resultListLen:
            string = ""
            if self.showRank: string += str(index+1) + ". "
            if self.showAccuracy: string += "%.2f" % (accuracy)
            if not self.showInstances and self.showAccuracy: string += " : "
            elif self.showInstances: string += " (%d) : " % (lenTable)

            string += self.buildAttrString(attrList, generalDict.get("reverse", []))

            self.resultList.insertItem(index, string)
            self.shownResults.insert(index, (accuracy, lenTable, other_results, attrList, tryIndex, generalDict))

            # remove worst projection if list is too long
            if self.resultList.count() > self.resultListLen:
                self.resultList.takeItem(self.resultList.count()-1)
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
            self.attrLenList.addItem(str(val))
            self.attrLenDict[val] = 1
        
        self.attrLenList.selectAll()
        self.resultList.setCurrentRow(0)

        # make sure that if the dialogs are shown we show the updated results
        if self.attributeHistogramDlg and self.attributeHistogramDlg.isVisible():
            self.attributeHistogramDlg.setResults(self.shownResults)
        if self.interactionAnalysisDlg and self.interactionAnalysisDlg.isVisible():
            self.interactionAnalysisDlg.setResults(self.shownResults)
        if self.identifyOutliersDlg and self.identifyOutliersDlg.isVisible():
            self.identifyOutliersDlg.setResults(self.results)


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

            table = self.graph.createProjectionAsExampleTable([self.graph.attributeNameIndex[attr] for attr in attrList], settingsDict = generalDict)
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

        if self.graph.dataHasContinuousClass:
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
                QMessageBox.information( None, self.parentName, 'Accuracy of the model is %.3f' % (acc), QMessageBox.Ok + QMessageBox.Default)



    # ##############################################################
    # Loading and saving projection files
    def abortOperation(self):
        self.abortCurrentOperation = 1

    # save the list into a file - filename can be set if you want to call this function without showing the dialog
    def saveProjections(self):
        self.setStatusBarText("Saving projections")

        # get file name
        datasetName = getattr(self.graph.rawData, "name", "")
        if datasetName != "":
            filename = "%s - %s" % (os.path.splitext(os.path.split(datasetName)[1])[0], self.parentName)
        else:
            filename = "%s" % (self.parentName)
        qname = QFileDialog.getSaveFileName(self, "Save Projections",  os.path.join(self.lastSaveDirName, filename), "Interesting projections (*.proj)")
        if qname.isEmpty(): return
        name = str(qname)

        self.lastSaveDirName = os.path.split(name)[0]

        # show button to stop saving
        butt = OWGUI.button(self.widgetStatusArea, self, "Stop Saving", callback = self.abortOperation); butt.show()

        self.save(name, self.shownResults, len(self.shownResults))

        self.setStatusBarText("Saved %s projections" % (orngVisFuncts.createStringFromNumber(len(self.shownResults))))
        butt.hide()


    # load projections from a file
    def loadProjections(self, name = None, ignoreCheckSum = 0, maxCount = -1):
        self.setStatusBarText("Loading projections")
        if not self.graph.haveData:
            QMessageBox.critical(None,'Load','There is no data. First load a data set and then load projection file',QMessageBox.Ok)
            return

        if name == None:
            name = QFileDialog.getOpenFileName(self, "Open Projections", self.lastSaveDirName, "Interesting projections (*.proj)")
            if name.isEmpty(): return
            name = str(name)

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        # show button to stop loading
        butt = OWGUI.button(self.widgetStatusArea, self, "Stop Loading", callback = self.abortOperation); butt.show()

        selectedClasses, count = self.load(name, ignoreCheckSum, maxCount)

        self.dontUpdate = 1
        if self.graph.dataHasDiscreteClass:
            for i in range(len(self.graph.dataDomain.classVar.values)): self.classesList.item(i).setSelected(i in selectedClasses)
        del self.dontUpdate
        self.finishedAddingResults()

        self.setStatusBarText("Loaded %s projections" % (orngVisFuncts.createStringFromNumber(count)))
        butt.hide()

    def showKNNCorect(self):
        self.showKNNWrongButton.setChecked(0)
        if self.parentWidget: self.parentWidget.updateGraph()

    # show quality of knn model by coloring accurate predictions with lighter color and bad predictions with dark color
    def showKNNWrong(self):
        self.showKNNCorrectButton.setChecked(0)
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
        (text, ok) = QInputDialog.getText('Graph count', 'How many of the best projections do you wish to save?')
        if not ok: return
        self.bestGraphsCount = int(str(text))

        self.sizeDlg = OWDlgs.OWChooseImageSizeDlg(self.graph, parent=self)
        self.sizeDlg.printButton.setEnabled(0)
        self.sizeDlg.saveMatplotlibButton.setEnabled(0)
        self.sizeDlg.disconnect(self.sizeDlg.saveImageButton, SIGNAL("clicked()"), self.sizeDlg.saveImage)
        self.sizeDlg.connect(self.sizeDlg.saveImageButton, SIGNAL("clicked()"), self.saveToFileAccept)
        self.sizeDlg.exec_()

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
            self.resultList.item(i).setSelected(1)
            self.graph.replot()
            name = fil + " (%02d, %.2f, %d)" % (i+1, self.shownResults[i][ACCURACY], self.shownResults[i][LEN_TABLE]) + extension
            self.sizeDlg.saveImage(name, closeDialog = 0)
        QDialog.accept(self.sizeDlg)

    # ##############################################################
    # create different dialogs
    def interactionAnalysis(self):
        if not self.interactionAnalysisDlg:
            self.interactionAnalysisDlg = OWInteractionAnalysis(self, VIZRANK_POINT, signalManager = self.signalManager)
        self.interactionAnalysisDlg.setResults(self.graph.rawData, self.shownResults)
        self.interactionAnalysisDlg.show()

    def attributeAnalysis(self):
        if not self.attributeHistogramDlg:
            self.attributeHistogramDlg = OWGraphAttributeHistogram(self, VIZRANK_POINT, signalManager = self.signalManager)
        self.attributeHistogramDlg.show()
        self.attributeHistogramDlg.setResults(self.graph.rawData, self.shownResults)

    def graphProjectionQuality(self):
        dialog = OWGraphProjectionQuality(self, VIZRANK_POINT, signalManager = self.signalManager)
        dialog.show()
        dialog.setResults(self.results)

    def identifyOutliers(self):
        if not self.identifyOutliersDlg:
            self.identifyOutliersDlg = OWGraphIdentifyOutliers(self, VIZRANK_POINT, signalManager = self.signalManager, widget = self.parentWidget)
        self.identifyOutliersDlg.show()
        self.identifyOutliersDlg.setResults(self.graph.rawData, self.shownResults)

    def closeEvent(self, ce):
        if self.interactionAnalysisDlg: self.interactionAnalysisDlg.close()
        if self.attributeHistogramDlg: self.attributeHistogramDlg.close()
        if self.identifyOutliersDlg: self.identifyOutliersDlg.close()
        OWWidget.closeEvent(self, ce)

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
        if self.resultList.selectedItems() == []: return None
        return self.shownResults[self.resultList.row(self.resultList.selectedItems()[0])]

    def evaluateProjections(self):
        if str(self.startOptimizationButton.text()) == "Start Evaluating Projections":
            if self.attributeCount >= 10 and self.projOptimizationMethod == 0 and self.visualizationMethod != SCATTERPLOT and self.attrSubsetSelection != GAMMA_SINGLE and QMessageBox.critical(self, 'VizRank', 'You chose to evaluate projections with a high number of attributes. Since VizRank has to evaluate different placements\nof these attributes there will be a high number of projections to evaluate. Do you still want to proceed?','Continue','Cancel', '', 0,1):
                return
            if not self.graph.dataHasDiscreteClass:
                if not orngDebugging.orngDebuggingEnabled:
                    QMessageBox.information( None, self.parentName, "Projections can be evaluated only for data with a discrete class.", QMessageBox.Ok + QMessageBox.Default)
                return
            self.startOptimizationButton.setText("Stop Evaluation")
            self.parentWidget.progressBarInit()
            self.disableControls()

            try:
                evaluatedProjections = VizRank.evaluateProjections(self, self.clearPreviousProjections)
            except:
                evaluatedProjections = 0
                type, val, traceback = sys.exc_info()
                sys.excepthook(type, val, traceback)  # print the exception

            self.enableControls()
            self.parentWidget.progressBarFinished()

            secs = time.time() - self.startTime
            self.setStatusBarText("Finished evaluation (evaluated %s projections in %d min, %d sec)" % (orngVisFuncts.createStringFromNumber(evaluatedProjections), secs/60, secs%60))
            self.finishedAddingResults()
            #qApp.processEvents()
            #if self.parentWidget:
            #    self.parentWidget.showSelectedAttributes()
            self.startOptimizationButton.setText("Start Evaluating Projections")
        else:
            self.cancelEvaluation = 1
            self.cancelOptimization = 1


    def optimizeBestProjections(self, restartWhenImproved = 1):
        self.startOptimizationButton.setText("Stop Optimization")
        self.disableControls()
        try:
            evaluatedProjections = VizRank.optimizeBestProjections(self, restartWhenImproved)
        except:
            evaluatedProjections = 0
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # print the exception

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

    # ######################################################
    # Argumentation functions
    def findArguments(self, example = None, selectBest = 1, showClassification = 1):
        self.clearArguments()
        self.arguments = [[] for i in range(len(self.graph.dataDomain.classVar.values))]

        if not example and self.subsetData == None:
            QMessageBox.information( None, "VizRank Argumentation", 'To find arguments you first have to provide a new example that you wish to classify. You can do this by sending the example through the "Example Subset" input signal. \n\nNext, you should press the "Start Evaluating Projections" button in the Main tab to evaluate some projections. \n\nBy pressing "Find Arguments" you will then find arguments why the given example should belong to a selected class.', QMessageBox.Ok + QMessageBox.Default)
            return (None,None)
        if len(self.shownResults) == 0:
            QMessageBox.information( None, "VizRank Argumentation", 'To find arguments you first have to evaluate some projections by clicking "Start evaluating projections" in the Main tab.', QMessageBox.Ok + QMessageBox.Default)
            return (None,None)
        if not example:
            example = self.subsetData[0]

        # call VizRank's function for finding arguments
        classValue, dist = VizRank.findArguments(self, example)

        if not self.arguments: return
        classIndex = self.classValueCombo.currentIndex() #currentItem()
        for i in range(len(self.arguments[0])):
            prob, d, attrList, index = self.arguments[classIndex][i]
            self.argumentList.insertItem(i, "%.3f - %s" %(prob, attrList))

        if self.argumentList.count() > 0 and selectBest:
            values = getVariableValuesSorted(self.graph.dataDomain[self.graph.dataClassIndex])
            self.argumentationClassValue = values.index(classValue)     # activate the class that has the highest probability
            self.argumentationClassChanged()
            self.argumentList.setCurrentRow(0)
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
        ind = self.classValueCombo.currentIndex() #currentItem()
        for i in range(len(self.arguments[ind])):
            val = self.arguments[ind][i]
            self.argumentList.addItem("%.2f - %s" %(val[0], val[2]))

    def argumentSelected(self):
        if self.argumentList.selectedItems() == []: return
        ind = self.argumentList.row(self.argumentList.selectedItems()[0])
        classInd = self.classValueCombo.currentIndex() #currentItem()
        generalDict = self.results[self.arguments[classInd][ind][3]][GENERAL_DICT]
        if self.visualizationMethod == SCATTERPLOT:
            attrs = self.arguments[classInd][ind][2]
            self.graph.updateData(attrs[0], attrs[1], self.graph.dataDomain.classVar.name)
        else:
            self.graph.updateData(self.arguments[classInd][ind][2], setAnchors = 1, XAnchors = generalDict.get("XAnchors"), YAnchors = generalDict.get("YAnchors"))
        self.graph.repaint()


# #############################################################################
# analyse the attributes that appear in the top projections. show how often do they appear also in other top projections
class OWInteractionAnalysis(OWWidget):
    settingsList = ["onlyLower", "rectColoring", "sortAttributesByQuality", "useGeneSets", "recentGeneSets"]
    def __init__(self,parent = None, dialogType = VIZRANK_POINT, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "VizRank's Interaction Analysis", wantGraph = 1, savePosition = True)

        self.parent = parent
        self.dialogType = dialogType
        self.attributeCount = 15
        self.projectionCount = 100
        self.rotateXAttributes = 1
        self.onlyLower = 0
        self.results = None
        self.sortAttributesByQuality = 0
        self.rectColoring = 1

        self.recentGeneSets = []
        self.geneToSet, self.setToGenes = None, None
        self.useGeneSets = 0

        self.graph = OWGraph(self.mainArea)
        self.mainArea.layout().addWidget(self.graph)

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        b1 = OWGUI.widgetBox(self.controlArea, 'Number of attributes')
        b2 = OWGUI.widgetBox(self.controlArea, 'Number of projections')
        b3 = OWGUI.widgetBox(self.controlArea, "Settings")
        b4 = OWGUI.widgetBox(self.controlArea, "Use color to represent ...")
        b5 = OWGUI.widgetBox(self.controlArea, "Gene Sets")

        OWGUI.qwtHSlider(b1, self, 'attributeCount', minValue = 5, maxValue = 100, step=0, callback = self.updateGraph, precision = 0, maxWidth = 170)
        self.projectionCountSlider = OWGUI.qwtHSlider(b2, self, 'projectionCount', minValue = 5, maxValue = 1000, step = 5, callback = self.updateGraph, precision = 0, maxWidth = 170)
        OWGUI.checkBox(b3, self, 'rotateXAttributes', label = "Rotate X labels", callback = self.updateGraph)
        OWGUI.checkBox(b3, self, 'onlyLower', label = "Show only lower diagonal", callback = self.updateGraph)
        OWGUI.checkBox(b3, self, 'sortAttributesByQuality', 'Sort attributes according to quality', callback = self.updateGraph, tooltip = "Do you want to show the attributes as they are ranked according to some quality measure\nor as they appear in the top ranked projections?")

        OWGUI.comboBox(b4, self, "rectColoring", tooltip = "What should darkness of color of rectangles represent?", items = ["(don't use color)", "projection quality", "frequency of occurence in projections", "both"], callback = self.updateGraph)

        OWGUI.checkBox(b5, self, "useGeneSets", label = "Use gene sets", callback = self.updateGraph)
        bb5 = OWGUI.widgetBox(b5, orientation  = "horizontal")
        self.fileCombo = OWGUI.comboBox(bb5, self, "geneSets")
        OWGUI.button(bb5, self, '...', callback = self.browseGeneFile, disabled=0, width=25)
        self.connect(self.fileCombo, SIGNAL('activated(int)'), self.selectGeneFile)

        self.controlArea.layout().addSpacing(100)

        #qApp.processEvents()
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
            if sys.platform == "darwin":
                startfile = user.home
            else:
                startfile = "."
        else:
            startfile = self.recentGeneSets[0]
        filename = str(QFileDialog.getOpenFileName(None, 'Open Gene Set File', startfile, 'Gene set files (*.gmt)\nAll files(*.*)'))
        if filename == "": return
        if filename in self.recentGeneSets: self.recentGeneSets.remove(filename)
        self.recentGeneSets.insert(0, filename)
        self.updateGeneCombo()
        self.loadGeneSet()

    def updateGeneCombo(self):
        self.fileCombo.clear()
        for file in self.recentGeneSets:
            self.fileCombo.addItem(os.path.split(file)[1])

    def loadGeneSet(self):
        if len(self.recentGeneSets) == 0: return
        self.geneToSet, self.setToGenes = loadGeneSetFile(self.recentGeneSets[0])
        self.updateGraph()
    # ----------------------------------- #

    def setResults(self, data, results):
        self.results = results
        if results:
            self.projectionCountSlider.setScale(0, (len(results)/50) * 50, 0) # the third parameter for logaritmic scale
        if self.dialogType in [VIZRANK_POINT, CLUSTER_POINT]:
            if self.parent.attrCont == CONT_MEAS_S2NMIX:
                self.attributes, attrsByClass = orngVisFuncts.findAttributeGroupsForRadviz(data, orngVisFuncts.S2NMeasureMix())
            else:
                self.attributes = self.parent.getEvaluatedAttributes()
            self.ATTR_LIST = ATTR_LIST
            self.ACCURACY = ACCURACY
        elif self.dialogType == VIZRANK_MOSAIC:
            relieff = orange.MeasureAttribute_relief(k=10, m=50)
            self.attributes = orngVisFuncts.evaluateAttributes(data, relieff, relieff)
            import orngMosaic
            self.ATTR_LIST = orngMosaic.ATTR_LIST
            self.ACCURACY = orngMosaic.SCORE
        self.updateGraph()

    def updateGraph(self):
        black = QColor(0,0,0)
        white = QColor(255,255,255)
        self.graph.clear()
        #self.graph.removeMarkers()
        self.graph.tips.removeAll()

        if not self.results or self.dialogType not in [VIZRANK_POINT, CLUSTER_POINT, VIZRANK_MOSAIC]: return

        self.projectionCount = int(self.projectionCount)
        self.attributeCount = int(self.attributeCount)

        attributes = []
        attrDict = {}
        countDict = {}
        bestDict = {}

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
                    countDict[(Min, Max)] = countDict.get((Min, Max), 0) + 1

                    # projection quality
                    if not bestDict.has_key((Min, Max)):
                        bestDict[(Min, Max)] = self.results[index][self.ACCURACY]
            index += 1

        bestCount = max([1] + countDict.values())
        worstCount = -1  # we could use 0 but those with 1 would be barely visible
        bestAcc = self.results[0][self.ACCURACY]
        worstAcc= self.results[min(len(self.results)-1, self.projectionCount)][self.ACCURACY]

        eps = 0.05 + 0.15 * self.useGeneSets
        eps2 = 0.05
        num = len(attributes)

        for x in range(num):
            for y in range(num-x):
                yy = num-y-1
                countVal = None; bestVal = None

                if countDict.has_key((attributes[x], attributes[yy])):
                    countVal = countDict[(attributes[x], attributes[yy])]
                elif countDict.has_key((attributes[yy], attributes[x])):
                    countVal = countDict[(attributes[yy], attributes[x])]

                if bestDict.has_key((attributes[x], attributes[yy])):
                    accVal = bestDict[(attributes[x], attributes[yy])]
                elif bestDict.has_key((attributes[yy], attributes[x])):
                    accVal = bestDict[(attributes[yy], attributes[x])]

                if countVal == bestVal == None:
                    continue

                if self.rectColoring == 0:
                    color = black
                elif self.rectColoring == 1:
                    v = int(255 - 255*((accVal-worstAcc)/float(bestAcc - worstAcc)))
                    color = QColor(v,v,v)
                elif self.rectColoring == 2:
                    v = int(255 - 255*((countVal-worstCount)/float(bestCount - worstCount)))
                    color = QColor(v,v,v)
                elif self.rectColoring == 3:
                    v1 = int(255 - 255*((accVal-worstAcc)/float(bestAcc - worstAcc)))
                    v2 = int(255 - 255*((countVal-worstCount)/float(bestCount - worstCount)))
                    color1 = QColor(v1,v1,v1)
                    color2 = QColor(v2,v2,v2)

                s = "Pair: <b>%s</b> and <b>%s</b>" % (attributes[x], attributes[yy])
                if self.rectColoring in [1,3]:    # projection quality
                    s += "<br>Best projection quality: <b>%.3f</b>" % (accVal)
                if self.rectColoring in [2,3]:    # count
                    s += "<br>Number of appearances: <b>%d</b>" % (countVal)

                sharedGeneSets = []
                if self.useGeneSets and self.geneToSet:
                    set1 = getGeneSet(self.geneToSet, attributes[x])
                    set2 = getGeneSet(self.geneToSet, attributes[yy])
                    for s1 in set2:
                        if s1 in set1: sharedGeneSets.append(s1)

                if self.useGeneSets and self.geneToSet:
                    if sharedGeneSets != []:
                        s += "<hr>"+"Shared gene sets: %s"+"<br>" % (sharedGeneSets)
                    s += "<hr>"+"Gene sets for individual genes:"+"<br>&nbsp; &nbsp; <b>%s</b>: %s<br>&nbsp; &nbsp; <b>%s</b>: %s" % (attributes[x], getGeneSet(self.geneToSet, attributes[x]), attributes[yy], getGeneSet(self.geneToSet, attributes[yy]))

                if self.rectColoring != 3:
                    RectangleCurve(QPen(color, 1), QBrush(color), [x-0.5+eps, x+0.5-eps, x+0.5-eps, x-0.5+eps], [y+1-0.5+eps, y+1-0.5+eps, y+1+0.5-eps, y+1+0.5-eps]).attach(self.graph)
                else:
                    PolygonCurve(QPen(color1, 1), QBrush(color1), [x-0.5+eps, x+0.5-eps, x-0.5+eps, x-0.5+eps], [y+1-0.5+eps, y+1-0.5+eps, y+1+0.5-eps, y+1-0.5+eps]).attach(self.graph)
                    PolygonCurve(QPen(color2, 1), QBrush(color2), [x-0.5+eps, x+0.5-eps, x+0.5-eps, x-0.5+eps], [y+1+0.5-eps, y+1-0.5+eps, y+1+0.5-eps, y+1+0.5-eps]).attach(self.graph)

                if self.useGeneSets and self.geneToSet and sharedGeneSets != []:
                    RectangleCurve(QPen(color, 1), QBrush(Qt.NoBrush), [x-0.5+eps2, x+0.5-eps2, x+0.5-eps2, x-0.5+eps2], [y+1-0.5+eps2, y+1-0.5+eps2, y+1+0.5-eps2, y+1+0.5-eps2]).attach(self.graph)
                if s != "":
                    self.graph.tips.addToolTip(x, y+1, s, 0.5, 0.5)

                if not self.onlyLower:
                    if self.rectColoring != 3:
                        RectangleCurve(QPen(color, 1), QBrush(color), [num-1-0.5-y+eps, num-1-0.5-y+eps, num-1+0.5-y-eps, num-1+0.5-y-eps], [num-0.5-x+eps, num+0.5-x-eps, num+0.5-x-eps, num-0.5-x+eps]).attach(self.graph)
                    else:
                        PolygonCurve(QPen(color1, 1), QBrush(color1), [num-1-0.5-y+eps, num-1+0.5-y-eps, num-1-0.5-y+eps, num-1-0.5-y+eps], [num-0.5-x+eps, num-0.5-x+eps, num+0.5-x-eps, num-0.5-x+eps]).attach(self.graph)
                        PolygonCurve(QPen(color2, 1), QBrush(color2), [num-1-0.5-y+eps, num-1+0.5-y-eps, num-1+0.5-y-eps, num-1-0.5-y+eps], [num+0.5-x-eps, num-0.5-x+eps, num+0.5-x-eps, num+0.5-x-eps]).attach(self.graph)

                    if self.useGeneSets and self.geneToSet and sharedGeneSets != []:
                        RectangleCurve(QPen(color, 1), QBrush(Qt.NoBrush), [num-1-0.5-y+eps2, num-1-0.5-y+eps2, num-1+0.5-y-eps2, num-1+0.5-y-eps2], [num-0.5-x+eps2, num+0.5-x-eps2, num+0.5-x-eps2, num-0.5-x+eps2]).attach(self.graph)
                    if s != "":
                        self.graph.tips.addToolTip(num-1-y, num-x, s, 0.5, 0.5)

        # draw empty boxes at the diagonal
        for x in range(num):
            RectangleCurve(QPen(black), QBrush(Qt.NoBrush), [x-0.5+2*eps2, x+0.5-2*eps2, x+0.5-2*eps2, x-0.5+2*eps2], [num-x-0.5+2*eps2, num-x-0.5+2*eps2, num-x+0.5-2*eps2, num-x+0.5-2*eps2]).attach(self.graph)

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
        self.graph.axisScaleDraw(QwtPlot.xBottom).enableComponent(QwtScaleDraw.Ticks, 0)
        self.graph.axisScaleDraw(QwtPlot.xBottom).enableComponent(QwtScaleDraw.Backbone, 0)
        self.graph.setAxisMaxMajor(QwtPlot.xBottom, len(attributes))
        self.graph.setAxisMaxMinor(QwtPlot.xBottom, 0)
        self.graph.setAxisScale(QwtPlot.xBottom, -1, len(attributes), 1)
        if self.rotateXAttributes:
            self.graph.axisScaleDraw(QwtPlot.xBottom).setLabelRotation(-90)
            self.graph.axisScaleDraw(QwtPlot.xBottom).setLabelAlignment(Qt.AlignLeft)

        self.graph.setAxisScaleDraw(QwtPlot.yLeft, OWGraphTools.DiscreteAxisScaleDraw([""] + attributes[::-1]))
        self.graph.axisScaleDraw(QwtPlot.yLeft).enableComponent(QwtScaleDraw.Ticks, 0)
        self.graph.axisScaleDraw(QwtPlot.yLeft).enableComponent(QwtScaleDraw.Backbone, 0)
        self.graph.setAxisMaxMajor(QwtPlot.yLeft, len(attributes))
        self.graph.setAxisMaxMinor(QwtPlot.yLeft, 0)
        self.graph.setAxisScale(QwtPlot.yLeft, 0, len(attributes)+1, 1)

        self.graph.update()  # don't know if this is necessary
        self.graph.replot()

    def hideEvent(self, ev):
        self.saveSettings()
        OWWidget.hideEvent(self, ev)


class OWGraphAttributeHistogram(OWWidget):
    settingsList = ["attributeCount", "projectionCount", "rotateXAttributes", "colorAttributes", "progressLines", "useGeneSets", "recentGeneSets"]
    def __init__(self, parent=None, dialogType=VIZRANK_POINT, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Attribute Histogram", wantGraph = 1, savePosition = True)

        self.results = None
        self.dialogType = dialogType
        self.parent = parent
        self.results = None
        self.data = None
        self.evaluatedAttributes = None
        self.evaluatedAttributesByClass = None

        self.graph = OWGraph(self.mainArea)
        self.mainArea.layout().addWidget(self.graph)

        self.graph.showYLaxisTitle = 1

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        self.attributeCount = 10
        self.projectionCount = 100
        self.rotateXAttributes = 1
        self.colorAttributes = 1
        self.progressLines = 1
        self.geneToSet = None
        self.useGeneSets = 0
        self.recentGeneSets = []
        self.useProjectionWeighting = 1

        b1 = OWGUI.widgetBox(self.controlArea, box = 1)
        b2 = OWGUI.widgetBox(self.controlArea, 'Number of attributes')
        b3 = OWGUI.widgetBox(self.controlArea, 'Number of projections')
        b4 = OWGUI.widgetBox(self.controlArea, "Gene sets")
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

        self.controlArea.layout().addSpacing(100)
        
        if self.dialogType in [VIZRANK_POINT, CLUSTER_POINT]:
            self.ATTR_LIST = ATTR_LIST
        elif dialogType == VIZRANK_MOSAIC:
            import orngMosaic
            self.ATTR_LIST = orngMosaic.ATTR_LIST

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
        if len(self.recentGeneSets) == 0 or self.recentGeneSets[0] == "(none)":
            startfile = "."
        else:
            startfile = self.recentGeneSets[0]
        filename = str(QFileDialog.getOpenFileName(None,'Open Gene Set File', startfile, 'Gene set files (*.gmt)\nAll files(*.*)'))
        if filename == "": return
        if filename in self.recentGeneSets: self.recentGeneSets.remove(filename)
        self.recentGeneSets.insert(0, filename)
        self.updateGeneCombo()
        self.loadGeneSet()

    def updateGeneCombo(self):
        self.fileCombo.clear()
        for file in self.recentGeneSets:
            self.fileCombo.addItem(os.path.split(file)[1])

    def loadGeneSet(self):
        if len(self.recentGeneSets) == 0: return
        self.geneToSet, self.setToGenes = loadGeneSetFile(self.recentGeneSets[0])
        self.updateGraph()
    # ----------------------------------- #

    def setResults(self, data, results):
        self.data = data
        self.results = results
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
        
        classVariableValues = getVariableValuesSorted(self.data.domain.classVar)
        classColors = ColorPaletteHSV(len(classVariableValues))
        if self.colorAttributes and self.evaluatedAttributes == None and self.dialogType in [VIZRANK_POINT, CLUSTER_POINT]:
            evalAttrs, attrsByClass = orngVisFuncts.findAttributeGroupsForRadviz(self.data, orngVisFuncts.S2NMeasureMix())
            classColors = ColorPaletteHSV(len(classVariableValues))
            self.evaluatedAttributes = evalAttrs
            self.evaluatedAttributesByClass = attrsByClass
        else:
            (evalAttrs, attrsByClass) = (self.evaluatedAttributes, self.evaluatedAttributesByClass)
            
        attrNames = []
        maxProjCount = attrs[0][0]      # the number of appearances of the most frequent attribute. used to determine when to stop drawing the progress lines
        for (ind, (count, progressCountDict, attr)) in enumerate(attrs):
            if self.colorAttributes and self.dialogType in [VIZRANK_POINT, CLUSTER_POINT]:
                if attr in evalAttrs:
                    classIndex = evalAttrs.index(attr) % len(classVariableValues)
                    color = classColors[classVariableValues.index(self.data.domain.classVar.values[classIndex])]
                else:
                    color = black
            else:
                color = black

            RectangleCurve(QPen(color, 1), QBrush(color), [ind-0.5+eps, ind+0.5-eps, ind+0.5-eps, ind-0.5+eps], [0, 0, count, count]).attach(self.graph)

            if self.progressLines and count*8 > maxProjCount:
                curr = 0
                for i in range(4):
                    c = progressCountDict.get(i, 0)
                    curr += c
                    self.graph.addCurve("", black, black, 2, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = [ind-0.5+0.5*eps,ind+0.5-0.5*eps], yData = [curr, curr], lineWidth = 3)

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
        self.graph.axisScaleDraw(QwtPlot.xBottom).enableComponent(QwtScaleDraw.Ticks, 0)
        self.graph.axisScaleDraw(QwtPlot.xBottom).enableComponent(QwtScaleDraw.Backbone, 0)
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

        if self.colorAttributes:
            classVariableValues = getVariableValuesSorted(self.data.domain.classVar)
            classColors = ColorPaletteHSV(len(classVariableValues))
            self.graph.addCurve("<b>" + self.data.domain.classVar.name + ":</b>", QColor(0,0,0), QColor(0,0,0), 0, symbol = QwtSymbol.NoSymbol, enableLegend = 1)
            for i,val in enumerate(classVariableValues):
                self.graph.addCurve(val, classColors[i], classColors[i], 15, symbol = QwtSymbol.Rect, enableLegend = 1)

        self.graph.updateLayout()
        self.graph.replot()  # don't know if this is necessary

    def hideEvent(self, ev):
        self.saveSettings()
        OWWidget.hideEvent(self, ev)

# #############################################################################
# draw a graph for all the evaluated projections that shows how is the classification accuracy falling when we are moving from the best to the worst evaluated projections
class OWGraphProjectionQuality(OWWidget):
    def __init__(self,parent=None, dialogType = VIZRANK_POINT, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Projection Quality", wantGraph = 1)

        self.lineWidth = 1
        self.showDistributions = 0
        self.smoothingParameter = 1

        self.results = None
        self.dialogType = dialogType

        b1 = OWGUI.widgetBox(self.controlArea, box = "Show...")
        self.smoothingBox = OWGUI.widgetBox(self.controlArea, 'Smoothing parameter')
        b3 = OWGUI.widgetBox(self.controlArea, 'Line width')

        OWGUI.comboBox(b1, self, "showDistributions", items = ["Drop in scores", "Distribution of scores"], callback = self.updateGraph)
        OWGUI.qwtHSlider(self.smoothingBox, self, "smoothingParameter", minValue = 0.0, maxValue = 5, step = 0.1, callback = self.updateGraph)
        OWGUI.comboBox(b3, self, "lineWidth", items = range(1,5), callback = self.updateGraph, sendSelectedValue = 1, valueType = int)
        self.controlArea.layout().addStretch(100)

        self.graph = OWGraph(self.mainArea)
        self.mainArea.layout().addWidget(self.graph)
        self.graph.showXaxisTitle = 1
        self.graph.showYLaxisTitle = 1
        
        if dialogType in [VIZRANK_POINT, CLUSTER_POINT]:
            self.ACCURACY = ACCURACY
        elif dialogType == VIZRANK_MOSAIC:
            import orngMosaic
            self.ACCURACY = orngMosaic.SCORE

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        self.updateGraph()

    def setResults(self, results):
        self.results = results
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
                from numpy.numarray.nd_image import gaussian_filter1d
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
            self.graph.addCurve("", c, c, 1, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = xVals, yData = yVals, lineWidth = self.lineWidth)
            self.graph.setAxisScale(QwtPlot.yLeft, min(yVals), max(yVals))
            self.graph.setAxisScale(QwtPlot.xBottom, min(xVals), max(xVals))
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
            self.graph.addCurve("", c, c, 1, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = xs, yData = ys, lineWidth = self.lineWidth)
            self.graph.setAxisScale(QwtPlot.yLeft, min(ys), max(ys))
            self.graph.setAxisScale(QwtPlot.xBottom, min(xs), max(xs))
            self.graph.setXaxisTitle("Projection score")
            self.graph.setYLaxisTitle("Number of projections")

        self.graph.updateLayout()
        self.graph.replot()


# #############################################################################
# draw a graph for all the evaluated projections that shows how is the classification accuracy falling when we are moving from the best to the worst evaluated projections
class OWGraphIdentifyOutliers(VizRankOutliers, OWWidget):
    settingsList = ["projectionCountList", "showLegend", "showAllClasses", "sortProjections", "showClickedProjection"]
    def __init__(self, vizrank, dialogType, signalManager = None, widget = None):
        OWWidget.__init__(self, vizrank, signalManager, "Outlier Identification", wantGraph = 1, wantStatusBar = 1, savePosition = True)
        VizRankOutliers.__init__(self, vizrank, dialogType)

        self.projectionCountList = ["5", "10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "Other..."]
        self.projectionCountStr = "20"
        self.selectedExampleIndex = 0
        self.showPredictionsInProjection = 0
        self.showLegend = 1
        self.showAllClasses = 0
        self.sortProjections = 1
        self.showClickedProjection = 1

        self.widget = widget

        self.loadSettings()
        self.projectionCountStr = str(self.projectionCount)

        b1 = OWGUI.widgetBox(self.controlArea, 'Projection Count')
        self.projectionCountEdit = OWGUI.comboBoxWithCaption(b1, self, "projectionCountStr", "Best projections to consider:   ", tooltip = "How many projections do you want to consider when computing probabilities of correct classification?", items = self.projectionCountList, callback = self.projectionCountChanged, sendSelectedValue = 1, valueType = str)

        b2 = OWGUI.widgetBox(self.controlArea, 'Example index', orientation="horizontal")
        self.selectedExampleCombo = OWGUI.comboBox(b2, self, "selectedExampleIndex", tooltip = "Select the index of the example whose predictions you wish to analyse in the graph", callback = self.selectedExampleChanged, sendSelectedValue = 1, valueType = int)
        butt = OWGUI.button(b2, self, "Get From Projection", self.updateIndexFromGraph, tooltip = "Use the index of the example that is selected in the projections")
##        butt.setMaximumWidth(60)

        b3 = OWGUI.widgetBox(self.controlArea, 'Graph settings')
        OWGUI.checkBox(b3, self, 'showAllClasses', 'Show probabilities for all classes', tooltip = "Show predicted probabilities for each class value", callback = self.updateGraph)
        OWGUI.checkBox(b3, self, 'sortProjections', 'Sort projections by decreasing probability', tooltip = "Don't show projections as they are ranked, but by decreasing probability of correct classification (this usually improves perception)", callback = self.updateGraph)
        OWGUI.checkBox(b3, self, 'showLegend', 'Show class legend', callback = self.updateGraph)
        OWGUI.checkBox(b3, self, 'showClickedProjection', 'Show selected projection', tooltip = "Show the corresponding projection by clicking its horizontal bar in the graph", callback = self.updateGraph)

        b6 = OWGUI.widgetBox(self.controlArea, "Show predictions for all examples")
        self.showGraphCheck = OWGUI.checkBox(b6, self, 'showPredictionsInProjection', 'Show probabilities in the projection', tooltip = "Color the points in the projection according to the average probability of correct classification over the selected projection count", callback = self.toggleShowPredictions)
        self.exampleList = OWGUI.listBox(b6, self, callback = self.exampleListSelectionChanged)
        self.exampleList.setToolTip("Average probabilities of correct classification and indices of corresponding examples")

        self.graph = OWGraph(self.mainArea)
        self.mainArea.layout().addWidget(self.graph)
        self.graph.showXaxisTitle = 1
        self.graph.showYLaxisTitle = 1
        self.graph.setXaxisTitle("Predicted class probabilities")
        self.graph.setYLaxisTitle("Projections")

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        self.graph.mouseMoveEventHandler = self.graphOnMouseMoved
        self.graph.mousePressEventHandler = self.graphOnMousePressed
        self.selectedRectangle = RectangleCurve(brush = QBrush(Qt.NoBrush))
        self.selectedRectangle.attach(self.graph)
        self.resize(600, 400)

    # on escape
    def hideEvent (self, e):
        if self.widget:
            self.widget.outlierValues = None
            self.widget.updateGraph()
        self.saveSettings()
        OWWidget.hideEvent(self, e)

    def setResults(self, data, results):
        VizRankOutliers.setResults(self, data, results)

        # example index combo
        self.selectedExampleCombo.clear()
        if data:
            for i in range(len(data)):
                self.selectedExampleCombo.addItem(str(i))

        self.evaluateProjections()
        self.selectedExampleChanged()

    def projectionCountChanged(self):
        self.exampleList.clear()
        self.evaluatedExamples = []

        if self.projectionCount == "Other...":
            (text, ok) = QInputDialog.getText('Projection Count', 'How many of the best projections do you wish to consider?')
            if ok and str(text).isdigit():
                text = str(text)
                if text not in self.projectionCountList:
                    i = 0
                    while i < len(self.projectionCountList)-1 and int(self.projectionCountList[i]) < int(text): i+=1
                    self.projectionCountList.insert(i, text)
                    self.projectionCountEdit.addItem(text, i)
                self.projectionCountStr = text
            else:
                self.projectionCountStr = "20"
            self.projectionCount = int(self.projectionCountStr)
        self.evaluateProjections()
        self.selectedExampleChanged()

    # change class label to most probable and update widget with new data
    def changeClassToMostProbable(self):
        data = VizRankOutliers.changeClassToMostProbable(self)
        self.widget.setData(data)
        self.widget.handleNewSignals()
        return data

    def evaluateProjections(self):
        if not self.results or not self.data: return

        self.widget.progressBarInit()
        self.widgetStatusArea.show()
        self.exampleList.clear()

        VizRankOutliers.evaluateProjections(self, qApp)

        for i, (prob, exIndex, classPredictions) in enumerate(self.evaluatedExamples):
            self.exampleList.addItem("%.2f - %d" % (prob, exIndex))

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
                predictions = numpy.compress(predictions != -100, predictions)
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
            valid = self.vizrank.graph.getValidList([self.vizrank.graph.attributeNameIndex[attr] for attr in self.widget.getShownAttributeList()])
            insideColors = numpy.zeros(len(self.data))
            insideColors[self.selectedExampleIndex] = 1
            self.widget.updateGraph(insideColors = (numpy.compress(valid, insideColors), "Focused example: %d"))


    # find which examples is selected in the graph and draw its predictions
    def updateIndexFromGraph(self):
        if self.dialogType != VIZRANK_POINT:
            return

        if self.vizrank.parentName == "Polyviz":
            selected, unselected = self.vizrank.graph.getSelectionsAsIndices(self.widget.getShownAttributeList(), self.widget.attributeReverse)
        else:
            selected, unselected = self.vizrank.graph.getSelectionsAsIndices(self.widget.getShownAttributeList())

        if len(selected) != 1:
            QMessageBox.information( None, "Outlier Identification", 'Exactly one example must be selected in the graph in order to complete this operation.', QMessageBox.Ok + QMessageBox.Default)
            return
        self.selectedExampleIndex = selected[0]
        self.selectedExampleChanged()


    def exampleListSelectionChanged(self):
        if self.exampleList.selectedItems() == []: return
        (val, exampleIndex, classPredictions) = self.evaluatedExamples[self.exampleList.row(self.exampleList.selectedItems()[0])]
        self.selectedExampleIndex = exampleIndex
        self.selectedExampleChanged()

    # draw the graph of predictions for the selected example
    def updateGraph(self):
        self.graph.clear()
        self.graph.tips.removeAll()
        if not self.data or self.graphMatrix == None: return

        classColors = ColorPaletteHSV(len(self.data.domain.classVar.values))

        self.graph.setAxisScale(QwtPlot.yLeft, 0, len(self.graphMatrix[0]), len(self.graphMatrix[0])/5)
        self.graph.setAxisScale(QwtPlot.xBottom, 0, 1, 0.2)

        valid = numpy.where(self.graphMatrix[0] != -100, 1, 0)
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
        classVariableValues = getVariableValuesSorted(self.data.domain.classVar)
        classColors = ColorPaletteHSV(len(classVariableValues))

        for i in range(len(self.graphMatrix[0])):
            x = 0
            s = "Predicted class probabilities:<br>"
            invalidValue = 0
            for j in classes:
                (prob, index) = indices[i]
                if self.graphMatrix[j][index] < 0:
                    invalidValue = 1
                    continue
                s += "&nbsp; &nbsp; &nbsp; %s: %.2f%%<br>" % (classVariableValues[j], 100*self.graphMatrix[j][index])
                if not self.showAllClasses and int(self.data[self.selectedExampleIndex].getclass()) != j:
                    continue
                xDiff = self.graphMatrix[j][index]
                RectangleCurve(QPen(classColors.getColor(j)), QBrush(classColors.getColor(j)), [x, x+xDiff, x+xDiff, x], [i, i, i+1, i+1]).attach(self.graph)
                x += xDiff
            if not invalidValue:
                self.graph.tips.addToolTip(0, i, s[:-4], 1, 1)

        if self.showLegend:
            self.graph.addCurve("<b>" + self.data.domain.classVar.name + ":</b>", QColor(0,0,0), QColor(0,0,0), 0, symbol = QwtSymbol.NoSymbol, enableLegend = 1)
            for i,val in enumerate(classVariableValues):
                self.graph.addCurve(val, classColors[i], classColors[i], 15, symbol = QwtSymbol.Rect, enableLegend = 1)

        self.selectedRectangle = RectangleCurve(brush = QBrush(Qt.NoBrush))
        self.selectedRectangle.attach(self.graph)

        self.graph.replot()

    def graphOnMouseMoved(self, e):
        y = int(self.graph.invTransform(QwtPlot.yLeft, e.y()))
        if self.showClickedProjection and y >= 0 and y < len(self.projectionIndices):
            diff  = 0.005
            self.selectedRectangle.setData([0-diff, 1+diff, 1+diff, 0-diff], [y-diff, y-diff, y+1+diff, y+1+diff])
        else:
            self.selectedRectangle.setData([], [])
        self.graph.replot()
        return 1

    def graphOnMousePressed(self, e):
        if self.showClickedProjection:
            y = int(self.graph.invTransform(QwtPlot.yLeft, e.y()))
            if y >= len(self.projectionIndices): return
            projIndex = self.projectionIndices[y]
            self.vizrank.resultList.setCurrentItem(self.vizrank.resultList.item(projIndex))

            if self.dialogType == VIZRANK_POINT:
                attrs = self.vizrank.shownResults[projIndex][self.ATTR_LIST]
                valid = self.vizrank.graph.getValidList([self.vizrank.graph.attributeNameIndex[attr] for attr in attrs])
                insideColors = numpy.zeros(len(self.data))
                insideColors[self.selectedExampleIndex] = 1
                self.widget.updateGraph(attrs, setAnchors = 1, insideColors = (numpy.compress(valid, insideColors), "Focused example: %d"))
        return 1


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
    ow.show()
    a.exec_()
