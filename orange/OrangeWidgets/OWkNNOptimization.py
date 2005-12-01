from OWBaseWidget import *
from OWWidget import OWWidget
import OWGUI, OWDlgs, OWVisTools
from OWGraph import *
from orngVizRank import *
from orngScaleData import getVariableValuesSorted


class OWVizRank(VizRank, OWBaseWidget):
    settingsList = ["kValue", "resultListLen", "percentDataUsed", "qualityMeasure", "testingMethod",
                    "lastSaveDirName", "attrCont", "attrDisc", "showRank", "showAccuracy", "showInstances",
                    "evaluationAlgorithm", "createSnapshots", "evaluationTime", "learnerName",
                    "argumentCount", "canUseMoreArguments", "moreArgumentsCount", 
                    "optimizeBestProjection", "optimizeBestProjectionTime",
                    "useHeuristicToFindAttributeOrders", "argumentValueFormula", "locOptMaxAttrsInProj",
                    "locOptProjCount", "useExampleWeighting", "useSupervisedPCA"]
    resultsListLenNums = [ 10, 100 ,  250 ,  500 ,  1000 ,  5000 ,  10000, 20000, 50000, 100000, 500000 ]
    percentDataNums = [ 5 ,  10 ,  15 ,  20 ,  30 ,  40 ,  50 ,  60 ,  70 ,  80 ,  90 ,  100 ]
    kNeighboursNums = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 30,35, 40, 50, 60, 70, 80, 100, 120, 150, 200]
    argumentCounts = [1, 3, 5, 10, 15, 20, 30, 50, 100, 200]
    #argumentCounts = range(21)[1:]
    evaluationTimeNums = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 60, 80, 120]
    moreArgumentsNums = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

    def __init__(self, parentWidget = None, signalManager = None, graph = None, visualizationMethod = SCATTERPLOT, parentName = "Visualization widget"):
        VizRank.__init__(self, visualizationMethod, graph)
        OWBaseWidget.__init__(self, None, signalManager, "Optimization Dialog")

        self.parentWidget = parentWidget
        self.parentName = parentName
        self.setCaption("Qt VizRank Optimization Dialog")
        self.controlArea = QVBoxLayout(self)

        self.resultListLen = 1000
        self.cancelOptimization = 0
        self.cancelEvaluation = 0
        self.learnerName = "VizRank Learner"
        self.subsetdata = None
        
        self.maxResultListLen = self.resultsListLenNums[len(self.resultsListLenNums)-1]
        #self.widgetDir = os.path.realpath(os.path.dirname(__file__)) + "/"
        self.lastSaveDirName = os.getcwd()
        
        self.evaluatedAttributes = None   # save last evaluated attributes
        self.evaluatedAttributesByClass = None
        self.createSnapshots = 1
        
        self.showRank = 0
        self.showAccuracy = 1
        self.showInstances = 0
        
        self.shownResults = []
        self.attrLenDict = {}
        
        self.loadSettings()

        self.tabs = QTabWidget(self, 'tabWidget')
        self.controlArea.addWidget(self.tabs)
        
        self.MainTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self)
        self.ManageTab = QVGroupBox(self)
        self.ArgumentationTab = QVGroupBox(self)
        self.ClassificationTab = QVGroupBox(self)
        
        self.tabs.insertTab(self.MainTab, "Main")
        self.tabs.insertTab(self.SettingsTab, "Settings")
        self.tabs.insertTab(self.ArgumentationTab, "Argumentation")
        self.tabs.insertTab(self.ClassificationTab, "Classification")
        self.tabs.insertTab(self.ManageTab, "Manage & Save")        

        # ###########################
        # MAIN TAB
        self.optimizationBox = OWGUI.widgetBox(self.MainTab, " Evaluate ")
        self.resultsBox = OWGUI.widgetBox(self.MainTab, " Projection List, Most Interesting Projections First ")
        self.resultsDetailsBox = OWGUI.widgetBox(self.MainTab, " Shown Details in Projections List " , orientation = "horizontal")
        self.buttonBox = OWGUI.widgetBox(self.optimizationBox, orientation = "horizontal")

        if visualizationMethod != SCATTERPLOT:
            self.label1 = QLabel('Projections with ', self.buttonBox)
            self.optimizationTypeCombo = OWGUI.comboBox(self.buttonBox, self, "optimizationType", items = ["    exactly    ", "  maximum  "] )
            self.attributeCountCombo = OWGUI.comboBox(self.buttonBox, self, "attributeCount", items = range(3, 15), tooltip = "Evaluate only projections with exactly (or maximum) this number of attributes", sendSelectedValue = 1, valueType = int)
            self.attributeLabel = QLabel(' attributes', self.buttonBox)

        self.startOptimizationButton = OWGUI.button(self.optimizationBox, self, "Start Evaluating Projections", callback = self.evaluateProjections)
        f = self.startOptimizationButton.font(); f.setBold(1);   self.startOptimizationButton.setFont(f)
        self.stopOptimizationButton = OWGUI.button(self.optimizationBox, self, "Stop evaluation", callback = self.stopOptimizationClick)
        self.stopOptimizationButton.setFont(f)
        self.stopOptimizationButton.hide()
        self.optimizeGivenProjectionButton = OWGUI.button(self.optimizationBox, self, "Optimize current projection", callback = self.optimizeBestProjections)
        self.useHeuristicToFindAttributeOrderCheck = OWGUI.checkBox(self.optimizationBox, self, 'useHeuristicToFindAttributeOrders', 'Use Heuristic to Find Attribute Orders', tooltip = "Don't try all possible permutations of an attribute subset but only those,\nthat will most likely produce interesting projections.", callback = self.removeEvaluatedAttributes)
        if visualizationMethod == SCATTERPLOT: self.useHeuristicToFindAttributeOrderCheck.hide()

        self.resultList = QListBox(self.resultsBox)
        #self.resultList.setSelectionMode(QListBox.Extended)   # this would be nice if could be enabled, but it has a bug - currentItem doesn't return the correct value if this is on
        self.resultList.setMinimumSize(200,200)
        if self.parentWidget: self.connect(self.resultList, SIGNAL("selectionChanged()"),self.parentWidget.showSelectedAttributes)

        self.showRankCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showRank', 'Rank', callback = self.updateShownProjections, tooltip = "Show projection ranks")
        self.showAccuracyCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showAccuracy', 'Score', callback = self.updateShownProjections, tooltip = "Show prediction accuracy of a k-NN classifier on the projection")
        self.showInstancesCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showInstances', '# Instances', callback = self.updateShownProjections, tooltip = "Show number of instances in the projection")

        # ##########################
        # SETTINGS TAB
        self.optimizationSettingsBox = OWGUI.widgetBox(self.SettingsTab, " VizRank Evaluation Settings ")
        self.methodTypeCombo = OWGUI.comboBoxWithCaption(self.optimizationSettingsBox, self, "evaluationAlgorithm", "Projection Evaluation Method", tooltip = "Which learning method to use to use to evaluate given projections.", items = ["k-Nearest Neighbor", "Fisher Discriminant Analysis", "Heuristic (very fast)"])
        self.attrKNeighboursEdit = OWGUI.lineEdit(self.optimizationSettingsBox, self, "kValue", "Number of neighbors (k):                ", orientation = "horizontal", tooltip = "Number of neighbors used in k-NN algorithm to evaluate the projection", valueType = int, validator = QIntValidator(self))
        self.percentDataUsedCombo= OWGUI.comboBoxWithCaption(self.optimizationSettingsBox, self, "percentDataUsed", "Percent of data used in evaluation: ", items = self.percentDataNums, sendSelectedValue = 1, valueType = int)
        OWGUI.checkBox(self.optimizationSettingsBox, self, 'useExampleWeighting', 'Use example weighting (in case of uneven class distribution)', tooltip = "Don't try all possible permutations of an attribute subset but only those,\nthat will most likely produce interesting projections.")

        if visualizationMethod == RADVIZ:
            OWGUI.checkBox(self.SettingsTab, self, 'useSupervisedPCA', 'Optimize class separation using supervised PCA', box = " Supervised PCA ")
        
        self.heuristicsSettingsBox = OWGUI.widgetBox(self.SettingsTab, " Heuristics for Attribute Ranking ")
        OWGUI.comboBoxWithCaption(self.heuristicsSettingsBox, self, "attrCont", " Ranking of Continuous Attributes: ", items = [val for (val, m) in contMeasures], callback = self.removeEvaluatedAttributes)
        OWGUI.comboBoxWithCaption(self.heuristicsSettingsBox, self, "attrDisc", " Ranking of Discrete Attributes: ", items = [val for (val, m) in discMeasures], callback = self.removeEvaluatedAttributes)

        self.measureCombo = OWGUI.comboBox(self.SettingsTab, self, "qualityMeasure", box = " Measure of Classification Success ", items = ["Classification accuracy", "Average probability assigned to the correct class", "Brier score"], tooltip = "Measure to evaluate prediction accuracy of k-NN method on the projected data set.")
        self.testingCombo = OWGUI.comboBox(self.SettingsTab, self, "testingMethod", box = " Testing Method ", items = ["Leave one out (slowest, most accurate)", "10 fold cross validation", "Test on learning set (fastest, least accurate)"], tooltip = "Method for evaluating the classifier. Slower are more accurate while faster give only a rough approximation.")        

        self.localOptimizationSettingsBox = OWGUI.widgetBox(self.SettingsTab, " Local Optimization Settings ")
        self.localOptimizationProjCountCombo = OWGUI.comboBoxWithCaption(self.localOptimizationSettingsBox , self, "locOptProjCount", "Number of best projections to optimize:           ", items = range(1,30), tooltip = "Specify the number of best projections in the list that you want to try to locally optimize.\nIf you select 1 only the currently selected projection will be optimized.", sendSelectedValue = 1, valueType = int)
        self.localOptimizationProjMaxAttr    = OWGUI.comboBoxWithCaption(self.localOptimizationSettingsBox , self, "locOptMaxAttrsInProj", "Maximum number of attributes in a projection: ", items = range(3,50), tooltip = "What is the number of attributes when local optimization should stop optimizing projections?", sendSelectedValue = 1, valueType = int)

        self.miscSettingsBox = OWGUI.widgetBox(self.SettingsTab, " Length of the Projection List ")
        self.resultListCombo = OWGUI.comboBoxWithCaption(self.miscSettingsBox, self, "resultListLen", "Maximum length of projection list:   ", tooltip = "Maximum length of the list of interesting projections. This is also the number of projections that will be saved if you click Save button.", items = self.resultsListLenNums, callback = self.updateShownProjections, sendSelectedValue = 1, valueType = int)        

        # ##########################
        # ARGUMENTATION TAB
        self.argumentationBox = OWGUI.widgetBox(self.ArgumentationTab, " Arguments ")
        self.findArgumentsButton = OWGUI.button(self.argumentationBox, self, "Find Arguments", callback = self.findArguments)
        f = self.findArgumentsButton.font(); f.setBold(1);  self.findArgumentsButton.setFont(f)
        self.stopArgumentationButton = OWGUI.button(self.argumentationBox, self, "Stop Searching", callback = self.stopArgumentationClick)
        self.stopArgumentationButton.setFont(f)
        self.stopArgumentationButton.hide()
        self.createSnapshotCheck = OWGUI.checkBox(self.argumentationBox, self, 'createSnapshots', 'Create snapshots of projections (a bit slower)', tooltip = "Show each argument with a projections screenshot.\nTakes a bit more time, since the projection has to be created.")
        self.classValueList = OWGUI.comboBox(self.ArgumentationTab, self, "argumentationClassValue", box = " Arguments For Class: ", tooltip = "Select the class value that you wish to see arguments for", callback = self.argumentationClassChanged)
        self.argumentBox = OWGUI.widgetBox(self.ArgumentationTab, " Arguments for The Selected Class Value ")
        self.argumentList = QListBox(self.argumentBox)
        self.argumentList.setMinimumSize(200,200)
        self.connect(self.argumentList, SIGNAL("selectionChanged()"),self.argumentSelected)

        # ##########################
        # CLASSIFICATION TAB
        self.classifierNameEdit = OWGUI.lineEdit(self.ClassificationTab, self, 'learnerName', box = ' Learner / Classifier Name ', tooltip='Name to be used by other widgets to identify your learner/classifier.')

        self.argumentValueFormulaIndex = OWGUI.comboBox(self.ClassificationTab, self, "argumentValueFormula", box="Argument Value is Computed As ...", items=["1.0 x Projection Value", "0.5 x Projection Value + 0.5 x Predicted Example Probability", "1.0 x Predicted Example Probability"], tooltip=None)

        b = OWGUI.widgetBox(self.ClassificationTab, " Evaluating Time ")
        self.evaluationTimeEdit = OWGUI.comboBoxWithCaption(b, self, "evaluationTime", "Time for evaluating projections (minutes):                ", tooltip = "What is the maximum time that the classifier is allowed for evaluating projections (learning)", items = self.evaluationTimeNums, sendSelectedValue = 1, valueType = float)
        b2 = OWGUI.widgetBox(b, orientation = "horizontal")
        self.optimizeBestProjectionCheck = OWGUI.checkBox(b2, self, "optimizeBestProjection", "Afterwards use local optimization for (minutes): ", tooltip = "Do you want to try to locally optimize the best projection when the time for evaluating projections runs out?")
        self.optimizeBestProjectionCombo = OWGUI.comboBox(b2, self, "optimizeBestProjectionTime", items = self.evaluationTimeNums, sendSelectedValue = 1, valueType = float)
        projCountBox = OWGUI.widgetBox(self.ClassificationTab, " Projection Count ")
        self.argumentCountEdit = OWGUI.comboBoxWithCaption(projCountBox, self, "argumentCount", "Number of projections used when classifying:                ", tooltip = "What is the maximum number of projections (arguments) that will be used when classifying an example.", items = self.argumentCounts, sendSelectedValue = 1, valueType = int)
        projCountBox2 = OWGUI.widgetBox(projCountBox, orientation = "horizontal")
        self.canUseMoreArgumentsCheck = OWGUI.checkBox(projCountBox2, self, "canUseMoreArguments", "Use additional projections until probability at least: ", tooltip = "If checked, it will allow the classifier to use more arguments when it is not confident enough in the prediction.\nIt will use additional arguments until the predicted probability of one class value will be at least as much as specified in the combo box")
        self.moreArgumentsCombo = OWGUI.comboBox(projCountBox2, self, "moreArgumentsCount", items = self.moreArgumentsNums, tooltip = "If checked, it will allow the classifier to use more arguments when it is not confident enough in the prediction.\nIt will use additional arguments until the predicted probability of one class value will be at least as much as specified in the combo box", sendSelectedValue = 1, valueType = int)

        # ##########################
        # SAVE & MANAGE TAB
        self.classesBox = OWGUI.widgetBox(self.ManageTab, " Select Class Values You Wish to Separate ")
        self.visualizedAttributesBox = OWGUI.widgetBox(self.ManageTab, " Number of Concurrently Visualized Attributes ")
        self.dialogsBox = OWGUI.widgetBox(self.ManageTab, " Dialogs ")        
        self.manageResultsBox = OWGUI.widgetBox(self.ManageTab, " Manage Projections ")        
        
        self.classesList = QListBox(self.classesBox)
        self.classesList.setSelectionMode(QListBox.Multi)
        self.classesList.setMinimumSize(60,60)
        self.connect(self.classesList, SIGNAL("selectionChanged()"), self.classesListChanged)
        
        self.attrLenList = QListBox(self.visualizedAttributesBox)
        self.attrLenList.setSelectionMode(QListBox.Multi)
        self.attrLenList.setMinimumSize(60,60)
        self.connect(self.attrLenList, SIGNAL("selectionChanged()"), self.attrLenListChanged)
        
        #self.removeSelectedButton = OWGUI.button(self.buttonBox5, self, "Remove selection", self.removeSelected)
        #self.filterButton = OWGUI.button(self.buttonBox5, self, "Save best graphs", self.exportMultipleGraphs)

        self.buttonBox7 = OWGUI.widgetBox(self.dialogsBox, orientation = "horizontal")
        OWGUI.button(self.buttonBox7, self, "Attribute Analysis", self.attributeAnalysis)
        OWGUI.button(self.buttonBox7, self, "Interaction Analysis", self.interactionAnalysis)

        self.buttonBox8 = OWGUI.widgetBox(self.dialogsBox, orientation = "horizontal")    
        OWGUI.button(self.buttonBox8, self, "Graph projections", self.graphProjectionQuality)
        OWGUI.button(self.buttonBox8, self, "Identify outliers", self.identifyOutliers)

        self.buttonBox6 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.loadButton = OWGUI.button(self.buttonBox6, self, "Load", self.loadProjections)
        self.saveButton = OWGUI.button(self.buttonBox6, self, "Save", self.saveProjections)

        self.buttonBox9 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.saveBestButton = OWGUI.button(self.buttonBox9, self, "Save best graphs", self.exportMultipleGraphs)
        OWGUI.button(self.buttonBox9, self, "Remove similar projections", callback = self.removeTooSimilarProjections)

        self.buttonBox3 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        if self.parentWidget:
            self.evaluateProjectionButton = OWGUI.button(self.buttonBox3, self, 'Evaluate projection', callback = self.evaluateCurrentProjection)
        self.reevaluateResults = OWGUI.button(self.buttonBox3, self, "Reevaluate projections", callback = self.reevaluateAllProjections)

        self.buttonBox4 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.showKNNCorrectButton = OWGUI.button(self.buttonBox4, self, 'Show k-NN correct', self.showKNNCorect)
        self.showKNNWrongButton = OWGUI.button(self.buttonBox4, self, 'Show k-NN wrong', self.showKNNWrong)
        self.showKNNCorrectButton.setToggleButton(1); self.showKNNWrongButton.setToggleButton(1)
        
        self.buttonBox5 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.clearButton = OWGUI.button(self.buttonBox5, self, "Clear results", self.clearResults)
            
        # ###########################
        self.statusBar = QStatusBar(self)
        self.controlArea.addWidget(self.statusBar)
        self.controlArea.activate()

        self.removeEvaluatedAttributes()
        self.resize(375,550)
        self.setMinimumWidth(375)
        self.tabs.setMinimumWidth(375)

    # ##############################################################
    # EVENTS
    # ##############################################################

    # the heuristic checkbox is enabled only if the signal to noise OVA measure is selected
    def removeEvaluatedAttributes(self):
        # clear the list of evaluated attributes
        self.evaluatedAttributes = None
        self.evaluatedAttributesByClass = None
        
        # update heuristic check box
        self.useHeuristicToFindAttributeOrderCheck.setEnabled(contMeasures[self.attrCont][0] == "Signal to Noise OVA")
        if not self.data or not self.data.domain.classVar or self.data.domain.classVar.varType != orange.VarTypes.Discrete or contMeasures[self.attrCont][0] != "Signal to Noise OVA":
            self.useHeuristicToFindAttributeOrders = 0
            self.useHeuristicToFindAttributeOrderCheck.setEnabled(0)
        
       
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
            for result in results:
                self.addResult(result[OTHER_RESULTS][0], result[OTHER_RESULTS], result[LEN_TABLE], result[ATTR_LIST], result[TRY_INDEX], result[GENERAL_DICT])
        else: 
            for result in results:
                acc = 0.0; sum = 0.0
                for index in self.selectedClasses:
                    acc += result[OTHER_RESULTS][OTHER_PREDICTIONS][index] * result[OTHER_RESULTS][OTHER_DISTRIBUTION][index]; sum += result[OTHER_RESULTS][OTHER_DISTRIBUTION][index]
                self.addResult(acc/sum, result[OTHER_RESULTS], result[LEN_TABLE], result[ATTR_LIST], result[TRY_INDEX], result[GENERAL_DICT])
                
        self.finishedAddingResults()

    def clearResults(self):
        VizRank.clearResults(self)
        del self.shownResults; self.shownResults = []
        self.resultList.clear()
        self.attrLenDict = {}
        self.attrLenList.clear()

    def clearArguments(self):
        del self.arguments
        self.arguments = []
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
    def removeTooSimilarProjections(self):
        i=0
        qApp.setOverrideCursor(QWidget.waitCursor)
        self.setStatusBarText("Removing similar projections")
        while i < self.resultList.count():
            qApp.processEvents()
            if self.existsABetterSimilarProjection(i):
                self.results.pop(i)
                self.shownResults.pop(i)
                self.resultList.removeItem(i)
            else:
                i += 1
                
        self.setStatusBarText("")
        qApp.restoreOverrideCursor()


    def updateShownProjections(self, *args):
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

        if not (data and data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete):
            self.classesList.clear()
            self.classValueList.clear()
            self.selectedClasses = []
            return

        if hasattr(data, "name"): self.datasetName = data.name
        else: self.datasetName = ""

        self.classesList.clear()
        self.classValueList.clear()
        self.selectedClasses = []

        if not (data and data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete): return

        # add class values
        for i in range(len(data.domain.classVar.values)):
            self.classesList.insertItem(data.domain.classVar.values[i])
            self.classValueList.insertItem(data.domain.classVar.values[i])
        self.classesList.selectAll(1)
        #if len(data.domain.classVar.values) > 0: self.classValueList.setCurrentItem(0)

                
    # given a dataset return a list of attributes where attribute are sorted by their decreasing importance for class discrimination
    def getEvaluatedAttributes(self, data):
        if self.evaluatedAttributes: return self.evaluatedAttributes
        
        self.setStatusBarText("Evaluating attributes...")
        qApp.setOverrideCursor(QWidget.waitCursor)
        #attrs = None

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
            

##            # is this the radviz widget, where anchors are not on the circle. if yes, then use the distance from the center of the circle as an indication of attribute usefulness. more distant attributes
##            # are expected to be more important for discriminating between classes
##            if self.parentName == "Radviz" and self.parentWidget.graph.anchorData != []:
##                for i in range(min(5, self.parentWidget.shownAttribsLB.count())):
##                    if attrs == None and abs(self.parentWidget.graph.anchorData[i][0]**2 + self.parentWidget.graph.anchorData[i][1]**2 -1) > 0.001:
##                        c = self.parentWidget.shownAttribsLB.count()
##                        attrs = [(self.graph.anchorData[j][0]**2 + self.graph.anchorData[j][1]**2, self.graph.anchorData[j][2]) for j in range(len(self.graph.anchorData))]
##                        attrs.sort()
##                        attrs.reverse()
##                        attrs = [attr for (val, attr) in attrs]
##
##            # evaluate attributes using the selected attribute measure
##            if attrs == None:
##                if self.evaluatedAttributes[0] != self.attrCont or self.evaluatedAttributes[1] != self.attrDisc or self.evaluatedAttributes[2] == None: 
##                    attrs = OWVisAttrSelection.evaluateAttributes(data, contMeasures[self.attrCont][1], discMeasures[self.attrDisc][1])
##                    self.evaluatedAttributes = (self.attrCont, self.attrDisc, attrs)
##                else:
##                    attrs = self.evaluatedAttributes[2]
            
            # evaluate attributes using the selected attribute measure
            self.evaluatedAttributes = OWVisAttrSelection.evaluateAttributes(data, contMeasures[self.attrCont][1], discMeasures[self.attrDisc][1])
        except:
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # print the exception
            
        self.setStatusBarText("")
        qApp.restoreOverrideCursor()

        if self.evaluatedAttributes == None: return []
        else:             return self.evaluatedAttributes

    
    # insert new result - give parameters: accuracy of projection, number of examples in projection and list of attributes.
    def insertItem(self, accuracy, other_results, lenTable, attrList, index, tryIndex, generalDict = {}):
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
            qApp.processEvents()        # allow processing of other events

    
    def finishedAddingResults(self):
        self.cancelOptimization = 0
        
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

        
    # ##############################################################
    # kNNClassifyData - compute classification error for every example in table
    def kNNClassifyData(self, table):
        qApp.processEvents()        # allow processing of other events
        
        #knn = orange.kNNLearner(k=self.kValue, rankWeight = 0, distanceConstructor = orange.ExamplesDistanceConstructor_Euclidean(normalize=0))
        #results = apply(testingMethods[self.testingMethod], [[knn], table])
        if self.externalLearner: learner = self.externalLearner
        else:                    learner = self.createkNNLearner()
        results = apply(testingMethods[self.testingMethod], [[learner], table])
            
        returnTable = []
        
        if table.domain.classVar.varType == orange.VarTypes.Discrete:
            probabilities = Numeric.zeros((len(table), len(table.domain.classVar.values)), Numeric.Float)
            lenClassValues = len(list(table.domain.classVar.values))
            if self.qualityMeasure == AVERAGE_CORRECT:
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
            probabilities = None
            # for continuous class we can't compute brier score and classification accuracy
            for res in results.results:
                if not res.probabilities[0]: returnTable.append(0)
                else:                        returnTable.append(res.probabilities[0].density(res.actualClass))

        del results
        return returnTable, probabilities


    # reevaluate projections in result list with the current VizRank settings (different k value, different measure of classification succes, ...)
    def reevaluateAllProjections(self):
        results = list(self.getShownResults())
        self.clearResults()

        self.parentWidget.progressBarInit()
        self.disableControls()

        testIndex = 0
        strTotal = OWVisFuncts.createStringFromNumber(len(results))
        for (acc, other, tableLen, attrList, tryIndex, generalDict) in results:
            if self.isOptimizationCanceled(): break
            testIndex += 1
            self.parentWidget.progressBarSet(100.0*testIndex/float(len(results)))

            accuracy, other_results = self.getProjectionQuality(attrList)         # TO DO: this does not work with polyviz with attrReverseList   

            self.addResult(accuracy, other_results, tableLen, attrList, tryIndex, generalDict)
            self.setStatusBarText("Reevaluated %s/%s projections..." % (OWVisFuncts.createStringFromNumber(testIndex), strTotal))

        self.setStatusBarText("")
        self.parentWidget.progressBarFinished()
        self.enableControls()
        self.finishedAddingResults()

    # evaluate knn accuracy on current projection
    def evaluateCurrentProjection(self):
        acc, other_results = self.getProjectionQuality(self.parentWidget.getShownAttributeList())

        if self.data.domain.classVar.varType == orange.VarTypes.Continuous:
            QMessageBox.information( None, self.parentName, 'Mean square error of kNN model is %.2f'%(acc), QMessageBox.Ok + QMessageBox.Default)
        else:
            if self.qualityMeasure == CLASS_ACCURACY:
                QMessageBox.information( None, self.parentName, 'Classification accuracy of kNN model is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
            elif self.qualityMeasure == AVERAGE_CORRECT:
                QMessageBox.information( None, self.parentName, 'Average probability of correct classification is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
            else:
                QMessageBox.information( None, self.parentName, 'Brier score of kNN model is %.2f' % (acc), QMessageBox.Ok + QMessageBox.Default)

    
    # ##############################################################
    # Loading and saving projection files
    # ##############################################################

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

        VizRank.save(self, name, self.shownResults, len(self.shownResults))

        self.setStatusBarText("Saved %d projections" % (len(self.shownResults)))


    # load projections from a file
    def loadProjections(self, name = None, ignoreCheckSum = 0):
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

        selectedClasses, count = VizRank.load(self, name, ignoreCheckSum)

        for i in range(len(self.data.domain.classVar.values)): self.classesList.setSelected(i, i in selectedClasses)
        self.setStatusBarText("Loaded %d projections" % (count))

    def showKNNCorect(self):
        self.showKNNWrongButton.setOn(0)
        if self.parentWidget: self.parentWidget.updateGraph()

    # show quality of knn model by coloring accurate predictions with lighter color and bad predictions with dark color
    def showKNNWrong(self):
        self.showKNNCorrectButton.setOn(0) 
        if self.parentWidget: self.parentWidget.updateGraph()


    # disable all controls while evaluating projections
    def disableControls(self):
        self.buttonBox.setEnabled(0)
        self.useHeuristicToFindAttributeOrderCheck.setEnabled(0)
        self.startOptimizationButton.hide()
        self.stopOptimizationButton.show()
        self.resultsDetailsBox.setEnabled(0)
        self.optimizeGivenProjectionButton.setEnabled(0)
        self.SettingsTab.setEnabled(0)
        self.ManageTab.setEnabled(0)
        self.ClassificationTab.setEnabled(0)
        self.ArgumentationTab.setEnabled(0)
        
    def enableControls(self):
        self.buttonBox.setEnabled(1)
        self.useHeuristicToFindAttributeOrderCheck.setEnabled(1)
        self.startOptimizationButton.show()
        self.stopOptimizationButton.hide()
        self.resultsDetailsBox.setEnabled(1)
        self.optimizeGivenProjectionButton.setEnabled(1)
        self.SettingsTab.setEnabled(1)
        self.ManageTab.setEnabled(1)
        self.ClassificationTab.setEnabled(1)
        self.ArgumentationTab.setEnabled(1)
        
    # ##############################################################
    # exporting multiple pictures
    # ##############################################################
    def exportMultipleGraphs(self):
        (text, ok) = QInputDialog.getText('Qt Graph count', 'How many of the best projections do you wish to save?')
        if not ok: return
        self.bestGraphsCount = int(str(text))

        self.sizeDlg = OWDlgs.OWChooseImageSizeDlg(self.graph)
        self.sizeDlg.disconnect(self.sizeDlg.okButton, SIGNAL("clicked()"), self.sizeDlg.accept)
        self.sizeDlg.connect(self.sizeDlg.okButton, SIGNAL("clicked()"), self.saveToFileAccept)
        self.sizeDlg.exec_loop()

    def saveToFileAccept(self):
        fileName = str(QFileDialog.getSaveFileName("Graph","Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)", None, "Save to..", "Save to.."))
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        if ext == "":	
        	ext = "PNG"  	# if no format was specified, we choose png
        	fileName = fileName + ".png"
        ext = ext.upper()

        (fil, extension) = os.path.splitext(fileName)
        size = self.sizeDlg.getSize()
        for i in range(1, min(self.resultList.count(), self.bestGraphsCount+1)):
            self.resultList.setSelected(i-1, 1)
            self.graph.replot()
            name = fil + " (%02d)" % i + extension
            self.sizeDlg.saveToFileDirect(name, ext, size)
        QDialog.accept(self.sizeDlg)

    def interactionAnalysis(self):
        dialog = OWInteractionAnalysis(self, signalManager = self.signalManager)
        dialog.setResults(self.shownResults, VIZRANK)
        dialog.show()


    def attributeAnalysis(self):
        dialog = OWGraphAttributeHistogram(self, signalManager = self.signalManager)
        dialog.setResults(self.shownResults)
        dialog.show()

    def graphProjectionQuality(self):
        dialog = OWGraphProjectionQuality(self, signalManager = self.signalManager)
        dialog.setResults(self.results, VIZRANK)
        dialog.show()

    def identifyOutliers(self):
        dialog = OWGraphIdentifyOutliers(self, signalManager = self.signalManager, widget = self.parentWidget, graph = self.graph)
        dialog.setData(self.results, self.data, VIZRANK)
        dialog.show()

    # ######################################################
    # Auxiliary functions
    # ######################################################
    
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
            strList = attrList[0]
            for attr in attrList[1:]:
                strList += ", " + attr
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

    def stopOptimizationClick(self):
        self.cancelOptimization = 1

    def stopEvaluationClick(self):
        self.cancelEvaluation = 1

    def isOptimizationCanceled(self):
        if hasattr(self, "useTimeLimit"):   return VizRank.isOptimizationCanceled(self)
        else:                               return self.cancelOptimization

    def isEvaluationCanceled(self):
        if hasattr(self, "useTimeLimit"): return VizRank.isEvaluationCanceled(self)
        else:                             return self.cancelOptimization

    def destroy(self, dw = 1, dsw = 1):
        self.saveSettings()

    # save subsetdata. first example from this dataset can be used with argumentation - it can find arguments for classifying the example to the possible class values
    def setSubsetData(self, subsetdata):
        self.subsetdata = subsetdata
        self.clearArguments()

    # ######################################################
    # Argumentation functions
    # ######################################################
    def findArguments(self, example = None, selectBest = 1, showClassification = 1):
        self.cancelArgumentation = 0
        self.clearArguments()
        self.arguments = [[] for i in range(self.classValueList.count())]
        snapshots = self.createSnapshots
        
        if not example and self.subsetdata == None:
            QMessageBox.information( None, "VizRank Argumentation", 'To find arguments you first have to provide a new example that you wish to classify. \nYou can do this by sending the example to the visualization widget through the "Example Subset" signal.', QMessageBox.Ok + QMessageBox.Default)
            return (None,None)
        if len(self.shownResults) == 0:
            QMessageBox.information( None, "VizRank Argumentation", 'To find arguments you first have to evaluate some projections by clicking "Start evaluating projections" in the Main tab.', QMessageBox.Ok + QMessageBox.Default)
            return (None,None)

        if example == None: example = self.subsetdata[0]
        scaleFunction = self.parentWidget.graph.scaleExampleValue   # so that we don't have to search the dictionaries each time
        #testExample = [scaleFunction(example, i) for i in range(len(example.domain.attributes))]
        testExample = ["?"] * len(example.domain.attributes)

        self.findArgumentsButton.hide()
        self.stopArgumentationButton.show()
        if snapshots: self.parentWidget.setMinimalGraphProperties()

        argumentList = []

        foundArguments = 0
        for index in range(min(len(self.shownResults), 1000)):       # use only best argumentCount projections for argumentation
            if self.cancelArgumentation: break          # user pressed cancel
            # we also stop if we are not allowed to search for more than argumentCount arguments or we are allowed and we have a reliable prediction or we have used a 100 additional arguments
            #if foundArguments >= argumentCount and (not self.canUseMoreArguments or (max(vals)*100.0 / sum(vals) > self.moreArgumentsNums[self.moreArgumentsIndex]) or foundArguments >= argumentCount + 100): break

            qApp.processEvents()
            (accuracy, other_results, lenTable, attrList, tryIndex, generalDict) = self.results[index]
            
            validExample = 1
            for attr in attrList:
                if example[attr].isSpecial():
                    validExample = 0
                    continue

            if not validExample:
                self.printVerbose("Warning: OWkNNOptimization.py:findArguments: Tested example has a missing value at one of the visualized attributes. Skipping the projection.")
                continue

            attrVals = []
            for i in range(len(attrList)):
                attrIndex = self.graph.attributeNameIndex[attrList[i]]
                if testExample[attrIndex] == "?":
                    testExample[attrIndex] = scaleFunction(example, attrIndex)
                attrVals.append(testExample[attrIndex])
                        
            if min(attrVals) < 0.0 or max(attrVals) > 1.0:
                self.printVerbose("Warning: OWkNNOptimization.py:findArguments: Scaled example value out of 0-1 range. Min value: %.3f, max value: %.3f." % (min(attrVals), max(attrVals)))

            if generalDict.has_key("anchors"): xanchors = generalDict["anchors"][0]; yanchors = generalDict["anchors"][1]
            else:   xanchors = None; yanchors = None
            [xTest, yTest] = self.graph.getProjectedPointPosition(attrList, attrVals, XAnchors = xanchors, YAnchors = yanchors)
            table = self.graph.createProjectionAsExampleTable([self.attributeNameIndex[attr] for attr in attrList], XAnchors = xanchors, YAnchors = yanchors)
            
            learner = self.externalLearner or self.createkNNLearner()
            classifier = learner(table)
            (classValue, prob) = classifier(orange.Example(table.domain, [xTest, yTest, "?"]), orange.GetBoth)
            del classifier
            classValue = int(classValue)
            if self.argumentValueFormula == 0:
                value = accuracy
                if index >= self.argumentCount-1: self.cancelArgumentation = 1   # we stop searching for arguments if argumentValueFormula = 0 and we already considered enough top projections
            elif self.argumentValueFormula == 1:
                value = 0.5 * accuracy + 50.0 * prob[classValue]
            else:
                value = 100.0 * prob[classValue]

            pic = None
            if snapshots:            
                # if the point lies inside a cluster -> save this figure into a pixmap
                if self.parentName == "Radviz": self.parentWidget.updateGraph(attrList, setAnchors = 1)
                else:                           self.parentWidget.updateGraph(attrList)
                painter = QPainter()
                pic = QPixmap(QSize(120,120))
                painter.begin(pic)
                painter.fillRect(pic.rect(), QBrush(Qt.white)) # make background same color as the widget's background
                self.graph.printPlot(painter, pic.rect())
                painter.flush();  painter.end()

            ind = self.getArgumentIndex(value, classValue)
            self.arguments[classValue].insert(ind, (pic, value, accuracy, 100.0 * prob[classValue], prob, attrList, index))
            argumentList.append((value, classValue))
            if classValue == self.classValueList.currentItem():
                if snapshots: self.argumentList.insertItem(pic, "%.2f (%.2f, %.2f) - %s" %(value, accuracy, 100.0*prob[classValue], attrList), ind)
                else:         self.argumentList.insertItem("%.2f (%.2f, %.2f) - %s" %(value, accuracy, 100.0*prob[classValue], attrList), ind)

        self.stopArgumentationButton.hide()
        self.findArgumentsButton.show()
        self.parentWidget.restoreGraphProperties()
        if self.argumentList.count() > 0 and selectBest: self.argumentList.setCurrentItem(0)
        if len(argumentList) == 0: return (None, None)

        # sort all arguments and compute the outcome
        argumentList.sort()
        argumentList.reverse()
        vals = [0.0 for i in range(len(self.arguments))]
        for i in range(min(self.argumentCount, len(argumentList))):
            vals[argumentList[i][1]] += argumentList[i][0]

        if self.canUseMoreArguments and (max(vals)*100.0 / sum(vals) < self.moreArgumentsCount):
            for i in range(self.argumentCount, min(self.argumentCount + 100, len(self.shownResults))):
                if max(vals)*100.0 / sum(vals) > self.moreArgumentsCount: break
                vals[argumentList[i][1]] += argumentList[i][0]

        suma = sum(vals)
        dist = orange.DiscDistribution([val/float(suma) for val in vals]);  dist.variable = self.data.domain.classVar
        classValue = example.domain.classVar[vals.index(max(vals))]
        s = '<nobr>Based on current classification settings, the example would be classified </nobr><br><nobr>to class <b>%s</b> with probability <b>%.2f%%</b>.</nobr><br><nobr>Predicted class distribution is:</nobr><br>' % (classValue, dist[classValue]*100)
        for key in dist.keys():
            s += "<nobr>&nbsp &nbsp &nbsp &nbsp %s : %.2f%%</nobr><br>" % (key, dist[key]*100)
        if foundArguments > self.argumentCount:
            s += "<nobr>Note: To get the current prediction, <b>%d</b> arguments had to be used (instead of %d)<br>" % (foundArguments, self.argumentCount)
        s = s[:-4]
        #print s
##        if showClassification or (not example[example.domain.classVar.name].isSpecial() and example.getclass().value != classValue):
##            self.show()
##            QMessageBox.information(None, "Classification results", s, QMessageBox.Ok + QMessageBox.Default)
##            while self.isVisible():
##                qApp.processEvents()
        # TO DO
        if showClassification:
            QMessageBox.information(None, "Classification results", s, QMessageBox.Ok + QMessageBox.Default)
        return classValue, dist
       
    def stopArgumentationClick(self):
        self.cancelArgumentation = 1
    
    def argumentationClassChanged(self):
        self.argumentList.clear()
        if len(self.arguments) == 0: return
        ind = self.classValueList.currentItem()
        for i in range(len(self.arguments[ind])):
            val = self.arguments[ind][i]
            if val[0] != None:  self.argumentList.insertItem(val[0], "%.2f (%.2f, %.2f) - %s" %(val[1], val[2], val[3], val[5]))
            else:               self.argumentList.insertItem("%.2f (%.2f, %.2f) - %s" %(val[1], val[2], val[3], val[5]))

    def argumentSelected(self):
        ind = self.argumentList.currentItem()
        classInd = self.classValueList.currentItem()
        if self.parentName == "Radviz": self.parentWidget.updateGraph(self.arguments[classInd][ind][5], setAnchors = 1)
        else:                           self.parentWidget.updateGraph(self.arguments[classInd][ind][5])
        
    def setStatusBarText(self, text):
        self.statusBar.message(text)
        qApp.processEvents()


VIZRANK = 0
CLUSTER = 1

# #############################################################################
# analyse the attributes that appear in the top projections. show how often do they appear also in other top projections
class OWInteractionAnalysis(OWWidget):
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Interaction Analysis", wantGraph = 1)

        self.attributeCount = 10
        self.projectionCount = 50
        self.rotateXAttributes = 1
        self.onlyLower = 1
        self.useDarkness = 1
        self.results = None
        self.dialogType = -1

        self.graph = OWGraph(self.mainArea)
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)
        self.box.activate()

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        b1 = OWGUI.widgetBox(self.controlArea, 'Number Of Attributes')
        b2 = OWGUI.widgetBox(self.controlArea, 'Number Of Projections')
        b3 = OWGUI.widgetBox(self.controlArea, "Settings")
        
        OWGUI.hSlider(b1, self, 'attributeCount', minValue=5, maxValue = 200, step=1, callback = self.updateGraph, ticks=5)
        self.projectionCountSlider = OWGUI.hSlider(b2, self, 'projectionCount', minValue=1, maxValue = 100000, step=1, callback = self.updateGraph, ticks=5)
        OWGUI.checkBox(b3, self, 'rotateXAttributes', label = "Rotate X labels", callback = self.updateGraph)
        OWGUI.checkBox(b3, self, 'onlyLower', label = "Show only lower diagonal", callback = self.updateGraph)
        OWGUI.checkBox(b3, self, 'useDarkness', label = "Use color to represent projection quality", callback = self.updateGraph)

        box = OWGUI.widgetBox(self.controlArea, "")
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.MinimumExpanding ))

        b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b2.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b3.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        
        self.updateGraph()

    def setResults(self, results, dialogType):
        self.results = results
        self.dialogType = dialogType
        if results:
            self.projectionCountSlider.setMaxValue(len(results))
            self.projectionCountSlider.setTickInterval(len(results)/10)
        else: self.projectionCountSlider.setMaxValue(1)
        self.updateGraph()

    def updateGraph(self):
        black = QColor(0,0,0)
        white = QColor(255,255,255)
        self.graph.clear()
        self.graph.removeMarkers()
        if self.results == None or self.dialogType not in [VIZRANK, CLUSTER]: return

        attributes = []
        attrDict = {}
        index = 0; projectionsUsed = 0

        best = self.results[0][ACCURACY]
        worst= self.results[min(len(self.results)-1, self.projectionCount)][ACCURACY]
        
        while index < len(self.results):
            if self.dialogType != VIZRANK:
                while index < len(self.results) and type(self.results[index][TRY_INDEX]) != dict: index += 1
                
            if index >= len(self.results): break
            if projectionsUsed >= self.projectionCount: break
            projectionsUsed += 1
            
            attrs = self.results[index][ATTR_LIST]

            if len(attributes) < self.attributeCount:
                for attr in attrs:
                    if attr not in attributes and len(attributes) < self.attributeCount:
                        attributes.append(attr)

            for i in range(len(attrs)):
                for j in range(i+1, len(attrs)):
                    if attrs[i] not in attributes or attrs[j] not in attributes: continue
                    if not attrDict.has_key((attrs[i], attrs[j])) and not attrDict.has_key((attrs[j], attrs[i])):
                        attrDict[(attrs[i], attrs[j])] = self.results[index][ACCURACY]
                        if attrs[i] not in attributes: attributes.append(attrs[i])
                        if attrs[j] not in attributes: attributes.append(attrs[j])
            index += 1
   
        eps = 0.05
        num = len(attributes)
        #for x in range(num-1):
        #    for y in range(num-x-1):
        for x in range(num):
            for y in range(num-x):
                yy = num-y-1
                if not attrDict.has_key((attributes[x], attributes[yy])) and not attrDict.has_key((attributes[yy], attributes[x])): continue

                if attrDict.has_key((attributes[x], attributes[yy])): val = attrDict[(attributes[x], attributes[yy])]
                else: val = attrDict[(attributes[yy], attributes[x])]

                if self.useDarkness:
                    v = 255 - 255*((val-worst)/float(best - worst))
                    color = QColor(v,v,v)
                else: color = black
                
                curve = PolygonCurve(self.graph, QPen(color, 1), QBrush(color))
                key = self.graph.insertCurve(curve)
                self.graph.setCurveData(key, [x+eps, x+1-eps, x+1-eps, x+eps], [y+eps, y+eps, y+1-eps, y+1-eps])

                if not self.onlyLower:
                    curve = PolygonCurve(self.graph, QPen(color, 1), QBrush(color))
                    key = self.graph.insertCurve(curve)
                    self.graph.setCurveData(key, [num-1-y+eps, num-1-y+eps, num-y-eps, num-y-eps], [num-1-x+eps, num-x-eps, num-x-eps, num-1-x+eps] )

        # draw empty boxes at the diagonal
        for x in range(num):
            curve = PolygonCurve(self.graph, QPen(black, 1), QBrush(white))
            key = self.graph.insertCurve(curve)
            self.graph.setCurveData(key, [x+eps, x+1-eps, x+1-eps, x+eps], [num-x-1+eps, num-x-1+eps, num-x-eps, num-x-eps])


        # draw x markers
        for x in range(num):
            marker = MyMarker(self.graph, attributes[x], x + 0.5, -0.3, 90*self.rotateXAttributes)
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
        
        self.graph.update()  # don't know if this is necessary
        self.graph.repaint()
            

class OWGraphAttributeHistogram(OWWidget):
    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Attribute Histogram", wantGraph = 1)

        self.graph = OWGraph(self.mainArea)
        self.results = None
        self.dialogType = -1
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)
        self.box.activate()

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        self.attributeCount = 200
        self.projectionCount = 1000
        self.rotateXAttributes = 1
        self.colorAttributes = 1

        b1 = OWGUI.widgetBox(self.controlArea, box = 1)
        b2 = OWGUI.widgetBox(self.controlArea, 'Number Of Attributes')
        b3 = OWGUI.widgetBox(self.controlArea, 'Number Of Projections')
        b4 = OWGUI.widgetBox(self.controlArea, box = 1)
        box = OWGUI.widgetBox(self.controlArea)

        OWGUI.checkBox(b1, self, 'colorAttributes', label = "Color attributes according to class vote", callback = self.updateGraph)
        OWGUI.hSlider(b2, self, 'attributeCount', minValue=0, maxValue = 2000, step = 10, callback = self.updateGraph, ticks = 50)
        OWGUI.hSlider(b3, self, 'projectionCount', minValue = 0, maxValue = 100000, step=50, callback = self.updateGraph, ticks = 1000)
        OWGUI.checkBox(b4, self, 'rotateXAttributes', label = "Rotate X Labels", callback = self.updateGraph)
        
        b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b2.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b3.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        b4.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        #box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.MinimumExpanding ))

        self.kNNOptimizationDlg = parent
        self.results = None

        self.evaluatedAttributes = None
        self.evaluatedAttributesByClass = None

    def setResults(self, results):
        self.results = results
        self.evaluatedAttributes = None
        self.evaluatedAttributesByClass = None
        self.updateGraph()

    def updateGraph(self):
        black = QColor(0,0,0)
        white = QColor(255,255,255)
        self.graph.clear()
        self.graph.removeMarkers()
        if self.results == None: return
        eps = 0.1

        attrCountDict = {}
        index = 0; projectionsUsed = 0

        for index in range(min(self.projectionCount, len(self.results))):
            attrs = self.results[index][3]
            for attr in attrs:
                if not attrCountDict.has_key(attr):
                    attrCountDict[attr] = 0
                attrCountDict[attr] += 1

        attrs = [(attrCountDict[key], key) for key in attrCountDict.keys()]
        attrs.sort()
        attrs.reverse()
        attrs = attrs[:self.attributeCount]

        if self.colorAttributes and self.evaluatedAttributes == None:
            evalAttrs, attrsByClass = OWVisAttrSelection.findAttributeGroupsForRadviz(self.kNNOptimizationDlg.data, OWVisAttrSelection.S2NMeasureMix())

            classVariableValues = getVariableValuesSorted(self.kNNOptimizationDlg.data, self.kNNOptimizationDlg.data.domain.classVar.name)
            classColors = ColorPaletteHSV(len(classVariableValues))
            self.evaluatedAttributes = evalAttrs
            self.evaluatedAttributesByClass = attrsByClass
        else:
            (evalAttrs, attrsByClass) = (self.evaluatedAttributes, self.evaluatedAttributesByClass)
            classVariableValues = getVariableValuesSorted(self.kNNOptimizationDlg.data, self.kNNOptimizationDlg.data.domain.classVar.name)
            classColors = ColorPaletteHSV(len(classVariableValues))
            

        for (ind, (count, attr)) in enumerate(attrs):
            if self.colorAttributes:
                if attr in evalAttrs:
                    classIndex = evalAttrs.index(attr) % len(classVariableValues)
                    color = classColors[classVariableValues.index(self.kNNOptimizationDlg.data.domain.classVar.values[classIndex])]
                else:
                    color = black
            else:
                color = black
            
            curve = PolygonCurve(self.graph, QPen(color, 1), QBrush(color))
            key = self.graph.insertCurve(curve)
            #print type(ind+eps), type(eps), type(count)
            self.graph.setCurveData(key, [ind+eps, ind + 1 - eps, ind + 1 - eps, ind+eps, ind+eps], [0, 0, count, count, 0])

            # draw attribute names
            y = -attrs[0][0] * 0.03
            if self.rotateXAttributes: marker = MyMarker(self.graph, attr, ind + 0.5, y, 90)
            else: marker = MyMarker(self.graph, attr, ind + 0.5, y, 0)
            mkey = self.graph.insertMarker(marker)
            if self.rotateXAttributes: self.graph.marker(mkey).setLabelAlignment(Qt.AlignLeft+ Qt.AlignVCenter)
            else: self.graph.marker(mkey).setLabelAlignment(Qt.AlignCenter + Qt.AlignBottom)
        
        self.graph.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
        #self.graph.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        self.graph.axisScaleDraw(QwtPlot.xBottom).setTickLength(0, 0, 0)
        #self.graph.axisScaleDraw(QwtPlot.yLeft).setTickLength(0, 0, 0)
        self.graph.axisScaleDraw(QwtPlot.xBottom).setOptions(0) 
        #self.graph.axisScaleDraw(QwtPlot.yLeft).setOptions(0) 
        self.graph.setAxisScale(QwtPlot.xBottom, - 0.5 , len(attrs), 1)
        #self.graph.setAxisScale(QwtPlot.yLeft, - 0.9 - 0.1*len(attrs) , attrs[0][0], 1)

        if self.colorAttributes:
            for i,val in enumerate(classVariableValues):
                self.graph.addCurve(self.kNNOptimizationDlg.data.domain.classVar.name+"="+val, classColors[i], classColors[i], 4, symbol = QwtSymbol.Ellipse, enableLegend = 1)
        
        self.graph.update()  # don't know if this is necessary
        self.graph.repaint()
            
       

# #############################################################################
# draw a graph for all the evaluated projections that shows how is the classification accuracy falling when we are moving from the best to the worst evaluated projections
class OWGraphProjectionQuality(OWWidget):
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Projection Quality", wantGraph = 1)

        self.lineWidth = 1
        OWGUI.comboBox(self.controlArea, self, "lineWidth", box = "Line width", items = range(5), callback = self.updateGraph, sendSelectedValue = 1, valueType = int)        

        self.graph = OWGraph(self.mainArea)
        self.results = None
        self.dialogType = -1
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)
        self.box.activate()

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        self.updateGraph()

    def setResults(self, results, dialogType):
        self.results = results
        self.dialogType = dialogType
        self.updateGraph()

    def updateGraph(self):
        colors = ColorPaletteHSV(2)
        self.graph.clear()
        if self.results == None or self.dialogType not in [VIZRANK, CLUSTER]: return

        yVals = []
        yVals2 = []
        for i in range(len(self.results)):
            if self.dialogType == VIZRANK:
                yVals.append(self.results[i][0])
            else:
                if type(self.results[i][4]) == dict:
                    yVals2.append(self.results[i][0])
                else:
                    yVals.append(self.results[i][0])

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

        #c = QColor(0,0,0)
        c = colors.getColor(0)
        self.graph.addCurve("", c, c, 1, QwtCurve.Lines, QwtSymbol.None, xData = xVals, yData = yVals, lineWidth = self.lineWidth)

        if yVals2 != []:
            c = colors.getColor(1)
            self.graph.addCurve("", c, c, 1, QwtCurve.Lines, QwtSymbol.None, xData = range(len(yVals2)), yData = yVals2, lineWidth = self.lineWidth)

        self.graph.update()  # don't know if this is necessary
        self.graph.repaint()
            

# #############################################################################
# draw a graph for all the evaluated projections that shows how is the classification accuracy falling when we are moving from the best to the worst evaluated projections
class OWGraphIdentifyOutliers(OWWidget):
    settingsList = ["projectionCountList", "projectionCount", "lineWidth"]
    def __init__(self,parent=None, signalManager = None, widget = None, graph = None):
        OWWidget.__init__(self, parent, signalManager, "Outlier Identification", wantGraph = 1, wantStatusBar = 1)

        self.projectionCountList = ["5", "10", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "Other..."]
        self.projectionCount = "100"
        self.selectedExampleIndex = 0
        self.showGraph = 0
        self.lineWidth = 1
        self.showAllClasses = 0
        self.matrixOfPredictions = None
        self.projectionGraph = graph
        self.VizRankDialog = parent
        self.widget = widget
        self.graphMatrix = None
        self.results = None
        self.data = None
        self.dialogType = -1
        self.evaluatedExamples = []

        self.loadSettings()
        
        b1 = OWGUI.widgetBox(self.controlArea, ' Projection Count ')
        self.projectionCountEdit = OWGUI.comboBoxWithCaption(b1, self, "projectionCount", "Number of best projections to consider: ", tooltip = "How many projections do you want to consider when computing probabilities of correct classification?", items = self.projectionCountList, callback = self.projectionCountChanged, sendSelectedValue = 1, valueType = str)
        
        b2 = OWGUI.widgetBox(self.controlArea, ' Show Predictions for All Examples ')
        #OWGUI.button(b2, self, "Update", self.updateProjectionCount)
        self.showPredictionsButton = OWGUI.button(b2, self, "Show predictions in graph", self.toggleShowPredictions); self.showPredictionsButton.setToggleButton(1)
        
        b3 = OWGUI.widgetBox(self.controlArea, ' Show Predictions for Selected Example ')
        self.showGraphCheck = OWGUI.checkBox(b3, self, 'showGraph', 'Show graph of predicted probabilities', tooltip = "Show the graph of probabilities for the selected example over the selected set of top ranked projections", callback = self.showGraphUpdate)
        self.showAllClassesCheck = OWGUI.checkBox(b3, self, 'showAllClasses', 'Show predicted probabilities for all classes', tooltip = "Show predicted probabilities for each class value", callback = self.updateGraph)
        self.selectedExampleCombo = OWGUI.comboBoxWithCaption(b3, self, "selectedExampleIndex", "Index of selected example: ", tooltip = "Select the index of the example whose predictions you wish to analyse in the graph?", callback = self.selectedExampleChanged, sendSelectedValue = 1, valueType = int)
        OWGUI.button(b3, self, "Update index from the graph", self.updateIndexFromGraph)

        b4 = OWGUI.widgetBox(self.controlArea, ' Graph Settings')
        OWGUI.comboBoxWithCaption(b4, self, "lineWidth", 'Line width: ', items = range(1,5), callback = self.updateGraph, sendSelectedValue = 1, valueType = int)

        b5 = OWGUI.widgetBox(self.controlArea, ' Examples by Their Uncertain Predictions ')
        OWGUI.button(b5, self, "Evaluate predictions for all examples", self.evaluateAllExamples)
        self.exampleList = QListBox(b5)
        self.connect(self.exampleList, SIGNAL("selectionChanged()"),self.exampleListSelectionChanged) 
        #self.exampleList.setMinimumSize(100,100)

        #b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        #b2.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        #b3.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))

        #box = OWGUI.widgetBox(self.controlArea, "")
        #box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.MinimumExpanding ))

        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWGraph(self.mainArea)
        self.graph.setXaxisTitle("Top projections")
        self.box.addWidget(self.graph)
        self.box.activate()

        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        self.showGraphUpdate()

    # on escape
    def hideEvent (self, e):
        if self.widget: self.widget.outlierValues = None
        self.widget.updateGraph()
        self.saveSettings()
        QDialog.hideEvent(self, e)
        
    def setData(self, results, data, dialogType):
        self.results = results
        self.data = data
        self.dialogType = dialogType
        self.matrixOfPredictions = None

        # example index combo
        self.selectedExampleCombo.clear()
        if self.data:
            for i in range(len(self.data)):
                self.selectedExampleCombo.insertItem(str(i))

        self.updateGraph()

    def projectionCountChanged(self):
        self.exampleList.clear()
        self.evaluatedExamples = []
        
        if self.projectionCount == "Other...":
            (text, ok) = QInputDialog.getText('Qt Projection count', 'How many of the best projections do you wish to consider?')
            if ok and str(text).isdigit():
                text = str(text)
                if text not in self.projectionCountList:
                    i = 0
                    while i < len(self.projectionCountList)-1 and int(self.projectionCountList[i]) < int(text): i+=1
                    self.projectionCountList.insert(i, text)
                    self.projectionCountEdit.insertItem(text, i)
                self.projectionCount = text
            else:
                self.projectionCount = "100"
        self.toggleShowPredictions()    # update changes


    def evaluateProjections(self):
        # compute predictions
        self.widget.progressBarInit()

        projCount = min(int(self.projectionCount), len(self.results))
        classCount = len(self.data.domain.classVar.values)
        existing = 0
        if self.matrixOfPredictions:
            existing = Numeric.shape(self.matrixOfPredictions)[0]/classCount
            if existing < projCount:
                self.matrixOfPredictions = Numeric.resize(self.matrixOfPredictions, (projCount*classCount, len(self.data)))
            elif existing > projCount:
                self.matrixOfPredictions = self.matrixOfPredictions[0:classCount*projCount,:]
                
        else:
            self.matrixOfPredictions = -1 * Numeric.ones((projCount*classCount, len(self.data)), Numeric.Float)


        # compute the matrix of predictions
        results = self.results[existing:min(len(self.results),projCount)]
        index = 0
        for (acc, other, tableLen, attrList, tryIndex, generalDict) in results:
            attrIndices = [self.projectionGraph.attributeNameIndex[attr] for attr in attrList]
            validDataIndices = self.projectionGraph.getValidIndices(attrIndices)
            table = self.projectionGraph.createProjectionAsExampleTable(attrIndices)    # TO DO: this does not work with polyviz!!!
            acc, probabilities = self.VizRankDialog.kNNClassifyData(table)

            #self.matrixOfPredictions[(existing + index)*classCount:(existing + index +1)*classCount] = Numeric.transpose(probabilities)
            probabilities = Numeric.transpose(probabilities)
            for i in range(classCount):
                Numeric.put(self.matrixOfPredictions[(existing + index)*classCount + i], validDataIndices, probabilities[i])            

            index += 1
            self.statusBar.message("Evaluated %s/%s projections..." % (OWVisFuncts.createStringFromNumber(existing + index), projCount))
            self.widget.progressBarSet(100.0*(index)/float(projCount-existing))

        self.widget.progressBarFinished()


    def toggleShowPredictions(self):
        if not self.widget: return
        if self.showPredictionsButton.isOn():
            self.evaluateProjections()
            
            self.statusBar.message("Computing averages...")

            projCount = min(int(self.projectionCount), len(self.results))
            classCount = len(self.data.domain.classVar.values)
    
            # compute the average probability of correct classification over the selected number of top projections
            values = [0.0 for i in range(len(self.data))]
            for i in range(len(self.data)):
                corrClass = int(self.data[i].getclass())
                predictions = self.matrixOfPredictions[corrClass::classCount,i]
                predictions = Numeric.compress(predictions != -1, predictions)
                if len(predictions):    # prevent division by zero!
                    values[i] = Numeric.sum(predictions) / float(len(predictions))

            self.widget.outlierValues = (values, "Probability of correct class value = %.2f%%")
        else:
            self.widget.outlierValues = None

        self.widget.updateGraph()
            
        self.statusBar.message("")
        self.widget.showSelectedAttributes()


    def showGraphUpdate(self):
        if self.showGraph:
            self.evaluateProjections()
            self.graph.show()
            self.resize(self.controlArea.size().width() + 400, self.size().height())
            self.selectedExampleChanged()
        else:
            self.graph.hide()
            self.resize(self.controlArea.size().width(), self.size().height())
        

    def selectedExampleChanged(self):
        if self.showGraph and self.results:
            projCount = min(int(self.projectionCount), len(self.results))
            classCount = len(self.data.domain.classVar.values)
            self.graphMatrix = Numeric.transpose(Numeric.reshape(self.matrixOfPredictions[:, self.selectedExampleIndex], (projCount, classCount)))
            self.updateGraph()
        

    def updateIndexFromGraph(self):
        if self.VizRankDialog.parentName == "Polyviz":
            selected, unselected = self.projectionGraph.getSelectionsAsIndices(self.widget.getShownAttributeList(), self.widget.attributeReverse)
        else:            
            selected, unselected = self.projectionGraph.getSelectionsAsIndices(self.widget.getShownAttributeList())

        if len(selected) != 1:
            QMessageBox.information( None, "Outlier Identification", 'Exactly one example must be selected in the graph in order to complete this operation.', QMessageBox.Ok + QMessageBox.Default)
            return
        self.selectedExampleIndex = selected[0]
        self.selectedExampleChanged()

    def evaluateAllExamples(self):
        if not self.data: return

        self.evaluateProjections()

        projCount = min(int(self.projectionCount), len(self.results))
        classCount = len(self.data.domain.classVar.values)

        for i in range(len(self.data)):
            matrix = Numeric.transpose(Numeric.reshape(self.matrixOfPredictions[:, i], (projCount, classCount)))
            valid = Numeric.where(matrix[int(self.data[i].getclass())] != -1, 1, 0)
            data = Numeric.compress(valid, matrix[int(self.data[i].getclass())])
            ave_acc = Numeric.sum(data) / float(len(data))
            self.insertItemToExampleList(ave_acc, i)
            
            
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


    def updateGraph(self):
        self.graph.clear()
        if not self.data or not self.graphMatrix: return
        
        classColors = ColorPaletteHSV(len(self.data.domain.classVar.values))
        
        if self.graphMatrix == None: return

        self.graph.setAxisScale(QwtPlot.xBottom, 0, len(self.graphMatrix[0]), len(self.graphMatrix[0])/5)
        self.graph.setAxisScale(QwtPlot.yLeft, 0, 1, 0.2)

        valid = Numeric.where(self.graphMatrix[0] != -1, 1, 0)
        allValid = Numeric.sum(valid) == len(valid)

        for i in range(len(self.data.domain.classVar.values)):
            if not self.showAllClasses and int(self.data[self.selectedExampleIndex].getclass()) != i: continue

            c = classColors.getColor(i)
            yData = self.graphMatrix[i].tolist()
           
            if allValid:
                self.graph.addCurve(self.data.domain.classVar.values[i], c, c, 1, QwtCurve.Lines, QwtSymbol.None, enableLegend = 1, xData = range(len(yData)), yData = yData, lineWidth = self.lineWidth)
            else:
                self.graph.addCurve(self.data.domain.classVar.values[i], c, c, 1, QwtCurve.Lines, QwtSymbol.Ellipse, enableLegend = 1, xData = [], yData = [], lineWidth = self.lineWidth)
                currData = []
                xData = []
                for j in range(len(yData)):
                    if yData[j] != -1:
                        currData.append(yData[j])
                        xData.append(j)
                    else:
                        if currData != []:
                            self.graph.addCurve("",  c, c, 1, QwtCurve.Lines, QwtSymbol.Ellipse, enableLegend = 1, xData = xData, yData = currData, lineWidth = self.lineWidth)
                            xData = []
                            currData = []

        self.graph.update()  # don't know if this is necessary
        self.graph.repaint()
   


#test widget appearance
if __name__=="__main__":
    import sys
    a=QApplication(sys.argv)
    ow=OWVizRank()
    #ow = OWInteractionAnalysis()
    #ow = OWGraphAttributeHistogram()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
    