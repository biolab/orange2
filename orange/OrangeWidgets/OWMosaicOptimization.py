from OWBaseWidget import *
from OWWidget import OWWidget
import os, orange, orngTest, time, Numeric, math, orngCI
import OWGUI, OWVisAttrSelection, OWVisFuncts
import random

discMeasures = [("None", None), ("ReliefF", orange.MeasureAttribute_relief(k=10, m=50)), ("Gain ratio", orange.MeasureAttribute_gainRatio()), ("Gini index", orange.MeasureAttribute_gini())]


SCORE = 0
ATTR_LIST = 1
TRY_INDEX = 2


CHI_SQUARE = 0
DIFFERENCE = 1
MAX_DIFFERENCE = 2
GAIN_RATIO = 3
INFORMATION_GAIN = 4
INTERACTION_GAIN = 5

RELATIVE = 0
LAPLACE = 1
M_ESTIMATE = 2


class MosaicOptimization(OWBaseWidget):
    resultsListLenNums = [ 100 ,  250 ,  500 ,  1000 ,  5000 ,  10000, 20000, 50000, 100000, 500000 ]
    resultsListLenList = [str(x) for x in resultsListLenNums]
    settingsList = ["attrDisc", "showScore", "showRank", "qualityMeasure", "resultListLen", "percentDataUsed",
                    "evaluationTimeIndex", "argumentCountIndex", "VizRankClassifierName", "mValue", "probabilityEstimationIndex"]

    percentDataNums = [ 5 ,  10 ,  15 ,  20 ,  30 ,  40 ,  50 ,  60 ,  70 ,  80 ,  90 ,  100 ]
    evaluationTimeNums = [0.5, 1, 2, 5, 10, 20, 30, 40, 60, 80, 120]
    evaluationTimeList = [str(x) for x in evaluationTimeNums]
    argumentCounts = range(101)[1:]
    
    def __init__(self, parentWidget = None, signalManager = None):
        OWBaseWidget.__init__(self, None, signalManager, "Mosaic Optimization Dialog")

        self.setCaption("Qt Mosaic Optimization Dialog")
        self.controlArea = QVBoxLayout(self)

        # loaded variables
        self.parentWidget = parentWidget
        self.showRank = 0
        self.showScore = 1
        self.attrDisc = 1
        self.qualityMeasure = 1
        self.resultListLen = 1000
        self.attributeCount = 2
        self.optimizationType = 0
        self.percentDataUsed = 100
        self.evaluationTimeIndex = 3
        self.argumentCountIndex = 4
        self.mValue = 2.0
        self.probabilityEstimationIndex = 2
        self.VizRankClassifierName = "Mosaic Classifier"

        self.aprioriDistribution = None
        self.lastSaveDirName = os.getcwd()
        self.selectedClasses = []
        self.cancelOptimization = 0
        self.data = None
        self.evaluatedAttributes = None   # save last evaluated attributes
        self.allResults = []
        self.shownResults = []
        self.attrLenDict = {}
        self.maxResultListLen = self.resultsListLenNums[len(self.resultsListLenNums)-1]
        
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

        self.label1 = QLabel('Projections with ', self.buttonBox)
        self.optimizationTypeCombo = OWGUI.comboBox(self.buttonBox, self, "optimizationType", items = ["    exactly    ", "  maximum  "] )
        self.attributeCountCombo = OWGUI.comboBox(self.buttonBox, self, "attributeCount", items = range(1, 5), tooltip = "Evaluate only projections with exactly (or maximum) this number of attributes", sendSelectedValue = 1, valueType = int)
        self.attributeLabel = QLabel(' attributes', self.buttonBox)

        self.startOptimizationButton = OWGUI.button(self.optimizationBox, self, "Start Evaluating Projections", callback = self.startProjectionEvaluation)
        f = self.startOptimizationButton.font(); f.setBold(1);   self.startOptimizationButton.setFont(f)
        self.stopOptimizationButton = OWGUI.button(self.optimizationBox, self, "Stop evaluation", callback = self.stopOptimizationClick)
        self.stopOptimizationButton.setFont(f)
        self.stopOptimizationButton.hide()
##        self.optimizeGivenProjectionButton = OWGUI.button(self.optimizationBox, self, "Optimize current projection")
##        self.optimizeGivenProjectionButton.hide()

        self.resultList = QListBox(self.resultsBox)
        self.resultList.setMinimumSize(200,200)
        self.connect(self.resultList, SIGNAL("selectionChanged()"), self.showSelectedAttributes) 

        self.showRankCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showRank', 'Rank', callback = self.updateShownProjections, tooltip = "Show projection ranks")
        self.showScoreCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showScore', 'Score', callback = self.updateShownProjections, tooltip = "Show projection score")

        # ##########################
        # SETTINGS TAB
        self.measureCombo = OWGUI.comboBox(self.SettingsTab, self, "qualityMeasure", box = " Measure Projection Interestingness ", items = ["Sum of Standardized Pearson Residuals", "Sum of differences between expected and actual counts", "Maximum of differences between expected and actual counts", "Gain Ratio", "Information Gain", "Interaction Gain"], tooltip = "What is interesting?")

        self.optimizationSettingsBox = OWGUI.widgetBox(self.SettingsTab, " VizRank Evaluation Settings ")
        self.percentDataUsedCombo= OWGUI.comboBoxWithCaption(self.optimizationSettingsBox, self, "percentDataUsed", "Percent of data used in evaluation: ", items = self.percentDataNums, sendSelectedValue = 1, valueType = int)
        
        #self.localOptimizationSettingsBox = OWGUI.widgetBox(self.SettingsTab, " Local Optimization Settings ")
        self.miscSettingsBox = OWGUI.widgetBox(self.SettingsTab, " Length of the Projection List ")
        
        OWGUI.comboBox(self.SettingsTab, self, "attrDisc", box = " Measure for Ranking Attributes ", items = [val for (val, m) in discMeasures], callback = self.removeEvaluatedAttributes)

        self.resultListCombo = OWGUI.comboBoxWithCaption(self.miscSettingsBox, self, "resultListLen", "Number of projections to show in projection list:   ", tooltip = "Maximum length of the list of interesting projections. This is also the number of projections that will be saved if you click Save button.", items = self.resultsListLenNums, callback = self.updateShownProjections, sendSelectedValue = 1, valueType = int)


        # ##########################
        # ARGUMENTATION TAB
        self.argumentationBox = OWGUI.widgetBox(self.ArgumentationTab, " Arguments ")
        self.findArgumentsButton = OWGUI.button(self.argumentationBox, self, "Find Arguments", callback = self.findArguments)
        f = self.findArgumentsButton.font(); f.setBold(1);  self.findArgumentsButton.setFont(f)
        self.stopArgumentationButton = OWGUI.button(self.argumentationBox, self, "Stop Searching", callback = self.stopArgumentationClick)
        self.stopArgumentationButton.setFont(f)
        self.stopArgumentationButton.hide()
        self.classValueList = OWGUI.comboBox(self.ArgumentationTab, self, "argumentationClassValue", box = " Arguments For Class: ", tooltip = "Select the class value that you wish to see arguments for", callback = self.argumentationClassChanged)
        self.argumentBox = OWGUI.widgetBox(self.ArgumentationTab, " Arguments for The Selected Class Value ")
        self.argumentList = QListBox(self.argumentBox)
        self.argumentList.setMinimumSize(200,200)
        self.connect(self.argumentList, SIGNAL("selectionChanged()"),self.argumentSelected)

        # ##########################
        # CLASSIFICATION TAB
        self.classifierNameEdit = OWGUI.lineEdit(self.ClassificationTab, self, 'VizRankClassifierName', box = ' Learner / Classifier Name ', tooltip='Name to be used by other widgets to identify your learner/classifier.')

        #self.argumentValueFormulaIndex = OWGUI.comboBox(self.ClassificationTab, self, "argumentValueFormula", box="Argument Value is Computed As ...", items=["1.0 x Projection Value", "0.5 x Projection Value + 0.5 x Predicted Example Probability", "1.0 x Predicted Example Probability"], tooltip=None)
        probBox = OWGUI.widgetBox(self.ClassificationTab, box = " Probability Estimation ")
        self.probCombo = OWGUI.comboBox(probBox, self, "probabilityEstimationIndex", items = ["Relative Frequency", "Laplace", "m-Estimate"])

        mValid = QDoubleValidator(self)
        mValid.setRange(0,10000,1)
        self.mEditBox = OWGUI.lineEdit(probBox, self, 'mValue', label='Parameter for m-estimate:   ', orientation='horizontal', valueType = str, validator = mValid)

        b = OWGUI.widgetBox(self.ClassificationTab, " Evaluating Time ")
        self.evaluationTimeEdit = OWGUI.comboBoxWithCaption(b, self, "evaluationTimeIndex", "Time for evaluating projections (minutes):   ", tooltip = "What is the maximum time that the classifier is allowed for evaluating projections (learning)", items = self.evaluationTimeList)
        b2 = OWGUI.widgetBox(b, orientation = "horizontal")
        projCountBox = OWGUI.widgetBox(self.ClassificationTab, " Projection Count ")
        self.argumentCountEdit = OWGUI.comboBoxWithCaption(projCountBox, self, "argumentCountIndex", "Number of projections used when classifying:                ", tooltip = "What is the maximum number of projections (arguments) that will be used when classifying an example.", items = [str(x) for x in self.argumentCounts])

        # ##########################
        # SAVE & MANAGE TAB
        self.classesBox = OWGUI.widgetBox(self.ManageTab, " Select Class Values You Wish to Consider ")
        self.visualizedAttributesBox = OWGUI.widgetBox(self.ManageTab, " Number of Concurrently Visualized Attributes ")
        #self.dialogsBox = OWGUI.widgetBox(self.ManageTab, " Dialogs ")        
        self.manageResultsBox = OWGUI.widgetBox(self.ManageTab, " Manage Projections ")        
        
        self.classesList = QListBox(self.classesBox)
        self.classesList.setSelectionMode(QListBox.Multi)
        self.classesList.setMinimumSize(60,60)
        self.connect(self.classesList, SIGNAL("selectionChanged()"), self.classesListChanged)
        
        self.attrLenList = QListBox(self.visualizedAttributesBox)
        self.attrLenList.setSelectionMode(QListBox.Multi)
        self.attrLenList.setMinimumSize(60,60)
        self.connect(self.attrLenList, SIGNAL("selectionChanged()"), self.attrLenListChanged)

        self.buttonBox4 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.clearButton = OWGUI.button(self.buttonBox4, self, "Reevaluate visualizations", self.reevaluateProjections)
        
        self.buttonBox6 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.loadButton = OWGUI.button(self.buttonBox6, self, "Load", self.load)
        self.saveButton = OWGUI.button(self.buttonBox6, self, "Save", self.save)

        self.buttonBox5 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        self.clearButton = OWGUI.button(self.buttonBox5, self, "Clear results", self.clearResults)
            
        # ###########################
        self.statusBar = QStatusBar(self)
        self.controlArea.addWidget(self.statusBar)
        self.controlArea.activate()

        self.resize(375,550)
        self.setMinimumWidth(375)
        self.tabs.setMinimumWidth(375)
        random.seed()

        
    # ##############################################################
    # EVENTS
    # ##############################################################
    def showSelectedAttributes(self, attrs = None):
        if not self.parentWidget: return

        if not attrs: (score, attrs, index) = self.getSelectedProjection()
        self.parentWidget.setShownAttributes(attrs)
    
        
    def removeEvaluatedAttributes(self):
        self.evaluatedAttributes = None

    # result list can contain projections with different number of attributes
    # user clicked in the listbox that shows possible number of attributes of result list
    # result list must be updated accordingly
    def attrLenListChanged(self):
        # check which attribute lengths do we want to show
        self.attrLenDict = {}
        for i in range(self.attrLenList.count()):
            self.attrLenDict[int(str(self.attrLenList.text(i)))] = self.attrLenList.isSelected(i)
        self.updateShownProjections()

    def classesListChanged(self):
        results = self.allResults
        self.clearResults()
        
        self.selectedClasses = self.getSelectedClassValues()
        if len(self.selectedClasses) in [self.classesList.count(), 0]:
            for result in results:
                self.addResult(result[SCORE], result[ATTR_LIST], result[TRY_INDEX])
        else: 
            for result in results:
                if self.attrLenDict[len(result[ATTR_LIST])] == 1:
                    self.addResult(result[SCORE], result[ATTR_LIST], result[TRY_INDEX])
        self.finishedAddingResults()

    def clearResults(self):
        del self.allResults; self.allResults = []
        del self.shownResults; self.shownResults = []
        self.resultList.clear()
        self.attrLenDict = {}
        self.attrLenList.clear()

    # ##############################################################
    # ##############################################################

    def getSelectedClassValues(self):
        selectedClasses = []
        for i in range(self.classesList.count()):
            if self.classesList.isSelected(i): selectedClasses.append(i)
        return selectedClasses


    def updateShownProjections(self, *args):
        self.resultList.clear()
        self.shownResults = []
        i = 0
        qApp.setOverrideCursor(QWidget.waitCursor)

        while self.resultList.count() < self.resultListLen and i < len(self.allResults):
            if self.attrLenDict.has_key(len(self.allResults[i][ATTR_LIST])) and self.attrLenDict[len(self.allResults[i][ATTR_LIST])] == 1:
                string = ""
                if self.showRank: string += str(i+1) + ". "
                if self.showScore: string += "%.2f : " % (self.allResults[i][SCORE])
                string += self.buildAttrString(self.allResults[i][ATTR_LIST])
                
                self.resultList.insertItem(string)
                self.shownResults.append(self.allResults[i])
            i+=1
        qApp.processEvents()
        qApp.restoreOverrideCursor()
        
        if self.resultList.count() > 0: self.resultList.setCurrentItem(0)        


    def setData(self, data):
        self.setStatusBarText("")
        self.classValueList.clear()
        self.data = None
        
        if not data: return
        
        if hasattr(data, "name"): self.datasetName = data.name
        else: self.datasetName = ""

        # take only discrete attributes
        discAttrs = []
        for attr in data.domain:
            if attr.varType == orange.VarTypes.Discrete: discAttrs.append(attr.name)
        
        self.data = data.select(discAttrs)

        self.attributeNameIndex = dict([(self.data.domain[i].name, i) for i in range(len(self.data.domain))])
        
        self.evaluatedAttributes = None
        self.aprioriDistribution = None
        self.clearResults()
        
        self.classesList.clear()
        self.selectedClasses = []

        if not data or not (data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete): return

        # add class values
        for i in range(len(data.domain.classVar.values)):
            self.classesList.insertItem(data.domain.classVar.values[i])
            self.classValueList.insertItem(data.domain.classVar.values[i])
            
        if len(data.domain.classVar.values) > 0: self.classValueList.setCurrentItem(0)
        self.classesList.selectAll(1)

    # get only the data examples that belong to one of the selected class values
    def getData(self):
        if self.data and self.data.domain.classVar and self.data.domain.classVar.varType == orange.VarTypes.Discrete:
            return self.data.select({self.data.domain.classVar.name: [self.data.domain.classVar.values[i] for i in self.selectedClasses]})
        else: return self.data
        

    # given a dataset return a list of attributes where attribute are sorted by their decreasing importance for class discrimination
    def getEvaluatedAttributes(self, data):
        if not data.domain.classVar or data.domain.classVar.varType != orange.VarTypes.Discrete:
            QMessageBox.information( None, "Mosaic Dialog", 'In order to be able to find interesing projections the data set has to have a discrete class.', QMessageBox.Ok + QMessageBox.Default)
            return []
            
        if self.evaluatedAttributes: return self.evaluatedAttributes
        
        self.setStatusBarText("Evaluating attributes...")
        qApp.setOverrideCursor(QWidget.waitCursor)

        try:
            # evaluate attributes using the selected attribute measure
            self.evaluatedAttributes = OWVisAttrSelection.evaluateAttributes(data, None, discMeasures[self.attrDisc][1])
        except:
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # print the exception
            
        self.setStatusBarText("")
        qApp.restoreOverrideCursor()

        if self.evaluatedAttributes == None: return []
        else:   return self.evaluatedAttributes

        
    def startProjectionEvaluation(self):
        if not self.data: return

        self.clearResults()
        self.disableControls()
        self.cancelOptimization = 0
        
        hasMissingData = (len(self.data) != len(orange.Preprocessor_dropMissing(self.data)))
        if self.optimizationType == 0: maxLength = self.attributeCount; minLength = self.attributeCount
        else:                          maxLength = self.attributeCount; minLength = 1

        data = self.getData()   # get only the examples that have one of the class values that is selected in the class value list
        if not data:
            QMessageBox.critical(None,'No data','There is no data or no class value is selected in the Manage tab.',QMessageBox.Ok)
            return
            
        if self.percentDataUsed != 100:
            indices = orange.MakeRandomIndices2(data, 1.0-float(self.percentDataUsed)/100.0)
            data = data.select(indices)

        evaluatedAttrs = self.getEvaluatedAttributes(data)
        if evaluatedAttrs == []: return
        
        self.aprioriDistribution = orange.Distribution(data.domain.classVar.name, data)
        #attributes = [self.attributeNameIndex[name] for name in evaluatedAttrs]
        classIndex = self.attributeNameIndex[data.domain.classVar.name]

        self.parentWidget.progressBarInit()
        startTime = time.time()
        triedPossibilities = 0; totalPossibilities = 0
        if self.optimizationType == 0: totalPossibilities = OWVisFuncts.combinationsCount(self.attributeCount, len(evaluatedAttrs))
        else:
            for i in range(1, self.attributeCount): totalPossibilities += OWVisFuncts.combinationsCount(i, len(evaluatedAttrs))

        for z in range(len(evaluatedAttrs)):
            for u in range(minLength-1, maxLength):
                combinations = OWVisFuncts.combinations(evaluatedAttrs[:z], u)
                
                for attrList in combinations:
                    attrs = [evaluatedAttrs[z]] + attrList

                    val = self._Evaluate(data, attrs)

                    if self.isOptimizationCanceled():
                        secs = time.time() - startTime
                        self.setStatusBarText("Evaluation stopped (evaluated %s projections in %d min, %d sec)" % (OWVisFuncts.createStringFromNumber(triedPossibilities), secs/60, secs%60))
                        self.parentWidget.progressBarFinished()
                        self.finishedAddingResults()
                        self.enableControls()
                        return

                    self.addResult(val, attrs, triedPossibilities)
                    
                    triedPossibilities += 1
                    qApp.processEvents()        # allow processing of other events
                            
                    self.parentWidget.progressBarSet(100.0*triedPossibilities/float(totalPossibilities))
                    self.setStatusBarText("Evaluated %s visualizations..." % (OWVisFuncts.createStringFromNumber(triedPossibilities)))

                del combinations
                
        secs = time.time() - startTime
        self.setStatusBarText("Finished evaluation (evaluated %s visualization in %d min, %d sec)" % (OWVisFuncts.createStringFromNumber(triedPossibilities), secs/60, secs%60))
        self.parentWidget.progressBarFinished()
        self.finishedAddingResults()
        self.enableControls()


    def _Evaluate(self, data, attrs):
        newFeature, quality = orngCI.FeatureByCartesianProduct(data, attrs)
        
        if self.qualityMeasure == GAIN_RATIO:
            return orange.MeasureAttribute_gainRatio(newFeature, data)
        elif self.qualityMeasure == INFORMATION_GAIN:
            return orange.MeasureAttribute_info(newFeature, data)
        elif self.qualityMeasure == INTERACTION_GAIN:
            new = orange.MeasureAttribute_info(newFeature, data)
            gains = [orange.MeasureAttribute_info(attr, data) for attr in attrs]
            return new - sum(gains)
        else:
            aprioriSum = sum(self.aprioriDistribution)
            val = 0.0

            for dist in orange.ContingencyAttrClass(newFeature, data):
                for i in range(len(self.aprioriDistribution)):
                    expected = float(len(data) * self.aprioriDistribution.values()[i]) / float(aprioriSum)
                    
                    if self.qualityMeasure == CHI_SQUARE and expected:
                        val += (dist[i] - expected)**2 / expected
                    elif self.qualityMeasure == DIFFERENCE:
                        val += abs(expected-dist[i])
                    elif self.qualityMeasure == MAX_DIFFERENCE:
                        val = max(val, abs(expected-dist[i]))

            return val

    def getProjectionQuality(self, data, attrList):
        if not self.aprioriDistribution:
            self.aprioriDistribution = orange.Distribution(data.domain.classVar.name, data)

        return self._Evaluate(data, attrList)

    # reevaluate projections in the list using the current settings
    def reevaluateProjections(self):
        results = self.shownResults
        self.clearResults()

        self.parentWidget.progressBarInit()
        self.disableControls()

        if self.percentDataUsed != 100:
            indices = orange.MakeRandomIndices2(self.data, 1.0-float(self.percentDataUsed)/100.0)
            data = self.data.select(indices)
        else: data = self.data

        self.aprioriDistribution = orange.Distribution(data.domain.classVar.name, data)

        testIndex = 0
        strTotal = OWVisFuncts.createStringFromNumber(len(results))
        for (score, attrList, tryIndex) in results:
            if self.isOptimizationCanceled(): break
            testIndex += 1
            self.parentWidget.progressBarSet(100.0*testIndex/float(len(results)))

            newScore = self.getProjectionQuality(data, attrList)

            self.addResult(newScore, attrList, testIndex)
            self.setStatusBarText("Reevaluated %s/%s projections..." % (OWVisFuncts.createStringFromNumber(testIndex), strTotal))

        self.setStatusBarText("")
        self.parentWidget.progressBarFinished()
        self.enableControls()
        self.finishedAddingResults()
    
    
    
    def addResult(self, score, attrList, tryIndex):
        self.insertItem(score, attrList, self.findTargetIndex(score, max), tryIndex)
        qApp.processEvents()        # allow processing of other events

    # use bisection to find correct index
    def findTargetIndex(self, score, funct):
        top = 0; bottom = len(self.allResults)

        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if funct(score, self.allResults[mid][SCORE]) == score: bottom = mid
            else: top = mid

        if len(self.allResults) == 0: return 0
        if funct(score, self.allResults[top][SCORE]) == score:
            return top
        else: 
            return bottom

    # insert new result - give parameters: score of projection, number of examples in projection and list of attributes.
    # parameter attrReverseList can be a list used by polyviz
    def insertItem(self, score, attrList, index, tryIndex):
        if index < self.maxResultListLen:
            self.allResults.insert(index, (score, attrList, tryIndex))
        if index < self.resultListLen:
            string = ""
            if self.showRank: string += str(index+1) + ". "
            if self.showScore: string += "%.2f : " % (score)

            string += self.buildAttrString(attrList)

            self.resultList.insertItem(string, index)
            self.shownResults.insert(index, (score, attrList, tryIndex))

        # remove worst projection if list is too long
        if self.resultList.count() > self.resultListLen:
            self.resultList.removeItem(self.resultList.count()-1)
            self.shownResults.pop()
    
    def finishedAddingResults(self):
        self.cancelOptimization = 0
        
        self.attrLenList.clear()
        self.attrLenDict = {}
        for i in range(len(self.allResults)):
            self.attrLenDict[len(self.allResults[i][ATTR_LIST])] = 1

        for i in range(1,5):
            if self.attrLenDict.has_key(i):
                self.attrLenList.insertItem(str(i))

        self.attrLenList.selectAll(1)
        self.resultList.setCurrentItem(0)

   
    # ##############################################################
    # Loading and saving projection files
    # ##############################################################

    # save the list into a file - filename can be set if you want to call this function without showing the dialog
    def save(self, filename = None):
        if filename == None:
            # get file name
            if self.datasetName != "":
                filename = "%s - %s" % (os.path.splitext(os.path.split(self.datasetName)[1])[0], self.parentName)
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

        # open, write and save file
        dict = {}
        for attr in ["attrDisc", "qualityMeasure", "resultListLen", "percentDataUsed"]: dict[attr] = self.__dict__[attr]
        dict["dataCheckSum"] = self.data.checksum()

        file = open(name, "wt")        
        file.write("%s\n" % (str(dict)))
        file.write("%s\n" % str(self.selectedClasses))
        for (score, attrList, tryIndex) in self.shownResults:
            file.write("(%.3f, %s, %d\n" % (score, attrList, tryIndex))
        file.flush()
        file.close()
        self.setStatusBarText("Saved %d visualizations" % (len(self.shownResults)))


    # load projections from a file
    def load(self, name = None, ignoreCheckSum = 0):
        self.clearResults()
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

        file = open(name, "rt")
        settings = eval(file.readline()[:-1])

        if not ignoreCheckSum and settings.has_key("dataCheckSum") and settings["dataCheckSum"] != self.data.checksum():
            if QMessageBox.information(self, 'VizRank', 'The current data set has a different checksum than the data set that was used to evaluate visualizations in this file.\nDo you want to continue loading anyway, or cancel?','Continue','Cancel', '', 0,1):
                file.close()
                return

        self.setSettings(settings)

        ind = 0
        for line in file.xreadlines():
            (score, attrList, tryIndex) = eval(line)
            self.insertItem(score, attrList, ind, tryIndex)
            ind+=1
        file.close()

        # update loaded results
        self.finishedAddingResults()
        self.setStatusBarText("Loaded %d visualizations" % (ind))


    # disable all controls while evaluating projections
    def disableControls(self):
        self.startOptimizationButton.hide()
        self.stopOptimizationButton.show()
        self.resultsDetailsBox.setEnabled(0)
        self.SettingsTab.setEnabled(0)
        self.ManageTab.setEnabled(0)
        
    def enableControls(self):
        self.startOptimizationButton.show()
        self.stopOptimizationButton.hide()
        self.resultsDetailsBox.setEnabled(1)
        self.SettingsTab.setEnabled(1)
        self.ManageTab.setEnabled(1)

    # ######################################################
    # Auxiliary functions
    # ######################################################
    
    # from a list of attributes build a nice string with attribute names
    def buildAttrString(self, attrList):
        if len(attrList) == 0: return ""
        
        strList = attrList[0]
        for attr in attrList[1:]:
            strList += ", " + attr
        return strList


    def getSelectedProjection(self):
        if self.resultList.count() == 0: return None
        return self.shownResults[self.resultList.currentItem()]

    def stopOptimizationClick(self):
        self.cancelOptimization = 1

    def isOptimizationCanceled(self):
        return self.cancelOptimization

    def destroy(self, dw = 1, dsw = 1):
        self.saveSettings()

    def setStatusBarText(self, text):
        self.statusBar.message(text)
        qApp.processEvents()

           
    # ######################################################
    # Argumentation functions
    # ######################################################
    def findArguments(self, selectBest = 1, showClassification = 1, example = None):
        self.cancelArgumentation = 0

        self.argumentList.clear()
        self.arguments = [[] for i in range(self.classValueList.count())]
                
        if not example and not self.parentWidget.subsetData:
            QMessageBox.information( None, "Argumentation", 'To find arguments you first have to provide an example that you wish to classify. \nYou can do this by sending the example to the Mosaic display widget through the "Example Subset" signal.', QMessageBox.Ok + QMessageBox.Default)
            return None, None
        if len(self.shownResults) == 0:
            QMessageBox.information( None, "Argumentation", 'To find arguments you first have to evaluate some projections by clicking "Start evaluating projections" in the Main tab.', QMessageBox.Ok + QMessageBox.Default)
            return None, None
        
        data = self.getData()   # get only the examples that have one of the class values that is selected in the class value list
        if not data:
            QMessageBox.critical(None,'No data','There is no data or no class value is selected in the Manage tab.',QMessageBox.Ok)
            return None, None

        if example == None: example = self.parentWidget.subsetData[0]
        
        self.findArgumentsButton.hide()
        self.stopArgumentationButton.show()

        if self.percentDataUsed != 100:
            indices = orange.MakeRandomIndices2(data, 1.0-float(self.percentDataUsed)/100.0)
            data = data.select(indices)

        self.aprioriDistribution = orange.Distribution(data.domain.classVar.name, data)
        currentClassValue = self.classValueList.currentItem()

        for index in range(min(len(self.shownResults), self.argumentCounts[self.argumentCountIndex])):       # use only best argumentCount projections for argumentation
            if self.cancelArgumentation: break          # user pressed cancel
            
            qApp.processEvents()
            (accuracy, attrList, tryIndex) = self.allResults[index]

            attrVals = [example[attr] for attr in attrList]
            if "?" in attrVals:
                self.printVerbose("Missing value in attribute list %s. Projection not used in prediction." % (attrList))
                continue  # the testExample has a missing value at one of the visualized attributes

            d = orange.Preprocessor_take(data, values = dict([(data.domain[attr], example[attr]) for attr in attrList]))
            
            vals = self.getArguments(d)

            for i in range(len(vals)):
                pos = self.getArgumentIndex(vals[i], i)
                self.arguments[i].insert(pos, (vals[i], accuracy, attrList, index))
                if i == currentClassValue:
                    self.argumentList.insertItem("%.3f - %s" %(vals[i], attrList), pos)

        predictions = []
        for i in range(len(self.aprioriDistribution)):
            val = self.aprioriDistribution[i]
            for (v, a, l, ind) in self.arguments[i]: val *= v
            predictions.append(val)

        # return a randomly selected class value
        if sum(predictions) == 0:
            s = "Predicted probabilities for all class values are zero. Try using a different measure for probability estimation."
            self.setStatusBarText(s)
            self.printVerbose(s)
            i = random.randint(0, len(self.aprioriDistribution)-1)
            arr = [0] * len(self.aprioriDistribution);  arr[i] = 1
            dist = orange.DiscDistribution(arr); dist.variable = self.data.domain.classVar
            return self.data.domain.classVar[i], dist


        # find the most probable class value and return it with its probability
        ind = predictions.index(max(predictions))
        classValue = self.data.domain.classVar[ind]
        prob = predictions[ind] / sum(predictions)
        dist = orange.DiscDistribution([val/float(sum(predictions)) for val in predictions])
        dist.variable = self.data.domain.classVar

        self.stopArgumentationButton.hide()
        self.findArgumentsButton.show()
        if self.argumentList.count() > 0 and selectBest: self.argumentList.setCurrentItem(0)

        s = '<nobr>Based on the projections, the example would be classified </nobr><br><nobr>to class <b>%s</b> with probability <b>%.2f%%</b>.</nobr><br><nobr>Predicted class distribution is:</nobr><br>' % (str(classValue), prob*100)
        for key in dist.keys():
            s += "<nobr>&nbsp &nbsp &nbsp &nbsp %s : %.2f%%</nobr><br>" % (key, dist[key]*100)

        if showClassification:
            QMessageBox.information(None, "Classification results", s, QMessageBox.Ok + QMessageBox.Default)

        """
        if not example[example.domain.classVar.name].isSpecial() and example.getclass().value != classValue:
            self.show()
            QMessageBox.information(None, "Classification results", s, QMessageBox.Ok + QMessageBox.Default)
            while self.isVisible():
                qApp.processEvents()
        """
        
        return (classValue, dist)

    def getConditionalProbability(self, data, index, distribution = None):
        aprioriSum = sum(self.aprioriDistribution)
        if not distribution:
            distribution = orange.Distribution(data.domain.classVar.name, data)

        if self.probabilityEstimationIndex == RELATIVE:
            if not (len(data) and self.aprioriDistribution[index]):
                self.printVerbose("empty data subset. Unable to compute relative frequency.")
                return 0.0      # prevent division by zero
            return (distribution[index] * aprioriSum) / float(len(data) * self.aprioriDistribution[index])      # P(c_i | a_k) / P(c_i)
        elif self.probabilityEstimationIndex == LAPLACE:
            return ((distribution[index]+1) * aprioriSum) / float((len(data)+len(distribution)) * self.aprioriDistribution[index])      # (r+1 / n+c) / P(c_i)
        elif self.probabilityEstimationIndex == M_ESTIMATE:
            n = distribution[index]
            pa = self.aprioriDistribution[index]/float(sum(self.aprioriDistribution))
            return (pa * self.mValue + n) / float(sum(distribution) + self.mValue)       # p = (pa*m+n)/(N+m)
            

    def getArguments(self, data):
        actualDistribution = orange.Distribution(data.domain.classVar.name, data)
        aprioriSum = sum(self.aprioriDistribution)
        arguments = [self.getConditionalProbability(data, i, actualDistribution) for i in range(len(self.aprioriDistribution))]
        return arguments
    

    def getArgumentIndex(self, value, classValue):
        if len(self.arguments[classValue]) == 0: return 0
        
        top = 0; bottom = len(self.arguments[classValue])
        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if max(value, self.arguments[classValue][mid][0]) == value: bottom = mid
            else: top = mid

        if max(value, self.arguments[classValue][top][0]) == value:  return top
        else:                                                        return bottom
        
    def stopArgumentationClick(self):
        self.cancelArgumentation = 1
    
    def argumentationClassChanged(self):
        self.argumentList.clear()
        if len(self.arguments) == 0: return
        ind = self.classValueList.currentItem()
        for i in range(len(self.arguments[ind])):
            (val, accuracy, attrList, index) = self.arguments[ind][i]
            self.argumentList.insertItem("%.2f - %s" %(val, attrList), i)
            

    def argumentSelected(self):
        ind = self.argumentList.currentItem()
        classInd = self.classValueList.currentItem()
        self.showSelectedAttributes(self.arguments[classInd][ind][2])
        

# #############################################################################
# class that represents kNN classifier that classifies examples based on top evaluated projections
class MosaicVizRankClassifier(orange.Classifier):
    def __init__(self, VizRankDlg, data):
        self.VizRankDlg = VizRankDlg

        self.VizRankDlg.parentWidget.subsetdata(None)
        self.VizRankDlg.parentWidget.cdata(data)

        self.evaluating = 1
        t = QTimer(self.VizRankDlg.parentWidget)
        self.VizRankDlg.connect(t, SIGNAL("timeout()"), self.VizRankDlg.stopOptimizationClick)
        t.start(self.VizRankDlg.evaluationTimeNums[self.VizRankDlg.evaluationTimeIndex] * 60 * 1000)

        self.VizRankDlg.startProjectionEvaluation()
        t.stop()
        self.VizRankDlg.printVerbose("computing %d" % (len(data)))


    # for a given example run argumentation and find out to which class it most often fall        
    def __call__(self, example, returnType):
        table = orange.ExampleTable(example.domain)
        table.append(example)
        self.VizRankDlg.parentWidget.subsetdata(table)
                
        classVal, prob = self.VizRankDlg.findArguments(0, 0, example)

        if returnType == orange.GetBoth: return classVal, prob
        else:                            return classVal
        

# #############################################################################
# learner that builds VizRankClassifier
class MosaicVizRankLearner(orange.Learner):
    def __init__(self, VizRankDlg):
        self.VizRankDlg = VizRankDlg
        self.name = self.VizRankDlg.VizRankClassifierName
        
        
    def __call__(self, examples, weightID = 0):
        return MosaicVizRankClassifier(self.VizRankDlg, examples)




#data = orange.ExampleTable(r"E:\Development\Python23\Lib\site-packages\Orange\Datasets\jure\klopi_zeimelLB.tab")
#apri = orange.Distribution(data.domain.classVar.name, data)
#getDifference(data, apri, ["IGG1"])

#test widget appearance
if __name__=="__main__":
    import sys
    a=QApplication(sys.argv)
    ow=MosaicOptimization()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
