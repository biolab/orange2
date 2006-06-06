from OWBaseWidget import *
from OWWidget import OWWidget
import os
import OWGUI, orngVisFuncts
from orngMosaic import *
from orngScaleData import getVariableValuesSorted

mosaicMeasures = [("Pearson's Chi Square", CHI_SQUARE),("Cramer's Phi (Correlation With Class)", CRAMERS_PHI),("Information Gain (In Percents Of Class Entropy Removed)", INFORMATION_GAIN), ("Gain Ratio", GAIN_RATIO), ("Interaction Gain (In Percents Of Class Entropy Removed)", INTERACTION_GAIN),("Average Probability Of Correct Classification", AVERAGE_PROBABILITY_OF_CORRECT_CLASSIFICATION)]

class OWMosaicOptimization(OWBaseWidget, orngMosaic):
    resultsListLenNums = [ 100 ,  250 ,  500 ,  1000 ,  5000 ,  10000, 20000, 50000, 100000, 500000 ]
    resultsListLenList = [str(x) for x in resultsListLenNums]
    settingsList = ["attrDisc", "qualityMeasure", "percentDataUsed", "ignoreTooSmallCells",
                    "evaluationTime", "VizRankClassifierName", "mValue", "probabilityEstimation", "attributeCount",
                    "optimizeAttributeOrder", "optimizeAttributeValueOrder"]

    percentDataNums = [ 5 ,  10 ,  15 ,  20 ,  30 ,  40 ,  50 ,  60 ,  70 ,  80 ,  90 ,  100 ]
    evaluationTimeNums = [0.5, 1, 2, 5, 10, 20, 30, 40, 60, 80, 120]
    
    def __init__(self, parentWidget = None, signalManager = None):
        OWBaseWidget.__init__(self, None, signalManager, "Mosaic Optimization Dialog")
        orngMosaic.__init__(self)

        self.setCaption("Qt Mosaic Optimization Dialog")
        self.controlArea = QVBoxLayout(self)

        # loaded variables
        self.parentWidget = parentWidget
        self.showConfidence = 1
        self.optimizeAttributeOrder = 0
        self.optimizeAttributeValueOrder = 0
        self.VizRankClassifierName = "Mosaic Learner"
        self.useOnlyRelevantInteractionsInArgumentation = 0 # unused variable present in old ini files
        
        
        self.lastSaveDirName = os.getcwd()
        self.selectedClasses = []
        self.cancelOptimization = 0
        self.cancelArgumentation = 0
        
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
        self.optimizeOrderBox = OWGUI.widgetBox(self.MainTab, " Attribute and Value Order ")
        self.buttonBox = OWGUI.widgetBox(self.optimizationBox, orientation = "horizontal")

        self.label1 = QLabel('Projections with ', self.buttonBox)
        self.optimizationTypeCombo = OWGUI.comboBox(self.buttonBox, self, "optimizationType", items = ["    exactly    ", "  maximum  "] )
        self.attributeCountCombo = OWGUI.comboBox(self.buttonBox, self, "attributeCount", items = range(1, 5), tooltip = "Evaluate only projections with exactly (or maximum) this number of attributes", sendSelectedValue = 1, valueType = int)
        self.attributeLabel = QLabel(' attributes', self.buttonBox)

        self.startOptimizationButton = OWGUI.button(self.optimizationBox, self, "Start Evaluating Projections", callback = self.evaluateProjections)
        f = self.startOptimizationButton.font(); f.setBold(1);   self.startOptimizationButton.setFont(f)
        self.stopOptimizationButton = OWGUI.button(self.optimizationBox, self, "Stop evaluation", callback = self.stopOptimizationClick)
        self.stopOptimizationButton.setFont(f)
        self.stopOptimizationButton.hide()

        self.resultList = QListBox(self.resultsBox)
        self.resultList.setMinimumSize(200,200)
        self.connect(self.resultList, SIGNAL("selectionChanged()"), self.showSelectedAttributes) 

        OWGUI.checkBox(self.optimizeOrderBox, self, "optimizeAttributeOrder", "Optimize order of attributes", callback = self.optimizeCurrentAttributeOrder, tooltip = "Order the visualized attributes so that it will enhance class separation")
        OWGUI.checkBox(self.optimizeOrderBox, self, "optimizeAttributeValueOrder", "Optimize order of attribute values", callback = self.optimizeCurrentAttributeOrder, tooltip = "Order also the values of visualized attributes so that it will enhance class separation.\nWARNING: This can take a lot of time when visualizing attributes with many values.")
        self.optimizeOrderButton = OWGUI.button(self.optimizeOrderBox, self, "Optimize Current Order", callback = self.optimizeCurrentAttributeOrder, tooltip = "Optimize the order of currently visualized attributes")

        # ##########################
        # SETTINGS TAB
        self.measureCombo = OWGUI.comboBox(self.SettingsTab, self, "qualityMeasure", box = " Measure Projection Interestingness ", items = [item[0] for item in mosaicMeasures], tooltip = "What is interesting?", callback = self.updateGUI)

        #self.optimizationSettingsBox = OWGUI.widgetBox(self.SettingsTab, " VizRank Evaluation Settings ")
        #self.percentDataUsedCombo= OWGUI.comboBoxWithCaption(self.optimizationSettingsBox, self, "percentDataUsed", "Percent of data used in evaluation: ", items = self.percentDataNums, sendSelectedValue = 1, valueType = int)

        self.ignoreSmallCellsBox = OWGUI.widgetBox(self.SettingsTab, " " )
        self.ignoreSmallCellsCombo = OWGUI.checkBox(self.ignoreSmallCellsBox, self, "ignoreTooSmallCells", "Ignore cells where expected number of cases is less than 5", tooltip = "Statisticians advise that in cases when the number of expected examples is less than 5 we ignore the cell \nsince it can significantly influence the chi-square value.")
        
        self.testingBox = OWGUI.widgetBox(self.SettingsTab, " Testing Method ")
        self.testingCombo = OWGUI.comboBox(self.testingBox, self, "testingMethod", items = ["10 fold cross validation", "70/30 separation 10 times "], tooltip = "Method for evaluating the class separation in the projection.")
        
        OWGUI.comboBox(self.SettingsTab, self, "attrDisc", box = " Measure for Ranking Attributes ", items = [val for (val, m) in discMeasures], callback = self.removeEvaluatedAttributes)

        # ##########################
        # ARGUMENTATION TAB
        self.argumentationBox = OWGUI.widgetBox(self.ArgumentationTab, " Arguments ")
        self.findArgumentsButton = OWGUI.button(self.argumentationBox, self, "Find Arguments", callback = self.findArguments, tooltip = "Evaluate arguments for each possible class value using settings in the Classification tab.")
        f = self.findArgumentsButton.font(); f.setBold(1);  self.findArgumentsButton.setFont(f)
        self.stopArgumentationButton = OWGUI.button(self.argumentationBox, self, "Stop Searching", callback = self.stopArgumentationClick)
        self.stopArgumentationButton.setFont(f)
        self.stopArgumentationButton.hide()

        self.argumentsClassBox = OWGUI.widgetBox(self.ArgumentationTab, " Show Arguments For Class: ", orientation = "horizontal")
        self.classValueList = OWGUI.comboBox(self.argumentsClassBox, self, "argumentationClassValue", tooltip = "Select the class value that you wish to see arguments for", callback = self.updateShownArguments)
        self.logitLabel = OWGUI.widgetLabel(self.argumentsClassBox, " ", labelWidth = 100)

        self.argumentBox = OWGUI.widgetBox(self.ArgumentationTab, " Arguments/Odds Ratios For The Selected Class Value ")
        self.argumentList = QListBox(self.argumentBox)
        self.argumentList.setMinimumSize(200,200)
        self.connect(self.argumentList, SIGNAL("selectionChanged()"),self.argumentSelected)
        self.resultsDetailsBox = OWGUI.widgetBox(self.ArgumentationTab, " Shown Details in Arguments List " , orientation = "horizontal")
        self.showConfidenceCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showConfidence', '95% Confidence Interval', callback = self.updateShownArguments, tooltip = "Show confidence interval of the argument.")

        # ##########################
        # CLASSIFICATION TAB
        self.classifierNameEdit = OWGUI.lineEdit(self.ClassificationTab, self, 'VizRankClassifierName', box = ' Learner / Classifier Name ', tooltip='Name to be used by other widgets to identify your learner/classifier.')

        #self.argumentValueFormulaIndex = OWGUI.comboBox(self.ClassificationTab, self, "argumentValueFormula", box="Argument Value is Computed As ...", items=["1.0 x Projection Value", "0.5 x Projection Value + 0.5 x Predicted Example Probability", "1.0 x Predicted Example Probability"], tooltip=None)
        probBox = OWGUI.widgetBox(self.ClassificationTab, box = " Probability Estimation ")
        self.probCombo = OWGUI.comboBox(probBox, self, "probabilityEstimation", items = ["Relative Frequency", "Laplace", "m-Estimate"], callback = self.updateMestimateComboState)

        mValid = QDoubleValidator(self)
        mValid.setRange(0,10000,1)
        self.mEditBox = OWGUI.lineEdit(probBox, self, 'mValue', label='              Parameter for m-estimate:   ', orientation='horizontal', valueType = float, validator = mValid)

        b = OWGUI.widgetBox(self.ClassificationTab, " Evaluating Time ")
        self.evaluationTimeEdit = OWGUI.comboBoxWithCaption(b, self, "evaluationTime", "Maximum time for evaluating projections (minutes):  ", tooltip = "What is the maximum time that the classifier is allowed for evaluating projections (learning)", items = self.evaluationTimeNums, sendSelectedValue = 1, valueType = float)
        b2 = OWGUI.widgetBox(b, orientation = "horizontal")
        projCountBox = OWGUI.widgetBox(self.ClassificationTab, " Class Prediction Settings ")
        #OWGUI.checkBox(projCountBox, self, 'useOnlyRelevantInteractionsInArgumentation', 'Ignore arguments without relevant interactions', tooltip = "Consider only relevant interactions. Use Kononenko's criterion \nto find interacting attribute values (Igor Kononenko: Semi-naive Bayesian Classifier)")
        OWGUI.checkBox(projCountBox, self, 'automaticallyRemoveWeakerArguments', 'Remove weaker arguments')
        OWGUI.button(self.ClassificationTab, self, "Apply Changes", callback = self.resendLearner)

        # ##########################
        # SAVE & MANAGE TAB
        self.visualizedAttributesBox = OWGUI.widgetBox(self.ManageTab, " Number of Concurrently Visualized Attributes ")
        #self.dialogsBox = OWGUI.widgetBox(self.ManageTab, " Dialogs ")        
        self.manageResultsBox = OWGUI.widgetBox(self.ManageTab, " Manage Projections ")        
        
        self.attrLenList = QListBox(self.visualizedAttributesBox)
        self.attrLenList.setSelectionMode(QListBox.Multi)
        self.attrLenList.setMinimumSize(60,60)
        self.connect(self.attrLenList, SIGNAL("selectionChanged()"), self.attrLenListChanged)

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

        self.updateMestimateComboState()
        self.updateGUI()

    # ##############################################################    
    # EVENTS
    def showSelectedAttributes(self, attrs = None):
        if not self.parentWidget: return
        if not attrs:
            projection = self.getSelectedProjection()
            if not projection: return
            (score, attrs, index) = projection
        valueOrder = None
        if self.optimizeAttributeOrder:
            self.resultList.setEnabled(0)
            self.optimizeCurrentAttributeOrder(attrs)
            self.resultList.setEnabled(1)
        else:
            self.parentWidget.setShownAttributes(attrs)
        self.resultList.setFocus()
            

    def optimizeCurrentAttributeOrder(self, attrs = None, updateGraph = 1):
        if str(self.optimizeOrderButton.text()) == "Optimize Current Order":
            self.stopOptimization = 0
            self.optimizeOrderButton.setText("Stop optimization")

            if not attrs:
                attrs = self.parentWidget.getShownAttributes()
                
            bestPlacements = self.findOptimalAttributeOrder(attrs, self.optimizeAttributeValueOrder)
            if updateGraph:
                self.parentWidget.bestPlacements = bestPlacements
                if bestPlacements:
                    attrList, valueOrder = bestPlacements[0][1], bestPlacements[0][2]
                    self.parentWidget.setShownAttributes(attrList, customValueOrderDict = dict([(attrList[i], tuple(valueOrder[i])) for i in range(len(attrList))]) )
            
            self.optimizeOrderButton.setText("Optimize Current Order")
            return bestPlacements
        else:
            self.stopOptimization = 1
            return []
        

    def updateGUI(self):
        if self.qualityMeasure in [CHI_SQUARE, CRAMERS_PHI]: self.ignoreSmallCellsBox.show()
        else:                                                self.ignoreSmallCellsBox.hide()
        if self.qualityMeasure == AVERAGE_PROBABILITY_OF_CORRECT_CLASSIFICATION: self.testingBox.show()
        else:   self.testingBox.hide()

    def updateMestimateComboState(self):
        self.mEditBox.setEnabled(self.probabilityEstimation == M_ESTIMATE)
        
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

        for i in range(len(self.results)):
            if self.attrLenDict.has_key(len(self.results[i][ATTR_LIST])) and self.attrLenDict[len(self.results[i][ATTR_LIST])] == 1:
                self.resultList.insertItem("%.2f : %s" % (self.results[i][SCORE], self.buildAttrString(self.results[i][ATTR_LIST])))
                self.resultListIndices.append(i)
        qApp.processEvents()
        
        if self.resultList.count() > 0: self.resultList.setCurrentItem(0)


    def setData(self, data):
        orngMosaic.setData(self, data)
        self.setStatusBarText("")
        self.classValueList.clear()
        self.argumentList.clear()
        self.selectedClasses = []
                
        if not data: return
        
        if hasattr(data, "name"): self.datasetName = data.name
        else: self.datasetName = ""

        if not data or not (data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete): return

        # add class values
        for val in getVariableValuesSorted(data, data.domain.classVar.name):
            self.classValueList.insertItem(val)
        self.updateShownArguments()
        
        if len(data.domain.classVar.values) > 0: self.classValueList.setCurrentItem(0)


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
            self.evaluatedAttributes = orngVisFuncts.evaluateAttributes(data, None, discMeasures[self.attrDisc][1])
        except:
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # print the exception
            
        self.setStatusBarText("")
        qApp.restoreOverrideCursor()

        if self.evaluatedAttributes == None: return []
        else:   return self.evaluatedAttributes


    # ######################################################
    # Argumentation functions
    def findArguments(self, example = None, selectBest = 1, showClassification = 1):
        self.cancelArgumentation = 0

        self.argumentList.clear()
        self.arguments = [[] for i in range(self.classValueList.count())]
                
        if not example and not self.parentWidget.subsetData:
            QMessageBox.information( None, "Argumentation", 'To find arguments you first have to provide an example that you wish to classify. \nYou can do this by sending the example to the Mosaic display widget through the "Example Subset" signal.', QMessageBox.Ok + QMessageBox.Default)
            return None, None
        if len(self.results) == 0:
            QMessageBox.information( None, "Argumentation", 'To find arguments you first have to evaluate some projections by clicking "Start evaluating projections" in the Main tab.', QMessageBox.Ok + QMessageBox.Default)
            return None, None
        
        if not self.data:
            QMessageBox.critical(None,'No data','There is no data or no class value is selected in the Manage tab.',QMessageBox.Ok)
            return None, None

        if example == None: example = self.parentWidget.subsetData[0]
        
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
            s = '<nobr>Based on the projections, the example would be classified </nobr><br><nobr>to class <b>%s</b> with probability <b>%.2f%%</b>.</nobr><br><nobr>Predicted class distribution is:</nobr><br>' % (str(classValue), max(dist)*100. / float(sum(dist)))
            for key in values:
                s += "<nobr>&nbsp &nbsp &nbsp &nbsp %s : %.2f%%</nobr><br>" % (key, dist[key]*100)
            QMessageBox.information(None, "Classification results", s, QMessageBox.Ok + QMessageBox.Default)
        
        return (classValue, dist)

   
    def finishedAddingResults(self):
        self.cancelOptimization = 0
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

        # open, write and save file
        dict = {}
        for attr in ["attrDisc", "qualityMeasure", "percentDataUsed"]: dict[attr] = self.__dict__[attr]
        dict["dataCheckSum"] = self.data.checksum()

        file = open(name, "wt")        
        file.write("%s\n" % (str(dict)))
        for (score, attrList, tryIndex) in self.results:
            file.write("(%.4f, %s, %d)\n" % (score, attrList, tryIndex))
        file.flush()
        file.close()
        self.setStatusBarText("Saved %d visualizations" % (len(self.results)))


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
        self.SettingsTab.setEnabled(0)
        self.ManageTab.setEnabled(0)
        
    def enableControls(self):
        self.startOptimizationButton.show()
        self.stopOptimizationButton.hide()
        self.SettingsTab.setEnabled(1)
        self.ManageTab.setEnabled(1)

    # ######################################################
    # Auxiliary functions
    # ######################################################
    
    def getSelectedProjection(self):
        currentItem = self.resultList.currentItem()
        if self.resultList.count() == 0 or currentItem == -1:
            return None
        return self.results[self.resultListIndices[currentItem]]      # we have to look into resultListIndices, since perhaps not all projections from the self.results are shown

    def stopOptimizationClick(self):
        self.cancelOptimization = 1

    def isOptimizationCanceled(self):
        if hasattr(self, "useTimeLimit"): return orngMosaic.isOptimizationCanceled(self)
        else:                             return self.cancelOptimization
        
    def destroy(self, dw = 1, dsw = 1):
        self.saveSettings()

    def setStatusBarText(self, text):
        self.statusBar.message(text)
        qApp.processEvents()

    def insertArgument(self, argScore, error, attrList, index):
        s = "%.3f " % argScore
        if self.showConfidence: s += "+-%.2f " % error
        s += "- " + self.buildAttrString(attrList)
        self.argumentList.insertItem(s, index)
           
    def updateShownArguments(self):
        self.argumentList.clear()
        if len(self.arguments) == 0: return
        classVal = str(self.classValueList.currentText())
        
        self.logitLabel.setText("log odds = %.2f" % self.logits.get(classVal, -1))
        self.logitLabel.show()

        if not self.arguments.has_key(classVal): return
        for i in range(len(self.arguments[classVal])):
            (argScore, accuracy, attrList, index, error) = self.arguments[classVal][i]
            self.insertArgument(argScore, error, attrList, i)
            

    def argumentSelected(self):
        ind = self.argumentList.currentItem()
        classVal = str(self.classValueList.currentText())
        self.showSelectedAttributes(self.arguments[classVal][ind][2])


    def resendLearner(self):
        self.parentWidget.send("Learner", self.parentWidget.VizRankLearner)

    def stopArgumentationClick(self):
        self.cancelArgumentation = 1

#test widget appearance
if __name__=="__main__":
    import sys
    a = QApplication(sys.argv)
    ow = OWMosaicOptimization()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
