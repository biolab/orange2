from OWBaseWidget import *
from qt import *
from qwt import *
import sys
import cPickle
import os
import orange
import orngTest
import orngStat
from copy import copy


CLASS_ACCURACY = 0
AVERAGE_CORRECT = 1
BRIER_SCORE = 2

class kNNOptimization(OWBaseWidget):
    settingsList = ["resultListLen", "percentDataUsed", "kValue", "minExamples", "qualityMeasure", "useHeuristics", "bestSubsets", "onlyOnePerSubset", "useLeaveOneOut", "lastSaveDirName"]
    resultsListLenList = ['10', '20', '50', '100', '150', '200', '250', '300', '400', '500', '700', '1000', '2000', '4000', '8000']
    resultsListLenNums = [ 10 ,  20 ,  50 ,  100 ,  150 ,  200 ,  250 ,  300 ,  400 ,  500 ,  700 ,  1000 ,  2000 ,  4000,   8000 ]
    percentDataList = ['5', '10', '15', '20', '30', '40', '50', '60', '70', '80', '90', '100']
    percentDataNums = [ 5 ,  10 ,  15 ,  20 ,  30 ,  40 ,  50 ,  60 ,  70 ,  80 ,  90 ,  100 ]
    kNeighboursList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '15', '17', '20', '25', '30', '40', '60', '80', '100', '150', '200']
    kNeighboursNums = [ 0 ,  1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7 ,  8 ,  9 ,  10 ,  12 ,  15 ,  17 ,  20 ,  25 ,  30 ,  40 ,  60 ,  80 ,  100 ,  150 ,  200 ]

    def __init__(self,parent=None):
        #QWidget.__init__(self, parent)
        OWBaseWidget.__init__(self, parent, "Optimization Dialog", "optimize visualization impression and manage result", FALSE, FALSE, FALSE)

        self.setCaption("Qt VizRank Optimization Dialog")
        self.topLayout = QVBoxLayout( self, 10 ) 
        self.grid=QGridLayout(4,2)
        self.topLayout.addLayout( self.grid, 10 )

        self.currentSubset = 0  # used in FSS
        self.totalSubsets = 0   # used in FSS
        self.kValue = 1
        self.minExamples = 0
        self.resultListLen = 100
        self.percentDataUsed = 100
        self.bestSubsets = 100
        self.qualityMeasure = 0
        self.useLeaveOneOut = 1
        self.widgetDir = os.path.realpath(os.path.dirname(__file__)) + "/"
        self.parentName = "Projection"
        self.useHeuristics = 0
        self.onlyOnePerSubset = 0
        self.lastSaveDirName = os.getcwd() + "/"
                
        self.optimizedListFull = []
        self.optimizedListFiltered = []
        self.attrLenDict = {}

        self.loadSettings()
        
        self.optimizeButtonBox =QVGroupBox("Optimize toolbox", self)
        self.manageResultsBox = QVGroupBox ("Manage projections", self)

        self.evaluateBox = QVGroupBox("Evaluate projection / classifier ", self)
        self.resultsBox = QVGroupBox ("List of interesting projections", self)

        self.grid.addWidget(self.optimizeButtonBox,0,0)
        self.grid.addWidget(self.evaluateBox,1,0)
        self.grid.addWidget(self.manageResultsBox,2,0)
        #self.grid.addWidget(self.infoBox, 2,0)
        self.grid.addMultiCellWidget (self.resultsBox,0,2, 1, 1)
        self.grid.setColStretch(0, 0)
        self.grid.setColStretch(1, 100)
        self.grid.setRowStretch(0, 0)
        self.grid.setRowStretch(1, 0)
        self.grid.setRowStretch(2, 100)
                
        self.interestingList = QListBox(self.resultsBox)
        #self.interestingList.setSelectionMode(QListBox.Extended)   # this would be nice if could be enabled, but it has a bug - currentItem doesn't return the correct value if this is on
        self.interestingList.setMinimumSize(200,200)

        self.hbox1 = QHBox(self.optimizeButtonBox)
        self.attrOrdLabel = QLabel('Number of neighbours (k): ', self.hbox1)
        self.attrKNeighbour = QComboBox(self.hbox1)

        self.hbox2 = QHBox(self.optimizeButtonBox)
        self.resultListLabel = QLabel('Number of most interesting projections: ', self.hbox2)
        self.resultListCombo = QComboBox(self.hbox2)
        
        self.hbox3 = QHBox(self.optimizeButtonBox)
        self.minTableLenLabel = QLabel('Minimum examples in data set:         ', self.hbox3)
        self.minTableLenEdit = QLineEdit(self.hbox3)
        self.hbox4 = QHBox (self.optimizeButtonBox)
        self.percentDataUsedLabel = QLabel('Percent of data used in evaluation:  ', self.hbox4)
        self.percentDataUsedCombo = QComboBox(self.hbox4)

        self.useLeaveOneOutCB = QCheckBox("Test using Leave one out (slower)", self.optimizeButtonBox)
        self.useLeaveOneOutCB.hide()
        self.connect(self.useLeaveOneOutCB, SIGNAL("clicked()"), self.setUseLeaveOneOut)
        self.useLeaveOneOutCB.setChecked(self.useLeaveOneOut)

        self.measureBox = QHButtonGroup(" Quality measure ", self.optimizeButtonBox)
        self.measureCombo = QComboBox(self.measureBox)
        self.measureCombo.insertItem("Classification accuracy")
        self.measureCombo.insertItem("Average probability of correct classification")
        self.measureCombo.insertItem("Brier score")
        self.connect(self.measureCombo, SIGNAL("activated(int)"), self.setQualityMeasure)
        
        self.numberOfAttrBox = QVGroupBox ("Find interesting projections", self.optimizeButtonBox)
    
        self.hbox5 = QHBox(self.numberOfAttrBox)
        self.optimizeSeparationButton = QPushButton(' Optimize for exactly  ', self.hbox5)
        self.exactlyLenCombo = QComboBox(self.hbox5)    # maximum number of attributes in subset
        self.exactlyAttrLabel = QLabel(' attributes', self.hbox5)
        
        self.hbox6 = QHBox(self.numberOfAttrBox)
        self.optimizeAllSubsetSeparationButton = QPushButton('Optimize for maximum', self.hbox6)
        self.maxLenCombo = QComboBox(self.hbox6)    # maximum number of attributes in subset
        self.exactlyAttrLabel2 = QLabel(' attributes', self.hbox6)

        self.hbox11 = QHBox(self.numberOfAttrBox)
        self.hbox11.hide()
        #self.useHeuristicsCB = QCheckBox("Test only best ", self.hbox11)
        #self.useHeuristicsCB.hide()
        #self.connect(self.useHeuristicsCB, SIGNAL("clicked()"), self.setUseHeuristics)
        #self.useHeuristicsCB.setChecked(self.useHeuristics)
        #self.numberOfBestSubsetsEdit = QLineEdit(self.hbox11)
        #self.numberOfBestSubsetsEdit.hide()
        #self.numberOfBestSubsetsEdit.setMaximumWidth(40)
        #self.numberOfBestSubsetsLabel = QLabel(' feature subsets (FSS)', self.hbox11)

        #self.onlyOnePerSubsetCB = QCheckBox("Save only one projection per attribute subset", self.numberOfAttrBox)
        #self.onlyOnePerSubsetCB.hide()
        #self.onlyOnePerSubsetCB.setChecked(self.onlyOnePerSubset)
        #self.connect(self.onlyOnePerSubsetCB, SIGNAL("clicked()"), self.setOnlyOnePerSubset)
        
        self.exactlyLenCombo.insertItem("ALL")
        self.maxLenCombo.insertItem("ALL")
        
        for i in range(3, 15):
            self.exactlyLenCombo.insertItem(str(i))
            self.maxLenCombo.insertItem(str(i))
        self.maxLenCombo.setCurrentItem(0)
        self.exactlyLenCombo.setCurrentItem(0)

        self.hbox10 = QHBox(self.evaluateBox)
        self.evaluateProjectionButton = QPushButton("Evaluate projection", self.hbox10)
        self.saveProjectionButton = QPushButton("Save projection", self.hbox10)

        self.hbox7 = QHBox(self.evaluateBox)
        self.showKNNCorrectButton = QPushButton('kNN correct', self.hbox7)
        self.showKNNWrongButton = QPushButton('kNN wrong', self.hbox7)
        self.showKNNResetButton = QPushButton('Original', self.hbox7) 
                
        #self.resize(200, 500)
        self.attrLenCaption = QLabel('Number of concurrently visualized attributes:', self.manageResultsBox)
        self.attrLenList = QListBox(self.manageResultsBox)
        self.attrLenList.setSelectionMode(QListBox.Multi)
        self.attrLenList.setMinimumSize(60,60)

        self.reevaluateResults = QPushButton("Reevaluate results with different k values", self.manageResultsBox)
        self.filterButton = QPushButton("Remove attribute", self.manageResultsBox)
        self.removeSelectedButton = QPushButton("Remove selected projections", self.manageResultsBox)
        self.hbox8 = QHBox(self.manageResultsBox)
        self.hbox9 = QHBox(self.manageResultsBox)
        self.loadButton = QPushButton("Load", self.hbox8)
        self.saveButton = QPushButton("Save", self.hbox8)
        self.clearButton = QPushButton("Clear results", self.hbox9)
        self.closeButton = QPushButton("Close", self.hbox9)

        self.connect(self.attrKNeighbour, SIGNAL("activated(int)"), self.setKNeighbours)
        self.connect(self.resultListCombo, SIGNAL("activated(int)"), self.setResultListLen)
        self.connect(self.percentDataUsedCombo, SIGNAL("activated(int)"), self.setPercentDataUsed)
        self.connect(self.minTableLenEdit, SIGNAL("textChanged(const QString &)"), self.setMinTableLen)
        #self.connect(self.numberOfBestSubsetsEdit, SIGNAL("textChanged(const QString &)"), self.setBestSubsetsEdit)
        self.connect(self.attrLenList, SIGNAL("selectionChanged()"), self.attrLenListChanged)
        self.connect(self.filterButton, SIGNAL("clicked()"), self.filter)
        self.connect(self.removeSelectedButton, SIGNAL("clicked()"), self.removeSelected)
        self.connect(self.saveButton, SIGNAL("clicked()"), self.save)
        self.connect(self.loadButton, SIGNAL("clicked()"), self.load)
        self.connect(self.clearButton, SIGNAL("clicked()"), self.clear)
        self.connect(self.closeButton, SIGNAL("clicked()"), self.hide)

        #self.optimizeButtonBox.setMinimumSize(180,150)
        #self.manageResultsBox.setMinimumSize(180,150)
        self.activateLoadedSettings()


    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        for item in self.resultsListLenList:
            self.resultListCombo.insertItem(item)
        self.resultListCombo.setCurrentItem(self.resultsListLenNums.index(self.resultListLen))

        for val in self.percentDataList:
            self.percentDataUsedCombo.insertItem(val)
        self.percentDataUsedCombo.setCurrentItem(self.percentDataNums.index(self.percentDataUsed))
        
        for i in range(len(self.kNeighboursList)):
            self.attrKNeighbour.insertItem(self.kNeighboursList[i])
        self.attrKNeighbour.setCurrentItem(self.kNeighboursNums.index(self.kValue))

        self.measureCombo.setCurrentItem(self.qualityMeasure)

        self.minTableLenEdit.setText(str(self.minExamples))
        #self.numberOfBestSubsetsEdit.setText(str(self.bestSubsets))
        #self.useHeuristicsCB.setChecked(self.useHeuristics)

    def destroy(self, dw, dsw):
        self.saveSettings()
        #OWBaseWidget.destroy(self, dw, dsw)

    def setQualityMeasure(self, n):
        self.qualityMeasure = n

    def getQualityMeasure(self):
        return self.qualityMeasure

    def setKNeighbours(self, n):
        self.kValue = self.kNeighboursNums[n]

    def setUseHeuristics(self):
        self.useHeuristics = self.useHeuristicsCB.isChecked()

    def setUseLeaveOneOut(self):
        self.useLeaveOneOut = self.useLeaveOneOutCB.isChecked()

    def setOnlyOnePerSubset(self):
        self.onlyOnePerSubset = self.onlyOnePerSubsetCB.isChecked()
        if self.onlyOnePerSubset and self.optimizedListFull != []:
            res = QMessageBox.information(self,'Question','Do you want to remove projections with the same attribute list?','Yes','No', QString.null,0,1)
            if res == 1: return
            full = self.optimizedListFull
            self.clear()
            while full != []:
                (accuracy, tableLen, list, strList) = full[0]

                # try to find a list of attributes in the rest of list that matches this list
                found = 0; i = 0
                while i < len(self.optimizedListFull) and not found:
                    (acc2, tab2, list2, strList2) = self.optimizedListFull[i]
                    i += 1
                    if len(list) != len(list2): continue
                    missingAttrs = 0
                    for attr in list:
                        if attr not in list2: missingAttrs += 1
                    if missingAttrs == 0: found = 1
                full.remove((accuracy, tableLen, list, strList))
                if not found:
                    self.insertItem(accuracy, tableLen, list, strList)
            self.updateNewResults()
            
    # set the length of the list of best projections
    def setResultListLen(self, n):
        self.resultListLen = self.resultsListLenNums[n]

    # we may not want to use all the data when performing projection evaluation.
    def setPercentDataUsed(self, n):
        self.percentDataUsed = self.percentDataNums[n]

    def setMinTableLen(self, val):
        self.minExamples = int(str(val))

    def setBestSubsetsEdit(self, val):
        self.bestSubsets = int(str(val))

    # result list can contain projections with different number of attributes
    # user clicked in the listbox that shows possible number of attributes of result list
    # result list must be updated accordingly
    def attrLenListChanged(self):
        self.interestingList.clear()
        self.optimizedListFiltered = []

        # check which attribute lengths do we want to show
        self.attrLenDict = {}
        for i in range(self.attrLenList.count()):
            intVal = int(str(self.attrLenList.text(i)))
            selected = self.attrLenList.isSelected(i)
            self.attrLenDict[intVal] = selected

        # show in results list only those results, that are the correct length        
        for i in range(len(self.optimizedListFull)):
            (accuracy, itemCount, list, strList) = self.optimizedListFull[i]
            if self.attrLenDict[len(list)] == 1:
                self.interestingList.insertItem("(%.2f, %d) - %s"%(accuracy, itemCount, strList))
                self.optimizedListFiltered.append((accuracy, itemCount, list, strList))
        self.interestingList.setCurrentItem(0)        

    # insert new result - give parameters: accuracy of projection, number of examples in projection and list of attributes.
    # parameter strList can be a pre-formated string containing attribute list (used by polyviz)
    def insertItem(self, accuracy, tableLen, list, strList = None):
        if strList == None:
            strList = list[0]
            for item in list[1:]:
                strList = strList + ", " + item

        for i in range(len(self.optimizedListFull)):
            (accuracy2, tableLen2, list2, strList2) = self.optimizedListFull[i]
            if accuracy2 == accuracy and tableLen2 == tableLen and list2 == list and strList2 == strList:
                return
        self.optimizedListFull.append((accuracy, tableLen, list, strList))

    # check result list and update list with number of attributes
    # + select the first result in the list
    def updateNewResults(self):
        # update list of attribute lengths
        self.attrLenList.clear()
        self.attrLenDict = {}
        found = []
        for i in range(len(self.optimizedListFull)):
            (acc, tableLen, list, strList) = self.optimizedListFull[i]
            if len(list) not in found:
                found.append(len(list))
        found.sort()
        for val in found:
            self.attrLenList.insertItem(str(val))
            self.attrLenDict[val] = 1
        self.attrLenList.selectAll(1)
        self.interestingList.setCurrentItem(0)

    def clear(self):
        self.optimizedListFull = []
        self.optimizedListFiltered = []
        self.attrLenDict = {}        
        self.interestingList.clear()
        self.attrLenList.clear()

    # we can remove projections that have a specific attribute
    def filter(self):
        (Qstring,ok) = QInputDialog.getText("Remove attribute", "Remove projections with attribute:")
        if ok:
            attributeName = str(Qstring)
            for i in range(len(self.optimizedListFiltered)-1, -1, -1):
                (accuracy, itemCount, list, strList) = self.optimizedListFiltered[i]
                found = 0
                for val in list:
                    if val == attributeName: found = 1
                if found:
                    # remove from  listbox and original list of results
                    self.interestingList.removeItem(i)
                    self.optimizedListFull.remove((accuracy, itemCount, list, strList))
        self.updateNewResults()

    # remove projections that are selected
    def removeSelected(self):
        for i in range(self.interestingList.count()-1, -1, -1):
            if self.interestingList.isSelected(i):
                # remove from listbox and original list of results
                self.interestingList.removeItem(i)
                (accuracy, itemCount, list, strList) = self.optimizedListFiltered[i]
                self.optimizedListFiltered.remove((accuracy, itemCount, list, strList))
                self.optimizedListFull.remove((accuracy, itemCount, list, strList))

    # save the list into a file - filename can be set if you want to call this function without showing the dialog
    def save(self, filename = None):
        if filename == None:
            # get file name
            filename = "%s (k - %2d)" % (self.parentName, self.kValue )
            qname = QFileDialog.getSaveFileName( self.lastSaveDirName + "/" + filename, "Interesting projections (*.proj)", self, "", "Save Projections")
            if qname.isEmpty(): return
            name = str(qname)
        else:
            name = filename

        # take care of extension
        if os.path.splitext(name)[1] != ".proj":
            name = name + ".proj"

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        # open, write and save file
        file = open(name, "wt")
        file.write("%d\n%d\n%d\n%d\n" % (self.kValue, self.resultListLen, self.minExamples, self.percentDataUsed))
        file.write("%d\n"  % (self.qualityMeasure))
        file.write("%d\n%d\n" % (self.useHeuristics, self.bestSubsets))
        for val in self.optimizedListFiltered:
            file.write(str(val) + "\n")
        file.flush()
        file.close()

        os.chdir(os.path.dirname(name))

        

    # load projections from a file
    def load(self):
        self.clear()
                
        name = QFileDialog.getOpenFileName( self.lastSaveDirName, "Interesting projections (*.proj)", self, "", "Open Projections")
        if name.isEmpty(): return
        name = str(name)

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        file = open(name, "rt")
        self.kValue = int(file.readline()[:-1])
        self.resultListLen = int(file.readline()[:-1])
        self.minExamples = int(file.readline()[:-1])
        self.percentDataUsed = int(file.readline()[:-1])
        self.qualityMeasure = int(file.readline()[:-1])
        self.useHeuristics = int(file.readline()[:-1])
        self.bestSubsets = int(file.readline()[:-1])
        line = file.readline()[:-1]
        while (line != ""):
            self.optimizedListFull.append(eval(line))
            line = file.readline()[:-1]
        file.close()

        # show new settings in controls
        self.attrKNeighbour.setCurrentItem(self.kNeighboursNums.index(self.kValue))
        self.resultListCombo.setCurrentItem(self.resultsListLenNums.index(self.resultListLen))
        self.minTableLenEdit.setText(str(self.minExamples))
        self.percentDataUsedCombo.setCurrentItem(self.percentDataNums.index(self.percentDataUsed))

        self.measureCombo.setCurrentItem(self.qualityMeasure)
        #self.numberOfBestSubsetsEdit.setText(str(self.bestSubsets))
        #self.useHeuristicsCB.setChecked(self.useHeuristics)

        # update loaded results
        self.updateNewResults()


    # ###########################
    # kNNEvaluate - compute classification accuracy or brier score for data in table in percents
    def kNNComputeAccuracy(self, table):
        temp = 0.0
        percentDataUsed = int(str(self.percentDataUsedCombo.currentText()))
        experiments = 0            
        knn = orange.kNNLearner(table, k=self.kValue, rankWeight = 0)
        selection = orange.MakeRandomIndices2(table, 1.0-float(percentDataUsed)/100.0)

        qApp.processEvents()        # allow processing of other events

        # continuous class value
        if table.domain.classVar.varType == orange.VarTypes.Continuous:
            for j in range(len(table)):
                if selection[j] == 0: continue
                temp += pow(table[j].getclass() - knn(table[j]), 2)
                experiments += 1
            accuracy = temp/float(experiments)
            return 100.0*accuracy
    
        if not self.useLeaveOneOut:
            for j in range(len(table)):
                if selection[j] == 0: continue
                out = knn(table[j], orange.GetProbabilities)
                if self.qualityMeasure == AVERAGE_CORRECT:
                    temp += out[table[j].getclass()]
                elif self.qualityMeasure == CLASS_ACCURACY:
                    temp += out[table[j].getclass()] == max(out)
                else:
                    sum = 0
                    for val in out: sum += val*val
                    temp += sum - 2*out[table[j].getclass()]
                experiments += 1

            if self.qualityMeasure == BRIER_SCORE:
                return (temp + experiments)/(float(len(list(table.domain.classVar.values))*experiments))
            else:
                return 100.0 * temp / float(experiments)
        else:
            results = orngTest.leaveOneOut([orange.kNNLearner(k=self.kValue, rankWeight=0)], table)
            for res in results.results:
                if self.qualityMeasure == AVERAGE_CORRECT:
                    temp += res.probabilities[0][res.actualClass]
                elif self.qualityMeasure == CLASS_ACCURACY:
                    temp += res.probabilities[0][res.actualClass] == max(res.probabilities[0])
                else:
                    sum = 0
                    for val in res.probabilities[0]: sum += val*val
                    temp += sum - 2*res.probabilities[0][res.actualClass]
            if self.qualityMeasure == BRIER_SCORE:
                return (temp + len(results.results))/(float(len(list(table.domain.classVar.values)) * len(results.results)))
            else:
                return 100.0 * temp / len(results.results)

        
    # #############################
    # kNNClassifyData - compute classification error for every example in table
    def kNNClassifyData(self, table):
        knn = orange.kNNLearner(table, k=self.kValue, rankWeight = 0)
        
        qApp.processEvents()        # allow processing of other events

        # continuous class variable
        if table.domain.classVar.varType == orange.VarTypes.Continuous:
            for j in range(len(table)):
                returnTable.append(pow(table[j][2].value - knn(table[j]), 2))
            # normalize the data to the 0-1 interval
            maxError = max(returnTable)
            return [val/maxError for val in returnTable]

        lenClassValues = len(list(table.domain.classVar.values))
        returnTable = []
        if not self.useLeaveOneOut:
            for j in range(len(table)):
                out = knn(table[j], orange.GetProbabilities)
                if self.qualityMeasure == AVERAGE_CORRECT:
                    returnTable.append(out[table[j].getclass()])
                elif self.qualityMeasure == CLASS_ACCURACY:
                    returnTable.append(int(out[table[j].getclass()] == max(out)))
                else:
                    sum = 0
                    for val in out: sum += val*val
                    returnTable.append((sum + 1 - 2*out[table[j].getclass()])/float(lenClassValues))

        else:
            results = orngTest.leaveOneOut([orange.kNNLearner(k=self.kValue, rankWeight=0)], table)
            for res in results.results:
                if self.qualityMeasure == AVERAGE_CORRECT:
                    returnTable.append(res.probabilities[0][res.actualClass])
                elif self.qualityMeasure == CLASS_ACCURACY:
                    returnTable.append(int(res.probabilities[0][res.actualClass] == max(res.probabilities[0])))
                else:
                    sum = 0
                    for val in res.probabilities[0]: sum += val*val
                    returnTable.append((sum + 1 - 2*res.probabilities[0][res.actualClass])/float(lenClassValues))
        return returnTable

    # from a given dataset return list of (acc, attrs), where attrs are subsets of lenght subsetSize of attributes from table
    # that give the best kNN prediction on table
    def kNNGetInterestingSubsets(self, subsetSize, attributes, returnListSize, table, testingList = []):
        if attributes == [] or subsetSize == 0:
            if len(testingList) != subsetSize: return []

            # do the testing
            self.currentSubset += 1
            attrs = []
            for attr in testingList:
                attrs.append(table.domain[attr])
            domain = orange.Domain(attrs + [table.domain.classVar])
            shortTable = orange.Preprocessor_dropMissing(table.select(domain))
            text = "Current attribute subset (%d/%d): " % (self.currentSubset, self.totalSubsets)
            for attr in testingList[:-1]: text += attr + ", "
            text += testingList[-1]
            if len(shortTable) < self.minExamples:
                print text + " - ignoring (too few examples)"
                return []
            print text
            return [(self.kNNComputeAccuracy(shortTable), testingList)]

        full1 = self.kNNGetInterestingSubsets(subsetSize, attributes[1:], returnListSize, table, testingList)
        testingList2 = copy(testingList)
        testingList2.insert(0, attributes[0])
        full2 = self.kNNGetInterestingSubsets(subsetSize, attributes[1:], returnListSize, table, testingList2)

        # find max values in booth lists
        full = full1 + full2
        shortList = []

        if table.domain.classVar.varType == orange.VarTypes.Discrete and self.qualityMeasure == CLASS_ACCURACY: funct = max
        else: funct = min
        for i in range(min(returnListSize, len(full))):
            item = funct(full)
            full.remove(item)
            shortList.append(item)

        return shortList


    def disableControls(self):
        self.optimizeButtonBox.setEnabled(0)
        self.manageResultsBox.setEnabled(0)
        self.evaluateBox.setEnabled(0)
        self.measureCombo.setEnabled(0)

    def enableControls(self):    
        self.optimizeButtonBox.setEnabled(1)
        self.manageResultsBox.setEnabled(1)
        self.evaluateBox.setEnabled(1)
        self.measureCombo.setEnabled(1)

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=kNNOptimization()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()        