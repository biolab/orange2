from OWBaseWidget import *
from qt import *
from qwt import *
import sys
import cPickle
import os
import orange, orngTest, orngStat
from copy import copy
import OWGUI

CLASS_ACCURACY = 0
AVERAGE_CORRECT = 1
BRIER_SCORE = 2

LEAVE_ONE_OUT = 0
TEN_FOLD_CROSS_VALIDATION = 1
TEST_ON_LEARNING_SET = 2


class kNNOptimization(OWBaseWidget):
	EXACT_NUMBER_OF_ATTRS = 0
	MAXIMUM_NUMBER_OF_ATTRS = 1

	settingsList = ["kValue", "resultListLen", "percentDataUsed", "minExamples", "qualityMeasure", "testingMethod", "lastSaveDirName"]
	resultsListLenNums = [ 10 ,  20 ,  50 ,  100 ,  150 ,  200 ,  250 ,  300 ,  400 ,  500 ,  700 ,  1000 ,  2000 ]
	percentDataNums = [ 5 ,  10 ,  15 ,  20 ,  30 ,  40 ,  50 ,  60 ,  70 ,  80 ,  90 ,  100 ]
	kNeighboursNums = [ 0 ,  1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7 ,  8 ,  9 ,  10 ,  12 ,  15 ,  17 ,  20 ,  25 ,  30 ,  40 ,  60 ,  80 ,  100 ,  150 ,  200 ]
	resultsListLenList = [str(x) for x in resultsListLenNums]
	percentDataList = [str(x) for x in percentDataNums]
	kNeighboursList = [str(x) for x in kNeighboursNums]
	resultsListLenList = [str(x) for x in resultsListLenNums]

	def __init__(self,parent=None):
		OWBaseWidget.__init__(self, parent, "VizRank Optimization Dialog", "Find interesting projections of data", FALSE, FALSE, FALSE)

		self.setCaption("Qt VizRank Optimization Dialog")
		self.topLayout = QVBoxLayout( self, 10 ) 
		self.grid=QGridLayout(5,2)
		self.topLayout.addLayout( self.grid, 10 )

		self.kValue = 10
		self.minExamples = 0
		self.resultListLen = 100
		self.percentDataUsed = 100
		self.qualityMeasure = 0
		self.testingMethod = 0
		self.optimizationType = 0
		self.attributeCountIndex = 0
		self.onlyOnePerSubset = 1	# used in radviz and polyviz
		self.widgetDir = os.path.realpath(os.path.dirname(__file__)) + "/"
		self.parentName = "Projection"
		self.lastSaveDirName = os.getcwd() + "/"

		self.allResults = []
		self.shownResults = []
		self.attrLenDict = {}

		self.loadSettings()
		self.useLeaveOneOut = 1

		self.optimizationSettingsBox = OWGUI.widgetBox(self, " Optimization Settings ")
		self.optimizationBox = OWGUI.widgetBox(self, " Find interesting projections... ")
		self.evaluateBox = OWGUI.widgetBox(self, " Evaluate projection / classifier ")
		self.manageResultsBox = OWGUI.widgetBox(self, " Manage projections ")
		self.resultsBox = OWGUI.widgetBox(self, " List of interesting projections ")

		self.grid.addWidget(self.optimizationSettingsBox,0,0)
		self.grid.addWidget(self.optimizationBox,1,0)
		self.grid.addWidget(self.evaluateBox,2,0)
		self.grid.addWidget(self.manageResultsBox,3,0)
		self.grid.addMultiCellWidget (self.resultsBox,0,3, 1, 1)
		self.grid.setColStretch(0, 0)
		self.grid.setColStretch(1, 100)
		self.grid.setRowStretch(0, 0)
		self.grid.setRowStretch(1, 0)
		self.grid.setRowStretch(2, 0)
		self.grid.setRowStretch(3, 100)
				
		self.resultList = QListBox(self.resultsBox)
		#self.resultList.setSelectionMode(QListBox.Extended)   # this would be nice if could be enabled, but it has a bug - currentItem doesn't return the correct value if this is on
		self.resultList.setMinimumSize(200,200)

		self.attrKNeighboursCombo = OWGUI.comboBoxWithCaption(self.optimizationSettingsBox, self, "kValue", "Number of neighbors (k): ", tooltip = "Number of neighbors used in k-NN algorithm to evaluate the projection", items = self.kNeighboursNums, sendSelectedValue = 1, valueType = int)
		self.resultListCombo = OWGUI.comboBoxWithCaption(self.optimizationSettingsBox, self, "resultListLen", "Number of most interesting projections: ", tooltip = "Maximum length of the list of interesting projections", items = self.resultsListLenNums, callback = self.updateShownProjections, sendSelectedValue = 1, valueType = int)
		self.minTableLenEdit = OWGUI.lineEdit(self.optimizationSettingsBox, self, "minExamples", "Minimum examples in data set:        ", orientation = "horizontal", tooltip = "Due to missing values, different subsets of attributes can have different number of examples. Projections with less than this number of examples will be ignored.", valueType = int)
		self.percentDataUsedCombo= OWGUI.comboBoxWithCaption(self.optimizationSettingsBox, self, "percentDataUsed", "Percent of data used in evaluation: ", tooltip = "Maximum length of the list of interesting projections", items = self.percentDataNums, sendSelectedValue = 1, valueType = int)

		self.measureCombo = OWGUI.comboBox(self.optimizationSettingsBox, self, "qualityMeasure", box = " Measure of classification success ", items = ["Classification accuracy", "Average probability of correct classification", "Brier score"], tooltip = "Measure to evaluate prediction accuracy of k-NN method on the projected data set.")
		self.testingCombo = OWGUI.comboBox(self.optimizationSettingsBox, self, "testingMethod", box = " Testing method ", items = ["Leave one out (slowest, most accurate)", "10 fold cross validation", "Test on learning set (fastest, least accurate)"], tooltip = "Method for evaluating the classifier. Slower are more accurate while faster give only a rough approximation.") 

		self.buttonBox = OWGUI.widgetBox(self.optimizationBox, orientation = "horizontal")
		self.optimizationTypeCombo = OWGUI.comboBox(self.buttonBox, self, "optimizationType", items = ["    with exactly    ", "  with maximum  "] )
		self.attributeCountCombo = OWGUI.comboBox(self.buttonBox, self, "attributeCountIndex", tooltip = "Evaluate only projections with exactly (or maximum) this number of attributes")
		QLabel(' attributes', self.buttonBox)

		self.startOptimizationButton = OWGUI.button(self.optimizationBox, self, "Start evaluating projections")

		for i in range(3,15):
			self.attributeCountCombo.insertItem(str(i))
		self.attributeCountCombo.insertItem("ALL")
		self.attributeCountIndex = 0

		self.buttonBox3 = OWGUI.widgetBox(self.evaluateBox, orientation = "horizontal")
		self.evaluateProjectionButton = OWGUI.button(self.buttonBox3, self, 'Evaluate projection')
		self.saveProjectionButton = OWGUI.button(self.buttonBox3, self, 'Save projection')

		self.buttonBox4 = OWGUI.widgetBox(self.evaluateBox, orientation = "horizontal")
		self.showKNNCorrectButton = OWGUI.button(self.buttonBox4, self, 'kNN correct')
		self.showKNNWrongButton = OWGUI.button(self.buttonBox4, self, 'kNN wrong')
		self.showKNNResetButton = OWGUI.button(self.buttonBox4, self, 'Original') 
				
		self.attrLenCaption = QLabel('Number of concurrently visualized attributes:', self.manageResultsBox)
		self.attrLenList = QListBox(self.manageResultsBox)
		self.attrLenList.setSelectionMode(QListBox.Multi)
		self.attrLenList.setMinimumSize(60,60)

		self.reevaluateResults = OWGUI.button(self.manageResultsBox, self, "Reevaluate shown projections")
		self.buttonBox5 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
		self.buttonBox6 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
		self.buttonBox7 = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
		self.filterButton = OWGUI.button(self.buttonBox5, self, "Remove attribute", self.filter)
		self.removeSelectedButton = OWGUI.button(self.buttonBox5, self, "Remove selection", self.removeSelected)
		self.loadButton = OWGUI.button(self.buttonBox6, self, "Load", self.load)
		self.saveButton = OWGUI.button(self.buttonBox6, self, "Save", self.save)
		self.clearButton = OWGUI.button(self.buttonBox7, self, "Clear results", self.clearResults)
		self.closeButton = OWGUI.button(self.buttonBox7, self, "Close", self.hide)

	def destroy(self, dw, dsw):
		self.saveSettings()

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

	def updateShownProjections(self, *args):
		self.resultList.clear()
		self.shownResults = []
		i = 0
		while self.resultList.count() < self.resultListLen and i < len(self.allResults):
			if self.attrLenDict[len(self.allResults[i][2])] == 1:
				self.resultList.insertItem("(%.2f, %d) - %s"%(self.allResults[i][0], self.allResults[i][1], self.allResults[i][3]))
				self.shownResults.append(self.allResults[i])
			i+=1
		if self.resultList.count() > 0: self.resultList.setCurrentItem(0)		

	def getOptimizationType(self):
		return self.optimizationType

	def getQualityMeasure(self):
		return self.qualityMeasure

	def getAllResults(self):
		return self.allResults

	def getShownResults(self):
		return self.shownResults

	def getSelectedProjection(self):
		if self.resultList.count() == 0: return None
		return self.shownResults[self.resultList.currentItem()]
		

	def clearResults(self):
		self.allResults = []
		self.shownResults = []
		self.resultList.clear()
		self.attrLenDict = {}
		self.attrLenList.clear()

	def addResult(self, rawdata, accuracy, lenTable, attrList, strList = None):
		if rawdata.domain.classVar.varType == orange.VarTypes.Discrete and self.getQualityMeasure() != BRIER_SCORE: funct = max
		else: funct = min

		targetIndex = self.findTargetIndex(accuracy, funct)
		self.insertItem(accuracy, lenTable, attrList, targetIndex, strList)

	def findTargetIndex(self, accuracy, funct):
		# use bisection to find correct index
		top = 0; bottom = len(self.allResults)

		while (bottom-top) > 1:
			mid  = (bottom + top)/2
			if funct(accuracy, self.allResults[mid][0]) == accuracy: bottom = mid
			else: top = mid

		if len(self.allResults) == 0: return 0
		if funct(accuracy, self.allResults[top][0]) == accuracy:
			return top
		else: 
			return bottom

	# insert new result - give parameters: accuracy of projection, number of examples in projection and list of attributes.
	# parameter strList can be a pre-formated string containing attribute list (used by polyviz)
	#def insertItem(self, accuracy, tableLen, list, strList = None, index = -1):
	def insertItem(self, accuracy, lenTable, attrList, index, strList = None):
		if strList == None:
			strList = attrList[0]
			for item in attrList[1:]:
				strList = strList + ", " + item

		self.allResults.insert(index, (accuracy, lenTable, attrList, strList))
		if index < self.resultListLen:
			self.resultList.insertItem("(%.2f, %d) - %s"%(accuracy, lenTable, strList), index)
			self.shownResults.insert(index, (accuracy, lenTable, attrList, strList))

		# remove worst projection if list is too long
		if self.resultList.count() > self.resultListLen:
			self.resultList.removeItem(self.resultList.count()-1)
			self.shownResults.pop()
	
	def finishedAddingResults(self):
		self.attrLenList.clear()
		self.attrLenDict = {}
		found = []
		for i in range(len(self.shownResults)):
			if len(self.shownResults[i][2]) not in found:
				found.append(len(self.shownResults[i][2]))
		found.sort()
		for val in found:
			self.attrLenList.insertItem(str(val))
			self.attrLenDict[val] = 1
		self.attrLenList.selectAll(1)
		self.resultList.setCurrentItem(0)

	
	# we can remove projections that have a specific attribute
	def filter(self):
		(Qstring,ok) = QInputDialog.getText("Remove attribute", "Remove projections with attribute:")
		if ok:
			attributeName = str(Qstring)
			for i in range(len(self.shownResults)-1, -1, -1):
				(accuracy, itemCount, list, strList) = self.shownResults[i]
				if attributeName in list:		# remove from  listbox and original list of results
					self.shownResults.remove(self.shownResults[i])
					self.resultList.removeItem(i)

	# remove projections that are selected
	def removeSelected(self):
		for i in range(self.resultList.count()-1, -1, -1):
			if self.resultList.isSelected(i):
				# remove from listbox and original list of results
				self.resultList.removeItem(i)
				self.shownResults.remove(self.shownResults[i])

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
		attrs = ["kValue", "minExamples", "resultListLen", "percentDataUsed", "qualityMeasure", "testingMethod"]
		dict = {}
		for attr in attrs:
			dict[attr] = self.__dict__[attr]
		file.write("%s\n" % (str(dict)))
		for val in self.shownResults:
			file.write(str(val) + "\n")
		file.flush()
		file.close()


	# load projections from a file
	def load(self):
		self.clearResults()
				
		name = QFileDialog.getOpenFileName( self.lastSaveDirName, "Interesting projections (*.proj)", self, "", "Open Projections")
		if name.isEmpty(): return
		name = str(name)

		dirName, shortFileName = os.path.split(name)
		self.lastSaveDirName = dirName

		file = open(name, "rt")
		settings = eval(file.readline()[:-1])
		self.setSettings(settings)

		line = file.readline()[:-1]; ind = 0
		while (line != ""):
			(acc, lenTable, attrList, strList) = eval(line)
			self.insertItem(acc, lenTable, attrList, ind, strList)
			line = file.readline()[:-1]
		file.close()

		# update loaded results
		self.finishedAddingResults()


	# ###########################
	# kNNEvaluate - compute classification accuracy or brier score for data in table in percents
	def kNNComputeAccuracy(self, table):
		# select a subset of the data if necessary
		percentDataUsed = int(str(self.percentDataUsedCombo.currentText()))
		if percentDataUsed != 100:
			indices = orange.MakeRandomIndices2(table, 1.0-float(percentDataUsed)/100.0)
			testTable = table.select(indices)
		else: testTable = table
		
		qApp.processEvents()		# allow processing of other events

		knn = orange.kNNLearner(k=self.kValue, rankWeight = 0, distanceConstructor = orange.ExamplesDistanceConstructor_Euclidean(normalize=0))

		if self.testingMethod == LEAVE_ONE_OUT:	
			results = orngTest.leaveOneOut([knn], testTable)
		elif self.testingMethod == TEN_FOLD_CROSS_VALIDATION:
			results = orngTest.crossValidation([knn], testTable)
		else:
			# wrong but fast way to evaluate the accuracy
			results = orngTest.ExperimentResults(1, ["kNN"], list(table.domain.classVar.values), 0, table.domain.classVar.baseValue)
			results.results = [orngTest.TestedExample(i, int(table[i].getclass()), 1) for i in range(len(table))]
			classifier = knn(table)
			for i in range(len(table)):
				cls, pro = classifier(table[i], orange.GetBoth)
				results.results[i].setResult(0, cls, pro)
	
		# compute classification success using selected measure
		if self.qualityMeasure == AVERAGE_CORRECT:
			val = 0.0
			for res in results.results:
				val += res.probabilities[0][res.actualClass]
			val/= float(len(results.results))
			return 100.0*val
		elif self.qualityMeasure == BRIER_SCORE:
			return orngStat.BrierScore(results)[0]
		elif self.qualityMeasure == CLASS_ACCURACY:
			return 100*orngStat.CA(results)[0]

		
	# #############################
	# kNNClassifyData - compute classification error for every example in table
	def kNNClassifyData(self, table):
		qApp.processEvents()		# allow processing of other events
		
		knn = orange.kNNLearner(k=self.kValue, rankWeight = 0, distanceConstructor = orange.ExamplesDistanceConstructor_Euclidean(normalize=0))

		if self.testingMethod == LEAVE_ONE_OUT:	
			results = orngTest.leaveOneOut([knn], testTable)
		elif self.testingMethod == TEN_FOLD_CROSS_VALIDATION:
			results = orngTest.crossValidation([knn], testTable)
		else:
			# wrong but fast way to evaluate the accuracy
			results = orngTest.ExperimentResults(1, ["kNN"], list(table.domain.classVar.values), 0, table.domain.classVar.baseValue)
			results.results = [orngTest.TestedExample(i, int(table[i].getclass()), 1) for i in range(len(table))]
			classifier = knn(table)
			for i in range(len(table)):
				cls, pro = classifier(table[i], orange.GetBoth)
				results.results[i].setResult(0, cls, pro)
	
		returnTable = []
		lenClassValues = len(list(table.domain.classVar.values))
		if self.qualityMeasure == AVERAGE_CORRECT:
			for res in results.results:
				returnTable.append(res.probabilities[0][res.actualClass])
		elif self.qualityMeasure == BRIER_SCORE:
			for res in results.results:
				sum = 0
				for val in res.probabilities[0]: sum += val*val
				returnTable.append((sum + 1 - 2*res.probabilities[0][res.actualClass])/float(lenClassValues))
		elif self.qualityMeasure == CLASS_ACCURACY:
			for res in results.results:
				returnTable.append(res.probabilities[0][res.actualClass] == max(res.probabilities[0]))

		return returnTable

	# TEST TEST TEST TEST !!!
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
		self.optimizationSettingsBox.setEnabled(0)
		self.manageResultsBox.setEnabled(0)
		self.evaluateBox.setEnabled(0)
		self.measureCombo.setEnabled(0)

	def enableControls(self):	
		self.optimizationSettingsBox.setEnabled(1)
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