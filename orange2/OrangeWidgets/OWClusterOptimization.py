from OWBaseWidget import *
import os
import orange, orngTest
from copy import copy
from math import sqrt
import OWGUI, OWDlgs
import orngVisFuncts
import numpy

VALUE = 0
CLOSURE = 1
VERTICES = 2
ATTR_LIST = 3
CLASS = 4
ENL_CLOSURE = 5
OTHER = 6
STR_LIST = 7

MUST_BE_INSIDE = 0  # values for self.conditionForArgument
CAN_LIE_NEAR = 1

BEST_GROUPS = 0
BEST_GROUPS_IN_EACH_CLASS = 1

contMeasures = [("None", None), ("ReliefF", orange.MeasureAttribute_relief(k=10, m=50)), ("Fisher discriminant", orngVisFuncts.MeasureFisherDiscriminant()), ("Signal to Noise Ratio", orngVisFuncts.S2NMeasure()), ("Signal to Noise Ratio For Each Class", orngVisFuncts.S2NMeasureMix())]
discMeasures = [("None", None), ("ReliefF", orange.MeasureAttribute_relief(k=10, m=50)), ("Gain ratio", orange.MeasureAttribute_gainRatio()), ("Gini index", orange.MeasureAttribute_gini())]

VALUE = 0
CLUSTER = 1
DISTANCE = 2

OTHER_CLASS = 0
OTHER_VALUE = 1
OTHER_POINTS = 2
OTHER_DISTANCE = 3
OTHER_AVERAGE = 4
OTHER_AVERAGE_DIST = 5


class ClusterOptimization(OWBaseWidget):
    EXACT_NUMBER_OF_ATTRS = 0
    MAXIMUM_NUMBER_OF_ATTRS = 1

    settingsList = ["resultListLen", "minExamples", "lastSaveDirName", "attrCont", "attrDisc", "showRank",
                    "showValue", "jitterDataBeforeTriangulation", "useProjectionValue",
                    "evaluationTime", "distributionScale", "removeDistantPoints", "useAlphaShapes", "alphaShapesValue",
                    "argumentCountIndex", "evaluationTimeIndex", "conditionForArgument", "moreArgumentsIndex", "canUseMoreArguments",
                    "parentWidget.clusterClassifierName"]
    #resultsListLenNums = [ 100 ,  250 ,  500 ,  1000 ,  5000 , 10000, 20000, 50000, 100000, 500000 ]
    resultsListLenNums = [ 100 ,  250 ,  500 ,  1000 ,  5000 , 10000]
    resultsListLenList = [str(x) for x in resultsListLenNums]
    argumentCounts = [1, 5, 10, 20, 40, 100, 100000]
    evaluationTimeNums = [0.5, 1, 2, 5, 10, 20, 60, 120]
    evaluationTimeList = [str(x) for x in evaluationTimeNums]

    moreArgumentsNums = [60, 65, 70, 75, 80, 85, 90, 95]
    moreArgumentsList = ["%d %%" % x for x in moreArgumentsNums]

    def __init__(self, parentWidget = None, signalManager = None, graph = None, parentName = "Visualization widget"):
        OWBaseWidget.__init__(self, None, signalManager, "Cluster Dialog")

        self.parentWidget = parentWidget
        self.parentName = parentName
        self.setCaption("Cluster Dialog")
        self.controlArea = QVBoxLayout(self)

        self.graph = graph
        self.minExamples = 0
        self.resultListLen = 1000
        self.maxResultListLen = self.resultsListLenNums[len(self.resultsListLenNums)-1]
        self.onlyOnePerSubset = 1    # used in radviz and polyviz
        self.lastSaveDirName = os.getcwd() + "/"
        self.attrCont = 1
        self.attrDisc = 1
        self.rawData = None
        self.subsetdata = None
        self.arguments = []
        self.selectedClasses = []
        self.optimizationType = 1
        self.jitterDataBeforeTriangulation = 0
        self.parentWidget.clusterClassifierName = "Visual cluster classifier"

        self.showRank = 0
        self.showValue = 1
        self.allResults = []
        self.shownResults = []
        self.attrLenDict = {}
        self.datasetName = ""
        self.dataset = None
        self.cancelOptimization = 0
        self.cancelArgumentation = 0
        self.pointStability = None
        self.pointStabilityCount = None
        self.attributeCountIndex = 1
        self.argumentationType = BEST_GROUPS

        self.argumentationClassValue = 0
        self.distributionScale = 1
        self.considerDistance = 1
        self.useProjectionValue = 0
        self.evaluationTime = 30
        self.useAlphaShapes = 1
        self.alphaShapesValue = 1.5
        self.removeDistantPoints = 1
        self.evaluationTimeIndex = 4
        self.conditionForArgument = 0
        self.argumentCountIndex = 1     # when classifying use 10 best arguments
        self.canUseMoreArguments = 0
        self.moreArgumentsIndex = 4

        self.loadSettings()

        self.tabs = QTabWidget(self, 'tabWidget')
        self.controlArea.addWidget(self.tabs)

        self.MainTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self)
        self.ArgumentationTab = QVGroupBox(self)
        self.ClassificationTab = QVGroupBox(self)
        self.ManageTab = QVGroupBox(self)

        self.tabs.insertTab(self.MainTab, "Main")
        self.tabs.insertTab(self.SettingsTab, "Settings")
        self.tabs.insertTab(self.ArgumentationTab, "Argumentation")
        self.tabs.insertTab(self.ClassificationTab, "Classification")
        self.tabs.insertTab(self.ManageTab, "Manage & Save")


        # ###########################
        # MAIN TAB
        self.optimizationBox = OWGUI.widgetBox(self.MainTab, "Evaluate")
        self.resultsBox = OWGUI.widgetBox(self.MainTab, "Projection list, most interesting first")
        self.resultsDetailsBox = OWGUI.widgetBox(self.MainTab, "Shown details in projections list" , orientation = "horizontal")
        self.buttonBox = OWGUI.widgetBox(self.optimizationBox, orientation = "horizontal")
        self.label1 = QLabel('Projections with ', self.buttonBox)
        self.optimizationTypeCombo = OWGUI.comboBox(self.buttonBox, self, "optimizationType", items = ["    exactly    ", "  maximum  "] )
        self.attributeCountCombo = OWGUI.comboBox(self.buttonBox, self, "attributeCountIndex", items = [str(x) for x in range(3, 15)] + ["ALL"], tooltip = "Evaluate only projections with exactly (or maximum) this number of attributes")
        self.attributeLabel = QLabel(' attributes', self.buttonBox)

        self.startOptimizationButton = OWGUI.button(self.optimizationBox, self, "Start Evaluating Projections")
        f = self.startOptimizationButton.font()
        f.setBold(1)
        self.startOptimizationButton.setFont(f)
        self.stopOptimizationButton = OWGUI.button(self.optimizationBox, self, "Stop Evaluation", callback = self.stopOptimizationClick)
        self.stopOptimizationButton.setFont(f)
        self.stopOptimizationButton.hide()

        self.resultList = OWGUI.listBox(self.resultsBox, self)
        #self.resultList.setSelectionMode(QListWidget.ExtendedSelection)   # this would be nice if could be enabled, but it has a bug - currentItem doesn't return the correct value if this is on
        self.resultList.setMinimumSize(200,200)

        self.showRankCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showRank', 'Rank', callback = self.updateShownProjections, tooltip = "Show projection ranks")
        self.showValueCheck = OWGUI.checkBox(self.resultsDetailsBox, self, 'showValue', 'Cluster value', callback = self.updateShownProjections, tooltip = "Show the cluster value")


        # ##########################
        # SETTINGS TAB
        self.jitteringBox = OWGUI.checkBox(self.SettingsTab, self, 'jitterDataBeforeTriangulation', 'Use data jittering', box = "Jittering options", tooltip = "Use jittering if you get an exception when evaluating clusters. \nIt adds a small random noise to poins which fixes the triangluation problems.")
        self.clusterEvaluationBox = OWGUI.widgetBox(self.SettingsTab, "Cluster detection settings")
        alphaBox = OWGUI.widgetBox(self.clusterEvaluationBox, orientation = "horizontal")
        OWGUI.checkBox(alphaBox, self, "useAlphaShapes", "Use alpha shapes. Alpha value is :  ", tooltip = "Break separated clusters with same class value into subclusters", callback = self.updateGraph)
        OWGUI.hSlider(alphaBox, self, "alphaShapesValue", minValue = 0.0, maxValue = 30, step = 1, callback = self.updateGraph, labelFormat="%.1f", ticks = 5, divideFactor = 10.0)
        OWGUI.checkBox(self.clusterEvaluationBox, self, "removeDistantPoints", "Remove distant points", tooltip = "Remove points from the cluster boundary that lie closer to examples with different class value", callback = self.updateGraph)

        valueBox = OWGUI.widgetBox(self.SettingsTab, "Cluster value computation")
        self.distributionScaleCheck = OWGUI.checkBox(valueBox, self, "distributionScale", "Scale cluster values according to class distribution", tooltip = "Cluster value is (among other things) determined by the number of points inside the cluster. \nThis criteria is unfair in data sets with uneven class distributions.\nThis option takes this into an account by transforming the number of covered points into percentage of all points with the cluster class value.")
        self.considerDistanceCheck = OWGUI.checkBox(valueBox, self, "considerDistance", "Consider distance between clusters", tooltip = "If checked, cluster value is defined also by the distance between the cluster points and nearest points that belong to a different class")

        self.heuristicsSettingsBox = OWGUI.widgetBox(self.SettingsTab, "Heuristics for attribute ordering")
        self.miscSettingsBox = OWGUI.widgetBox(self.SettingsTab, "Miscellaneous settings")

        OWGUI.comboBox(self.heuristicsSettingsBox, self, "attrCont", box = "Ordering of continuous attributes", items = [val for (val, m) in contMeasures])
        OWGUI.comboBox(self.heuristicsSettingsBox, self, "attrDisc", box = "Ordering of discrete attributes", items = [val for (val, m) in discMeasures])
        
        self.resultListCombo = OWGUI.comboBoxWithCaption(self.miscSettingsBox, self, "resultListLen", "Maximum length of projection list:"+"   ", tooltip = "Maximum length of the list of interesting projections. This is also the number of projections that will be saved if you click Save button.", items = self.resultsListLenNums, callback = self.updateShownProjections, sendSelectedValue = 1, valueType = int)
        self.minTableLenEdit = OWGUI.lineEdit(self.miscSettingsBox, self, "minExamples", "Minimum examples in data set:"+"        ", orientation = "horizontal", tooltip = "Due to missing values, different subsets of attributes can have different number of examples. Projections with less than this number of examples will be ignored.", valueType = int)

        # ##########################
        # ARGUMENTATION tab
        self.argumentationStartBox = OWGUI.widgetBox(self.ArgumentationTab, "Arguments")
        self.findArgumentsButton = OWGUI.button(self.argumentationStartBox, self, "Find Arguments", callback = self.findArguments)
        f = self.findArgumentsButton.font(); f.setBold(1);  self.findArgumentsButton.setFont(f)
        self.stopArgumentationButton = OWGUI.button(self.argumentationStartBox, self, "Stop Searching", callback = self.stopArgumentationClick)
        self.stopArgumentationButton.setFont(f)
        self.stopArgumentationButton.hide()
        self.classValueList = OWGUI.comboBox(self.ArgumentationTab, self, "argumentationClassValue", box = "Arguments for class:", tooltip = "Select the class value that you wish to see arguments for", callback = self.argumentationClassChanged)
        self.argumentBox = OWGUI.widgetBox(self.ArgumentationTab, "Arguments for the selected class value")
        self.argumentList = OWGUI.listBox(self.argumentBox, self)
        self.argumentList.setMinimumSize(200,200)
        self.connect(self.argumentList, SIGNAL("selectionChanged()"),self.argumentSelected)

        # ##########################
        # CLASSIFICATION TAB
        self.classifierNameEdit = OWGUI.lineEdit(self.ClassificationTab, self, 'parentWidget.clusterClassifierName', box = ' Learner / Classifier Name ', tooltip='Name to be used by other widgets to identify your learner/classifier.')
        self.useProjectionValueCheck = OWGUI.checkBox(self.ClassificationTab, self, "useProjectionValue", "Use projection score when voting", box = "Voting for class value", tooltip = "Does each projection count for 1 vote or is it dependent on the value of the projection")
        OWGUI.comboBox(self.ClassificationTab, self, "argumentationType", box = "When searching for arguments consider ... ", items = ["... best evaluated groups", "... best groups for each class value"], tooltip = "When you wish to find arguments or classify an example, do you wish to search groups from the begining of the list\nor do you want to consider best groups for each class value. \nExplanation: For some class value evaluated groups might have significantly lower values than for other classes. \nIf you select 'best evaluated groups' you therefore won't even give a chance to this class value, \nsince its groups will be much lower in the list of evaluated groups.")
        self.conditionCombo = OWGUI.comboBox(self.ClassificationTab, self, "conditionForArgument", box = "Condition for a cluster to be an argument for an example is that...", items = ["... the example lies inside the cluster", "... the example lies inside or near the cluster"], tooltip = "When searching for arguments or classifying an example we have to define when can a detected cluster be an argument for a class.\nDoes the point being classified have to lie inside that cluster or is it enough that it lies near it.\nIf nearness is enough than the point can be away from the cluster for the distance that is defined as an average distance between points inside the cluster.")
        self.evaluationTimeEdit = OWGUI.comboBoxWithCaption(self.ClassificationTab, self, "evaluationTimeIndex", "Time for evaluating projections (minutes): ", box = "Evaluating time", tooltip = "The maximum time that the classifier is allowed for evaluating projections (learning)", items = self.evaluationTimeList)
        projCountBox = OWGUI.widgetBox(self.ClassificationTab, "Argument count")
        self.argumentCountEdit = OWGUI.comboBoxWithCaption(projCountBox, self, "argumentCountIndex", "Maximum number of arguments used when classifying: ", tooltip = "The maximum number of arguments that will be used when classifying an example.", items = ["1", "5", "10", "20", "40", "100", "All"])
        projCountBox2 = OWGUI.widgetBox(projCountBox, orientation = "horizontal")
        self.canUseMoreArgumentsCheck = OWGUI.checkBox(projCountBox2, self, "canUseMoreArguments", "Use additional projections until probability at least: ", tooltip = "If checked, it will allow the classifier to use more arguments when it is not confident enough in the prediction.\nIt will use additional arguments until the predicted probability of one class value will be at least as much as specified in the combo box")
        self.moreArgumentsCombo = OWGUI.comboBox(projCountBox2, self, "moreArgumentsIndex", items = self.moreArgumentsList, tooltip = "If checked, it will allow the classifier to use more arguments when it is not confident enough in the prediction.\nIt will use additional arguments until the predicted probability of one class value will be at least as much as specified in the combo box")

        # ##########################
        # SAVE & MANAGE TAB
        self.classesBox = OWGUI.widgetBox(self.ManageTab, "Select class values you wish to separate")
        self.manageResultsBox = OWGUI.widgetBox(self.ManageTab, "Number of concurrently visualized attributes")
        self.manageBox = OWGUI.widgetBox(self.ManageTab, "Manage projections")

        self.classesList = OWGUI.listBox(self.classesBox, selectionMode = QListWidget.MultiSelection, callback = self.updateShownProjections)
        self.classesList.setMinimumSize(60,60)

        self.buttonBox6 = OWGUI.widgetBox(self.manageBox, orientation = "horizontal")
        OWGUI.button(self.buttonBox6, self, "Load", self.load)
        OWGUI.button(self.buttonBox6, self, "Save", self.save)

        #self.buttonBox7 = OWGUI.widgetBox(self.manageBox, orientation = "horizontal")
        #OWGUI.button(self.buttonBox7, self, "Graph projections", self.graphProjectionQuality)
        #OWGUI.button(self.buttonBox7, self, "Interaction Analysis", self.interactionAnalysis)

        self.buttonBox3 = OWGUI.widgetBox(self.manageBox, orientation = "horizontal")
        self.clusterStabilityButton = OWGUI.button(self.buttonBox3, self, 'Show cluster stability', self.evaluatePointsInClusters)
        self.clusterStabilityButton.setCheckable(1)
        #self.saveProjectionButton = OWGUI.button(self.buttonBox3, self, 'Save projection')
        OWGUI.button(self.buttonBox3, self, "Save Best Graphs", self.exportMultipleGraphs)

        OWGUI.button(self.manageBox, self, "Clear Results", self.clearResults)

        self.attrLenList = OWGUI.listBox(self.manageResultsBox, self, selectionMode = QListWidget.MultiSelection, callback = self.attrLenListChanged)
        self.attrLenList.setMinimumSize(60,60)

        self.resize(375,550)
        self.setMinimumWidth(375)
        self.tabs.setMinimumWidth(375)

        self.statusBar = QStatusBar(self)
        self.controlArea.addWidget(self.statusBar)
        self.controlArea.activate()

        self.connect(self.classifierNameEdit, SIGNAL("textChanged(const QString &)"), self.changeLearnerName)
        if self.parentWidget:
            if hasattr(self.parentWidget, "learnersArray"):
                self.parentWidget.learnersArray[1] = clusterLearner(self, self.parentWidget)
            else:
                self.clusterLearner = clusterLearner(self, self.parentWidget)
                self.parentWidget.send("Cluster learner", self.clusterLearner)


    # ##############################################################
    # EVENTS
    # ##############################################################
    # when text of vizrank or cluster learners change update their name
    def changeLearnerName(self, text):
        if self.parentWidget:
            if hasattr(self.parentWidget, "learnersArray"):
                self.parentWidget.learnersArray[1].name = self.parentWidget.clusterClassifierName
            else:
                self.clusterLearner.name = self.parentWidget.clusterClassifierName
        else: print "there is no instance of Cluster Learner"

    def updateGraph(self):
        if self.parentWidget: self.parentWidget.updateGraph()

    # result list can contain projections with different number of attributes
    # user clicked in the listbox that shows possible number of attributes of result list
    # result list must be updated accordingly
    def attrLenListChanged(self, *args):
        # check which attribute lengths do we want to show
        self.attrLenDict = {}
        for i in range(self.attrLenList.count()):
            intVal = int(str(self.attrLenList.item(i).text()))
            selected = self.attrLenList.item(i).isSelected()
            self.attrLenDict[intVal] = selected
        self.updateShownProjections()

    def clearResults(self):
        del self.allResults; self.allResults = []
        del self.shownResults; self.shownResults = []
        self.resultList.clear()
        self.attrLenDict = {}
        self.attrLenList.clear()
        del self.pointStability; self.pointStability = None

    def clearArguments(self):
        del self.arguments; self.arguments = []
        self.argumentList.clear()

    def getSelectedClassValues(self):
        selectedClasses = []
        for i in range(self.classesList.count()-1):
            if self.classesList.item(i).isSelected(): selectedClasses.append(i)
        if self.classesList.item(self.classesList.count()-1).isSelected():
            selectedClasses.append(-1)      # "all clusters" is represented by -1 in the selectedClasses array
        return selectedClasses

    # ##############################################################
    # ##############################################################
    def evaluateClusters(self, data):
        import orangeom
        graph = orangeom.triangulate(data,0,1,3)
        graph.returnIndices = 1
        computeDistances(graph)
        removeEdgesToDifferentClasses(graph, -3)    # None
        removeSingleLines(graph, None, None, None, 0, -1)
        edgesDict, clusterDict, verticesDict, count = enumerateClusters(graph, 0)

        closureDict = computeClosure(graph, edgesDict, 1, 1, 1, -3, -4)   # find points that define the edge of a cluster with one class value. these points used when we search for nearest points of a specific cluster
        #bigPolygonVerticesDict = copy(verticesDict)  # compute which points lie in which cluster
        otherClassDict = getPointsWithDifferentClassValue(graph, closureDict)   # this dict is used when evaluating a cluster to find nearest points that belong to a different class
        removeSingleTrianglesAndLines(graph, edgesDict, clusterDict, verticesDict, -1)
        #bigPolygonVerticesDict = copy(verticesDict)

        if self.useAlphaShapes:
            computeAlphaShapes(graph, edgesDict, self.alphaShapesValue / 10.0, -1, 0)          # try to break too empty clusters into more clusters
            del edgesDict, clusterDict, verticesDict
            edgesDict, clusterDict, verticesDict, count = enumerateClusters(graph, 1)    # create the new clustering
            fixDeletedEdges(graph, edgesDict, clusterDict, 0, 1, -1)  # None             # restore edges that were deleted with computeAlphaShapes and did not result breaking the cluster
        del closureDict
        closureDict = computeClosure(graph, edgesDict, 1, 1, 2, -3, -4)

        if self.removeDistantPoints:
            removeDistantPointsFromClusters(graph, edgesDict, clusterDict, verticesDict, closureDict, -2)
        removeSingleLines(graph, edgesDict, clusterDict, verticesDict, 1, -1)   # None
        del edgesDict, clusterDict, verticesDict
        edgesDict, clusterDict, verticesDict, count = enumerateClusters(graph, 1)     # reevaluate clusters - now some clusters might have disappeared
        removeSingleTrianglesAndLines(graph, edgesDict, clusterDict, verticesDict, -1)
        # add edges that were removed by removing single points with different class value
        del closureDict
        closureDict = computeClosure(graph, edgesDict, 1, 1, 2, -3, -2)
        polygonVerticesDict = verticesDict  # compute which points lie in which cluster

        aveDistDict = computeAverageDistance(graph, edgesDict)

        # compute areas of all found clusters
        #areaDict = computeAreas(graph, edgesDict, clusterDict, verticesDict, closureDict, 2)

        # create a list of vertices that lie on the boundary of all clusters - used to determine the distance to the examples with different class
        allOtherClass = []
        for key in otherClassDict.keys():
            allOtherClass += otherClassDict[key]
        allOtherClass.sort()
        for i in range(len(allOtherClass)-1)[::-1]:
            if allOtherClass[i] == allOtherClass[i+1]: allOtherClass.remove(allOtherClass[i])

        valueDict = {}
        otherDict = {}
        enlargedClosureDict = {}
        for key in closureDict.keys():
            if len(polygonVerticesDict[key]) < 6: continue            # if the cluster has less than 6 points ignore it
            points = len(polygonVerticesDict[key]) - len(closureDict[key])    # number of points in the interior
            if points < 2: continue                      # ignore clusters that don't have at least 2 points in the interior of the cluster
            points += len(closureDict[key]) #* 0.8        # points on the edge only contribute a little less - punishment for very complex boundaries
            if points < 5: continue                      # ignore too small clusters

            # compute the center of the current cluster
            xAve = sum([graph.objects[i][0] for i in polygonVerticesDict[key]]) / len(polygonVerticesDict[key])
            yAve = sum([graph.objects[i][1] for i in polygonVerticesDict[key]]) / len(polygonVerticesDict[key])

            # and compute the average distance of 3 nearest points that belong to different class to this center
            diffClass = []
            for v in allOtherClass:
                if graph.objects[v].getclass() == graph.objects[i].getclass(): continue # ignore examples with the same class value
                d = sqrt((graph.objects[v][0]-xAve)*(graph.objects[v][0]-xAve) + (graph.objects[v][1]-yAve)*(graph.objects[v][1]-yAve))
                diffClass.append(d)
            diffClass.sort()
            dist = sum(diffClass[:5]) / float(len(diffClass[:5]))

            """
            # one way of computing the value
            area = sqrt(sqrt(areaDict[key]))
            if area > 0: value = points * dist / area
            else: value = 0
            """

            # another way of computing value
            #value = points * dist / aveDistDict[key]

            if self.distributionScale:
                d = orange.Distribution(graph.objects.domain.classVar, graph.objects)
                v = d[graph.objects[polygonVerticesDict[key][0]].getclass()]
                if v == 0: continue
                points *= sum(d) / float(v)   # turn the number of points into a percentage of all points that belong to this class value and then multiply by the number of all data points in the data set

            # and another
            #dist = sqrt(dist*1000.0)/sqrt(aveDistDict[key]*1000.0)
            dist = sqrt(dist*1000.0)
            value = points
            if self.considerDistance: value *= dist

            valueDict[key] = value
            #enlargedClosureDict[key] = enlargeClosure(graph, closureDict[key], aveDistDict[key])
            enlargedClosureDict[key] = []

            #otherDict[key] = (graph.objects[polygonVerticesDict[key][0]].getclass(), value, points, dist, area)
            #otherDict[key] = (graph.objects[polygonVerticesDict[key][0]].getclass().value, value, points, dist, aveDistDict[key])
            otherDict[key] = (int(graph.objects[polygonVerticesDict[key][0]].getclass()), value, points, dist, (xAve, yAve), aveDistDict[key])

        del edgesDict, clusterDict, verticesDict, allOtherClass
        for key in closureDict.keys():
            if not otherDict.has_key(key):
                if closureDict.has_key(key): closureDict.pop(key)
                if polygonVerticesDict.has_key(key): polygonVerticesDict.pop(key)
                if enlargedClosureDict.has_key(key): enlargedClosureDict.pop(key)
        return graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict


    # for each point in the data set compute how often does if appear inside a cluster
    # for each point then return a float between 0 and 1
    def evaluatePointsInClusters(self):
        if self.clusterStabilityButton.isChecked():
            self.pointStability = numpy.zeros(len(self.rawData), numpy.float)
            self.pointStabilityCount = [0 for i in range(len(self.rawData.domain.classVar.values))]       # for each class value create a counter that will count the number of clusters for it
            
            (text, ok) = QInputDialog.getText('Projection count', 'How many of the best projections do you want to consider?')
            if not ok: return
            nrOfProjections = int(str(text))

            considered = 0
            for i in range(len(self.allResults)):
                if considered > nrOfProjections: break
                if type(self.allResults[i][CLASS]) != dict: continue    # ignore all projections except the ones that show all clusters in the picture
                considered += 1
                clusterClasses = [0 for j in range(len(self.rawData.domain.classVar.values))]
                for key in self.allResults[i][VERTICES].keys():
                    clusterClasses[self.allResults[i][CLASS][key]] = 1
                    vertices = self.allResults[i][VERTICES][key]
                    validData = self.graph.getValidList([self.graph.attributeNameIndex[self.allResults[i][ATTR_LIST][0]], self.graph.attributeNameIndex[self.allResults[i][ATTR_LIST][1]]])
                    indices = numpy.compress(validData, numpy.array(range(len(self.rawData))), axis = 1)
                    indicesToOriginalTable = numpy.take(indices, vertices)
                    tempArray = numpy.zeros(len(self.rawData))
                    numpy.put(tempArray, indicesToOriginalTable, numpy.ones(len(indicesToOriginalTable)))
                    self.pointStability += tempArray
                for j in range(len(clusterClasses)):        # some projections may contain more clusters of the same class. we make sure we don't process this wrong
                    self.pointStabilityCount[j] += clusterClasses[j]
        
            for i in range(len(self.rawData)):
                if self.pointStabilityCount[int(self.rawData[i].getclass())] != 0:
                    self.pointStability[i] /= float(self.pointStabilityCount[int(self.rawData[i].getclass())])

        #self.pointStability = [1.0 - val for val in self.pointStability]
        if self.parentWidget: self.parentWidget.showSelectedCluster()


    def updateShownProjections(self, *args):
        self.resultList.clear()
        self.shownResults = []
        self.selectedClasses = self.getSelectedClassValues()

        i = 0
        while self.resultList.count() < self.resultListLen and i < len(self.allResults):
            if self.attrLenDict[len(self.allResults[i][ATTR_LIST])] != 1: i+=1; continue
            if type(self.allResults[i][CLASS]) == dict and -1 not in self.selectedClasses: i+=1; continue
            if type(self.allResults[i][CLASS]) == int and self.allResults[i][CLASS] not in self.selectedClasses: i+=1; continue

            string = ""
            if self.showRank: string += str(i+1) + ". "
            if self.showValue: string += "%.2f - " % (self.allResults[i][VALUE])

            if self.allResults[i][STR_LIST] != "": string += self.allResults[i][STR_LIST]
            else: string += self.buildAttrString(self.allResults[i][ATTR_LIST])

            self.resultList.addItem(string)
            self.shownResults.append(self.allResults[i])
            i+=1
        if self.resultList.count() > 0: self.resultList.setCurrentItem(self.resultList.item(0))


    # save input dataset, get possible class values, ...
    def setData(self, data, clearResults = 1):
        if hasattr(data, "name"): self.datasetName = data.name
        else: self.datasetName = ""
        sameDomain = 0
        if not clearResults and self.rawData and data and self.rawData.domain == data.domain: sameDomain = 1
        self.rawData = data
        self.clearArguments()
        if not sameDomain: self.clearResults()

        if not data or not (data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete):
            self.selectedClasses = []
            self.classesList.clear()
            self.classValueList.clear()
            return

        if not sameDomain:
            self.classesList.clear()
            self.classValueList.clear()
            self.selectedClasses = []

            # add class values
            for i in range(len(data.domain.classVar.values)):
                self.classesList.addItem(data.domain.classVar.values[i])
                self.classValueList.addItem(data.domain.classVar.values[i])
            self.classesList.addItem("All clusters")
            self.classesList.selectAll()
            if len(data.domain.classVar.values) > 0: self.classValueList.setCurrentIndex(0)

    # save subsetdata. first example from this dataset can be used with argumentation - it can find arguments for classifying the example to the possible class values
    def setSubsetData(self, subsetdata):
        self.subsetdata = subsetdata
        self.clearArguments()


    # given a dataset return a list of (val, attrName) where val is attribute "importance" and attrName is name of the attribute
    def getEvaluatedAttributes(self, data):
        self.setStatusBarText("Evaluating attributes...")
        attrs = orngVisFuncts.evaluateAttributes(data, contMeasures[self.attrCont][1], discMeasures[self.attrDisc][1])
        self.setStatusBarText("")
        return attrs

    def addResult(self, value, closure, vertices, attrList, classValue, enlargedClosure, other, strList = ""):
        self.insertItem(self.findTargetIndex(value), value, closure, vertices, attrList, classValue, enlargedClosure, other, strList)
        qApp.processEvents()        # allow processing of other events

    # use bisection to find correct index
    def findTargetIndex(self, value):
        top = 0; bottom = len(self.allResults)

        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if max(value, self.allResults[mid][VALUE]) == value: bottom = mid
            else: top = mid

        if len(self.allResults) == 0: return 0
        if max(value, self.allResults[top][VALUE]) == value:
            return top
        else:
            return bottom

    # insert new result - give parameters: value of the cluster, closure, list of attributes.
    # parameter strList can be a pre-formated string containing attribute list (used by polyviz)
    def insertItem(self, index, value, closure, vertices, attrList, classValue, enlargedClosure, other, strList = ""):
        if index < self.maxResultListLen:
            self.allResults.insert(index, (value, closure, vertices, attrList, classValue, enlargedClosure, other, strList))
            if len(self.allResults) > self.maxResultListLen: self.allResults.pop()
        if index < self.resultListLen:
            string = ""
            if self.showRank: string += str(index+1) + ". "
            if self.showValue: string += "%.2f - " % (value)

            if strList != "": string += strList
            else: string += self.buildAttrString(attrList)

            self.resultList.insertItem(index, string)
            self.shownResults.insert(index, (value, closure, vertices, attrList, classValue, enlargedClosure, other, strList))

        # remove worst projection if list is too long
        if self.resultList.count() > self.resultListLen:
            self.resultList.takeItem(self.resultList.count()-1)
            self.shownResults.pop()

    def finishedAddingResults(self):
        self.cancelOptimization = 0

        self.attrLenList.clear()
        self.attrLenDict = {}
        maxLen = -1
        for i in range(len(self.shownResults)):
            if len(self.shownResults[i][ATTR_LIST]) > maxLen:
                maxLen = len(self.shownResults[i][ATTR_LIST])
        if maxLen == -1: return
        if maxLen == 2: vals = [2]
        else: vals = range(3, maxLen+1)
        for val in vals:
            self.attrLenList.addItem(str(val))
            self.attrLenDict[val] = 1
        self.attrLenList.selectAll()
        self.resultList.setCurrentItem(self.resultList.item(0))


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
            qname = QFileDialog.getSaveFileName( self.lastSaveDirName + "/" + filename, "Interesting clusters (*.clu)", self, "", "Save Clusters")
            if qname.isEmpty(): return
            name = str(qname)
        else:
            name = filename

        # take care of extension
        if os.path.splitext(name)[1] != ".clu":
            name = name + ".clu"

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        # open, write and save file
        file = open(name, "wt")
        attrs = ["resultListLen", "parentName"]
        Dict = {}
        for attr in attrs: Dict[attr] = self.__dict__[attr]
        file.write("%s\n" % (str(Dict)))
        file.write("%s\n" % str(self.selectedClasses))
        file.write("%d\n" % self.rawData.checksum())
        for (value, closure, vertices, attrList, classValue, enlargedClosure, other, strList) in self.shownResults:
            if type(classValue) != dict: continue
            s = "(%s, %s, %s, %s, %s, %s, %s, '%s')\n" % (str(value), str(closure), str(vertices), str(attrList), str(classValue), str(enlargedClosure), str(other), strList)
            file.write(s)
        file.flush()
        file.close()


    # load projections from a file
    def load(self):
        self.clearResults()
        self.clearArguments()
        if self.rawData == None:
            QMessageBox.critical(None,'Load','There is no data. First load a data set and then load a cluster file',QMessageBox.Ok)
            return

        name = QFileDialog.getOpenFileName( self.lastSaveDirName, "Interesting clusters (*.clu)", self, "", "Open Clusters")
        if name.isEmpty(): return
        name = str(name)

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        file = open(name, "rt")
        settings = eval(file.readline()[:-1])
        if settings.has_key("parentName") and settings["parentName"] != self.parentName:
            QMessageBox.critical( None, "Cluster Dialog", 'Unable to load cluster file. It was saved for %s method'%(settings["parentName"]), QMessageBox.Ok)
            file.close()
            return

        self.setSettings(settings)

        # find if it was computed for specific class values
        selectedClasses = eval(file.readline()[:-1])
        for i in range(len(self.rawData.domain.classVar.values)):
            self.classesList.item(i).setSelected(i in selectedClasses)
        if -1 in selectedClasses: self.classesList.item(self.classesList.count()-1).setSelected(1)
        checksum = eval(file.readline()[:-1])
        if self.rawData and self.rawData.checksum() != checksum:
            cancel = QMessageBox.critical(None, "Load", "Currently loaded data set is different than the data set that was used for computing these projections. \nThere may be differences in the number of examples or in actual data values. \nThe shown clusters will therefore most likely show incorrect information. \nAre you sure you want to continue with loading?", 'Yes','No', '', 1,0)
            if cancel: return

        line = file.readline()[:-1];
        while (line != ""):
            (value, closure, vertices, attrList, classValue, enlargedClosure, other, strList) = eval(line)
            self.addResult(value, closure, vertices, attrList, classValue, enlargedClosure, other, strList)
            for key in classValue.keys():
                self.addResult(other[key][1], closure[key], vertices[key], attrList, classValue[key], enlargedClosure[key], other[key], strList)
            line = file.readline()[:-1]
        file.close()

        # update loaded results
        self.finishedAddingResults()


    # disable all controls while evaluating projections
    def disableControls(self):
        self.optimizationTypeCombo.setEnabled(0)
        self.attributeCountCombo.setEnabled(0)
        self.startOptimizationButton.hide()
        self.stopOptimizationButton.show()
        self.SettingsTab.setEnabled(0)
        self.ManageTab.setEnabled(0)
        self.ArgumentationTab.setEnabled(0)
        self.ClassificationTab.setEnabled(0)

    def enableControls(self):
        self.optimizationTypeCombo.setEnabled(1)
        self.attributeCountCombo.setEnabled(1)
        self.startOptimizationButton.show()
        self.stopOptimizationButton.hide()
        self.SettingsTab.setEnabled(1)
        self.ManageTab.setEnabled(1)
        self.ArgumentationTab.setEnabled(1)
        self.ClassificationTab.setEnabled(1)

    # ##############################################################
    # exporting multiple pictures
    # ##############################################################
    def exportMultipleGraphs(self):
        (text, ok) = QInputDialog.getText('Graph count', 'How many of the best projections do you wish to save?')
        if not ok: return
        self.bestGraphsCount = int(str(text))

        self.sizeDlg = OWDlgs.OWChooseImageSizeDlg(self.graph, parent=self)
        self.sizeDlg.disconnect(self.sizeDlg.okButton, SIGNAL("clicked()"), self.sizeDlg.accept)
        self.sizeDlg.connect(self.sizeDlg.okButton, SIGNAL("clicked()"), self.saveToFileAccept)
        self.sizeDlg.exec_()

    def saveToFileAccept(self):
        fileName = str(QFileDialog.getSaveFileName("Graph","Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)", None, "Save to...", "Save to..."))
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
            self.resultList.item(i-1).setSelected(1)
            self.graph.replot()
            name = fil + " (%02d)" % i + extension
            self.sizeDlg.saveToFileDirect(name, ext, size)
        QDialog.accept(self.sizeDlg)

    """
    def interactionAnalysis(self):
        from OWkNNOptimization import OWInteractionAnalysis, CLUSTER_POINT
        dialog = OWInteractionAnalysis(self, signalManager = self.signalManager)
        dialog.setResults(self.shownResults, CLUSTER_POINT)
        dialog.show()

    def graphProjectionQuality(self):
        from OWkNNOptimization import OWGraphProjectionQuality, CLUSTER_POINT
        dialog = OWGraphProjectionQuality(self, signalManager = self.signalManager)
        dialog.setResults(self.allResults, CLUSTER_POINT)
        dialog.show()
    """



    # ######################################################
    # Auxiliary functions
    # ######################################################
    def getOptimizationType(self):
        return self.optimizationType

    # from a list of attributes build a nice string with attribute names
    def buildAttrString(self, attrList):
        if len(attrList) == 0: return ""
        strList = attrList[0]
        for item in attrList[1:]:
            strList = strList + ", " + item
        return strList

    def getAllResults(self):
        return self.allResults

    def getShownResults(self):
        return self.shownResults

    def getSelectedCluster(self):
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

    # use evaluated projections to find arguments for classifying the first example in the self.subsetdata to possible classes.
    # add found arguments into the list
    # find only the number of argument that is specified in Arugment count combo in Classification tab
    def findArguments(self, selectBest = 1, showClassification = 1):
        self.cancelArgumentation = 0
        self.clearArguments()
        self.arguments = [[] for i in range(self.classValueList.count())]

        if self.subsetdata == None:
            QMessageBox.information( None, "Cluster Dialog Argumentation", 'To find arguments you first have to provide a new example that you wish to classify. \nYou can do this by sending the example to the visualization widget through the "Example Subset" signal.', QMessageBox.Ok + QMessageBox.Default)
            return
        if len(self.shownResults) == 0:
            QMessageBox.information( None, "Cluster Dialog Argumentation", 'To find arguments you first have to find clusters in some projections by clicking "Find arguments" in the Main tab.', QMessageBox.Ok + QMessageBox.Default)
            return

        example = self.subsetdata[0]    # we can find arguments only for one example. We select only the first example in the example table
        testExample = [self.parentWidget.graph.scaleExampleValue(example, i) for i in range(len(example.domain.attributes))]

        self.findArgumentsButton.hide()
        self.stopArgumentationButton.show()

        foundArguments = 0
        argumentCount = self.argumentCounts[self.argumentCountIndex]
        vals = [0.0 for i in range(len(self.arguments))]

        # ####################################################################
        # find arguments so that we evaluate best argumentCount clusters for each class value and see how often does the point lie inside them
        # this way, we don't care if clusters of one class value have much higher values than clusters of another class.
        # NOTE: argumentCount in this case represents the number of evaluated clusters and not the number of clusters where the point actually lies inside
        if self.argumentationType == BEST_GROUPS_IN_EACH_CLASS:
            classIndices = [0 for i in range(len(self.rawData.domain.classVar.values))]
            canFinish = 0
            while not canFinish:
                for cls in range(len(classIndices)):
                    startingIndex = classIndices[cls]

                    for index in range(len(self.allResults) - startingIndex):
                        (value, closure, vertices, attrList, classValue, enlargedClosure, other, strList) = self.allResults[startingIndex+index]
                        if type(classValue) == dict: continue       # the projection contains several clusters
                        if classValue != cls: continue
                        classIndices[cls] = startingIndex + index + 1   # remember where to start next

                        qApp.processEvents()
                        inside = self.isExampleInsideCluster(attrList, testExample, closure, vertices, other[OTHER_AVERAGE_DIST])
                        if inside == 0 or (inside == 1 and self.conditionForArgument == MUST_BE_INSIDE): continue
                        vals[classValue] += 1

                        self.arguments[classValue].append((None, value, attrList, startingIndex+index))
                        if classValue == self.classValueList.currentItem():
                            self.argumentList.addItem("%.2f - %s" %(value, attrList))
                    if classIndices[cls] == startingIndex: classIndices[cls] = len(self.allResults)+1

                foundArguments += 1
                if min(classIndices) >= len(self.allResults): canFinish = 1  # out of possible arguments

                if sum(vals) > 0 and foundArguments >= argumentCount:
                    if not self.canUseMoreArguments or (max(vals)*100.0)/float(sum(vals)) > self.moreArgumentsNums[self.moreArgumentsIndex]:
                        canFinish = 1
        # ####################################################################
        # we consider clusters with the highest values and check if the point lies inside them. When it does we say that this cluster is an argument for
        # that class. We continue until we find argumentCount such arguments. Then we predict the most likely class value
        else:
            for index in range(len(self.allResults)):
                if self.cancelArgumentation: break
                # we also stop if we are not allowed to search for more than argumentCount arguments or we are allowed and we have a reliable prediction or we have used a 100 additional arguments
                if foundArguments >= argumentCount and (not self.canUseMoreArguments or (max(vals)*100.0 / sum(vals) > self.moreArgumentsNums[self.moreArgumentsIndex]) or foundArguments >= argumentCount + 100): break

                (value, closure, vertices, attrList, classValue, enlargedClosure, other, strList) = self.allResults[index]
                if type(classValue) == dict: continue       # the projection contains several clusters

                qApp.processEvents()
                inside = self.isExampleInsideCluster(attrList, testExample, closure, vertices, other[OTHER_AVERAGE_DIST])
                if self.conditionForArgument == MUST_BE_INSIDE and inside != 2: continue
                elif inside == 0: continue

                foundArguments += 1  # increase argument count

                if self.useProjectionValue: vals[classValue] += value
                else: vals[classValue] += 1

                self.arguments[classValue].append((None, value, attrList, index))
                if classValue == self.classValueList.currentItem():
                    self.argumentList.addItem("%.2f - %s" %(value, attrList))

        self.stopArgumentationButton.hide()
        self.findArgumentsButton.show()
        if self.argumentList.count() > 0 and selectBest: self.argumentList.setCurrentItem(self.argumentList.item(0))
        if foundArguments == 0: return (None, None)

        suma = sum(vals)
        dist = orange.DiscDistribution([val/float(suma) for val in vals]);  dist.variable = self.rawData.domain.classVar
        classValue = self.rawData.domain.classVar[vals.index(max(vals))]

        # show classification results
        s = '<nobr>Based on current classification settings, the example would be classified </nobr><br><nobr>to class <b>%(cls)s</b> with probability <b>%(prob).2f%%</b>.</nobr><br><nobr>Predicted class distribution is:</nobr><br>' % {"cls": classValue, "prob": dist[classValue]*100}
        for key in dist.keys():
            s += "<nobr>&nbsp &nbsp &nbsp &nbsp %s : %.2f%%</nobr><br>" % (key, dist[key]*100)
        if foundArguments > argumentCount:
            s += "<nobr>Note: To get the current prediction, <b>%(fa)d</b> arguments had to be used (instead of %(ac)d)<br>" % {"fa": foundArguments, "ac": argumentCount}
        s = s[:-4]
        if showClassification:
            QMessageBox.information(None, "Classification results", s, QMessageBox.Ok + QMessageBox.Default)
        return classValue, dist

    # is the example described with scaledExampleVals inside closure that contains points in vertices in projection of attrList attributes
    def isExampleInsideCluster(self, attrList, scaledExampleVals, closure, vertices, averageEdgeDistance):
        testExampleAttrVals = [scaledExampleVals[self.graph.attributeNameIndex[attrList[i]]] for i in range(len(attrList))]
        if min(testExampleAttrVals) < 0.0 or max(testExampleAttrVals) > 1.0: return 0

        array = self.graph.createProjectionAsNumericArray([self.graph.attributeNameIndex[attr] for attr in attrList])
        if array == None:
            return 0
        short = numpy.transpose(numpy.take(array, vertices))

        [xTest, yTest] = self.graph.getProjectedPointPosition(attrList, testExampleAttrVals)

        #if xTest < min(short[0]) or xTest > max(short[0]) or yTest < min(short[1]) or yTest > max(short[1]):
        #    del array, short; return 0       # the point is definitely not inside the cluster

        val = pointInsideCluster(array, closure, xTest, yTest)
        if val:
            del array, short
            return 2

        val = 0
        points = union([i for (i,j) in closure], [j for (i,j) in closure])
        for i in range(len(points)):
            xdiff = array[points[i]][0] - xTest
            ydiff = array[points[i]][1] - yTest
            if sqrt(xdiff*xdiff + ydiff*ydiff) < averageEdgeDistance:
                val = 1
                break

        return val


    # use bisection to find correct index
    def findArgumentTargetIndex(self, value, arguments):
        top = 0; bottom = len(arguments)

        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if max(value, arguments[mid][1]) == value: bottom = mid
            else: top = mid

        if len(arguments) == 0: return 0
        if max(value, arguments[top][1]) == value:
            return top
        else:
            return bottom

    def stopArgumentationClick(self):
        self.cancelArgumentation = 1

    def argumentationClassChanged(self):
        self.argumentList.clear()
        if len(self.arguments) == 0: return
        ind = self.classValueList.currentItem()
        for i in range(len(self.arguments[ind])):
            val = self.arguments[ind][i]
            if val[0] != None:  self.argumentList.insertItem(val[0], "%.2f - %s" %(val[1], val[2]))
            else:               self.argumentList.addItem("%.2f - %s" %(val[1], val[2]))

    def argumentSelected(self):
        ind = self.argumentList.currentItem()
        classInd = self.classValueList.currentItem()
        clusterClosure = (self.allResults[self.arguments[classInd][ind][3]][CLOSURE], self.allResults[self.arguments[classInd][ind][3]][ENL_CLOSURE], self.allResults[self.arguments[classInd][ind][3]][CLASS])
        self.parentWidget.updateGraph(self.arguments[classInd][ind][2], clusterClosure = clusterClosure)


class clusterClassifier(orange.Classifier):
    def __init__(self, clusterOptimizationDlg, visualizationWidget, data, firstTime = 1):
        self.clusterOptimizationDlg = clusterOptimizationDlg
        self.visualizationWidget = visualizationWidget
        self.data = data

        results = clusterOptimizationDlg.getAllResults()
        if firstTime and results != None and len(results) > 0:
            computeProjections = QMessageBox.information(clusterOptimizationDlg, 'Cluster classifier', 'Do you want to classify examples based the projections that are currently in the projection list \n or do you want to compute new projections?','Current projections','Compute new projections', '', 0,1)
            #computeProjections = 0
        elif results != None and len(results) > 0:
            computeProjections = 0
        else: computeProjections = 1

        self.visualizationWidget.setData(data, clearResults = computeProjections)
        if computeProjections == 1:     # run cluster detection
            self.evaluating = 1
            t = QTimer(self.visualizationWidget)
            self.visualizationWidget.connect(t, SIGNAL("timeout()"), self.stopEvaluation)
            t.start(self.clusterOptimizationDlg.evaluationTimeNums[self.clusterOptimizationDlg.evaluationTimeIndex] * 60 * 1000, 1)
            self.visualizationWidget.optimizeClusters()
            t.stop()
            self.evaluating = 0

    # timer event that stops evaluation of clusters
    def stopEvaluation(self):
        if self.evaluating:
            self.clusterOptimizationDlg.stopOptimizationClick()


    # for a given example run argumentation and find out to which class it most often falls
    def __call__(self, example, returnType):
        testExample = [self.visualizationWidget.graph.scaleExampleValue(example, i) for i in range(len(example.domain.attributes))]

        argumentCount = 0
        argumentValues = []
        allowedArguments = self.clusterOptimizationDlg.argumentCounts[self.clusterOptimizationDlg.argumentCountIndex]

        classProjectionVals = [0 for i in range(len(self.clusterOptimizationDlg.rawData.domain.classVar.values))]
        classIndices = [0 for i in range(len(self.clusterOptimizationDlg.rawData.domain.classVar.values))]
        if self.clusterOptimizationDlg.argumentationType == BEST_GROUPS_IN_EACH_CLASS:
            canFinish = 0
            while not canFinish:
                for cls in range(len(classIndices)):
                    startingIndex = classIndices[cls]

                    for index in range(len(self.clusterOptimizationDlg.allResults) - startingIndex):
                        (value, closure, vertices, attrList, classValue, enlargedClosure, other, strList) = self.clusterOptimizationDlg.allResults[startingIndex+index]
                        if type(classValue) == dict: continue       # the projection contains several clusters
                        if classValue != cls: continue
                        classIndices[cls] = startingIndex + index + 1   # remember where to start next

                        qApp.processEvents()
                        attrIndices = [self.visualizationWidget.graph.attributeNameIndex[attr] for attr in attrList]
                        data = self.visualizationWidget.graph.createProjectionAsExampleTable(attrIndices, jitterSize = 0.001 * self.clusterOptimizationDlg.jitterDataBeforeTriangulation)
                        graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict = self.clusterOptimizationDlg.evaluateClusters(data)
                        for key in valueDict.keys():
                            if classValue != otherDict[key][OTHER_CLASS]: continue
                            inside = self.clusterOptimizationDlg.isExampleInsideCluster(attrList, testExample, closureDict[key], polygonVerticesDict[key], otherDict[key][OTHER_AVERAGE_DIST])
                            if inside == 0 or (inside == 1 and self.clusterOptimizationDlg.conditionForArgument == MUST_BE_INSIDE): continue
                            classProjectionVals[classValue] += 1
                        break
                    if classIndices[cls] == startingIndex: classIndices[cls] = len(self.clusterOptimizationDlg.allResults)+1

                argumentCount += 1
                if min(classIndices) >= len(self.clusterOptimizationDlg.allResults): canFinish = 1  # out of possible arguments

                if sum(classProjectionVals) > 0 and argumentCount >= allowedArguments:
                    if not self.clusterOptimizationDlg.canUseMoreArguments or (max(classProjectionVals)*100.0)/float(sum(classProjectionVals)) > self.clusterOptimizationDlg.moreArgumentsNums[self.clusterOptimizationDlg.moreArgumentsIndex]:
                        canFinish = 1

            if max(classProjectionVals) == 0:
                print "there are no arguments for this example in the current projection list."
                dist = orange.DiscDistribution([1/float(len(classProjectionVals)) for i in classProjectionVals]); dist.variable = self.clusterOptimizationDlg.rawData.domain.classVar
                return (self.clusterOptimizationDlg.rawData.domain.classVar[0], dist)

            ind = classProjectionVals.index(max(classProjectionVals))
            s = sum(classProjectionVals)
            dist = orange.DiscDistribution([val/float(s) for val in classProjectionVals]);  dist.variable = self.clusterOptimizationDlg.rawData.domain.classVar

            classValue = self.clusterOptimizationDlg.rawData.domain.classVar[ind]
            s = '<nobr>Based on current classification settings, the example would be classified </nobr><br><nobr>to class <b>%(cls)s</b> with probability <b>%(prob).2f%%</b>.</nobr><br><nobr>Predicted class distribution is:</nobr><br>' % {"cls": classValue, "prob": dist[classValue]*100}
            for key in dist.keys():
                s += "<nobr>&nbsp &nbsp &nbsp &nbsp %s : %.2f%%</nobr><br>" % (key, dist[key]*100)
            if argumentCount > allowedArguments:
                s += "<nobr>Note: To get the current prediction, <b>%(fa)d</b> arguments had to be used (instead of %(ac)d)<br>" % {"fa": argumentCount, "ac": allowedArguments}
            print s[:-4]

            return (classValue, dist)
        else:
            for (value, closure, vertices, attrList, classValue, enlargedClosure, other, strList) in self.clusterOptimizationDlg.allResults:
                if type(classValue) != dict: continue       # the projection contains several clusters
                qApp.processEvents()

                attrIndices = [self.visualizationWidget.graph.attributeNameIndex[attr] for attr in attrList]
                data = self.visualizationWidget.graph.createProjectionAsExampleTable(attrIndices, jitterSize = 0.001 * self.clusterOptimizationDlg.jitterDataBeforeTriangulation)
                graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict = self.clusterOptimizationDlg.evaluateClusters(data)
                evaluation = []
                for key in valueDict.keys():
                    evaluation.append((self.clusterOptimizationDlg.isExampleInsideCluster(attrList, testExample, closureDict[key], polygonVerticesDict[key], otherDict[key][OTHER_AVERAGE_DIST]), int(graph.objects[polygonVerticesDict[key][0]].getclass()), valueDict[key]))

                evaluation.sort(); evaluation.reverse()
                if len(evaluation) == 0 or (len(evaluation) > 0 and evaluation[0][0] == 0): continue # if the point wasn't in any of the clusters or near them then continue
                elif len(evaluation) > 0 and evaluation[0][0] == 1 and self.clusterOptimizationDlg.conditionForArgument == MUST_BE_INSIDE: continue
                elif len(evaluation) == 1 or evaluation[0][0] != evaluation[1][0]:
                    argumentCount += 1
                    argumentValues.append((evaluation[0][2], evaluation[0][1]))

                # find 10 more arguments than it is necessary - this is because with a different fold of data the clusters can be differently evaluated
                if argumentCount >= self.clusterOptimizationDlg.argumentCounts[self.clusterOptimizationDlg.argumentCountIndex]:
                    argumentValues.sort(); argumentValues.reverse()
                    vals = [0.0 for i in range(len(self.clusterOptimizationDlg.rawData.domain.classVar.values))]
                    neededArguments = self.clusterOptimizationDlg.argumentCounts[self.clusterOptimizationDlg.argumentCountIndex]
                    sufficient = 0; consideredArguments = 0
                    for (val, c) in argumentValues:
                        if self.clusterOptimizationDlg.useProjectionValue:  vals[c] += val
                        else:                                               vals[c] += 1
                        consideredArguments += 1
                        if consideredArguments >= neededArguments and (not self.clusterOptimizationDlg.canUseMoreArguments or (max(vals)*100.0 / sum(vals) > self.clusterOptimizationDlg.moreArgumentsNums[self.clusterOptimizationDlg.moreArgumentsIndex]) or consideredArguments >= neededArguments + 30):
                            sufficient = 1
                            break

                    # if we got enough arguments to make a prediction then do it
                    if sufficient:
                        ind = vals.index(max(vals))
                        s = sum(vals)
                        dist = orange.DiscDistribution([val/float(s) for val in vals]);  dist.variable = self.clusterOptimizationDlg.rawData.domain.classVar

                        classValue = self.clusterOptimizationDlg.rawData.domain.classVar[ind]
                        s = '<nobr>Based on current classification settings, the example would be classified </nobr><br><nobr>to class <b>%(cls)s</b> with probability <b>%(prob).2f%%</b>.</nobr><br><nobr>Predicted class distribution is:</nobr><br>' % {"cls": classValue, "prob": dist[classValue]*100}
                        for key in dist.keys():
                            s += "<nobr>&nbsp &nbsp &nbsp &nbsp %s : %.2f%%</nobr><br>" % (key, dist[key]*100)
                        if consideredArguments > neededArguments:
                            s += "<nobr>Note: To get the current prediction, <b>%(fa)d</b> arguments had to be used (instead of %(ac)d)<br>" % {"fa": consideredArguments, "ac": neededArguments}
                        print s[:-4]

                        return (classValue, dist)

            # if we ran out of projections before we could get a reliable prediction use what we have
            vals = [0.0 for i in range(len(self.clusterOptimizationDlg.rawData.domain.classVar.values))]
            argumentValues.sort(); argumentValues.reverse()
            for (val, c) in argumentValues:
                if self.clusterOptimizationDlg.useProjectionValue:  vals[c] += val
                else:                                               vals[c] += 1

            if max(vals) == 0.0:
                print "there are no arguments for this example in the current projection list."
                dist = orange.DiscDistribution([1/float(len(vals)) for i in range(len(vals))]); dist.variable = self.clusterOptimizationDlg.rawData.domain.classVar
                return (self.clusterOptimizationDlg.rawData.domain.classVar[0], dist)

            ind = vals.index(max(vals))
            s = sum(vals)
            dist = orange.DiscDistribution([val/float(s) for val in vals]);  dist.variable = self.clusterOptimizationDlg.rawData.domain.classVar

            classValue = self.clusterOptimizationDlg.rawData.domain.classVar[ind]
            s = '<nobr>Based on current classification settings, the example would be classified </nobr><br><nobr>to class <b>%(cls)s</b> with probability <b>%(prob).2f%%</b>.</nobr><br><nobr>Predicted class distribution is:</nobr><br>' % {"cls": classValue, "prob": dist[classValue]*100}
            for key in dist.keys():
                s += "<nobr>&nbsp &nbsp &nbsp &nbsp %s : %.2f%%</nobr><br>" % (key, dist[key]*100)
            s += "<nobr>Note: There were not enough projections to get a reliable prediction<br>"
            print s[:-4]
            return (classValue, dist)


class clusterLearner(orange.Learner):
    def __init__(self, clusterOptimizationDlg, visualizationWidget):
        self.clusterOptimizationDlg = clusterOptimizationDlg
        self.visualizationWidget = visualizationWidget
        self.name = self.clusterOptimizationDlg.parentWidget.clusterClassifierName
        self.firstTime = 1

    def __call__(self, examples, weightID = 0):
        classifier = clusterClassifier(self.clusterOptimizationDlg, self.visualizationWidget, examples, self.firstTime)
        self.firstTime = 0
        return classifier


# ############################################################################################################################################
# ############################################################################################################################################


# compute union of two lists
def union(list1, list2):
    union_dict = {}
    for e in list1: union_dict[e] = 1
    for e in list2: union_dict[e] = 1
    return union_dict.keys()

# compute intersection of two lists
def intersection(list1, list2):
    int_dict = {}
    list1_dict = {}
    for e in list1:
        list1_dict[e] = 1
    for e in list2:
        if list1_dict.has_key(e): int_dict[e] = 1
    return int_dict.keys()

# on which side of the line through (x1,y1), (x2,y2) lies the point (xTest, yTest)
def getPosition(x1, y1, x2, y2, xTest, yTest):
    if x1 == x2:
        if xTest > x1: return 1
        else: return -1
    k = (y2-y1)/float(x2-x1)
    c = y1 - k*x1
    if k*xTest - yTest > -c : return 1
    else: return -1

# compute distances between all connected vertices and store it in the DISTANCE field
def computeDistances(graph):
    for (i,j) in graph.getEdges():
        xdiff = graph.objects[i][0] - graph.objects[j][0]
        ydiff = graph.objects[i][1] - graph.objects[j][1]
        graph[i,j][DISTANCE] = sqrt(xdiff*xdiff + ydiff*ydiff)


# Slightly deficient function to determine if the two lines p1, p2 and p2, p3 turn in counter clockwise direction
def ccw(p1, p2, p3):
    dx1 = p2[0] - p1[0]; dy1 = p2[1] - p1[1]
    dx2 = p3[0] - p2[0]; dy2 = p3[1] - p2[1]
    if(dy1*dx2 < dy2*dx1): return 1
    else: return 0


# do lines with (p1,p2) and (p3,p4) intersect?
def lineIntersect(p1, p2, p3, p4):
    return ((ccw(p1, p2, p3) != ccw(p1, p2, p4)) and (ccw(p3, p4, p1) != ccw(p3, p4, p2)))

# compute distance between points (x1,y1), (x2, y2)
def computeDistance(x1, y1, x2, y2):
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

# remove edges to different class values from the graph. set VALUE field of such edges to newVal
def removeEdgesToDifferentClasses(graph, newVal):
    if newVal == None:
        for (i,j) in graph.getEdges():
            if graph.objects[i].getclass() != graph.objects[j].getclass(): graph[i,j] = None
    else:
        for (i,j) in graph.getEdges():
            if graph.objects[i].getclass() != graph.objects[j].getclass(): graph[i,j][VALUE] = newVal


# for a given dictionary that contains edges on the closure, return a dictionary with these points as keys and a list of points with a different class value as the value
# this is used when we have to compute the nearest points that belong to a different class
def getPointsWithDifferentClassValue(graph, closureDict):
    differentClassPoinsDict = {}
    for key in closureDict:
        vertices = union([i for (i,j) in closureDict[key]], [j for (i,j) in closureDict[key]])
        points = []
        for v in vertices:
            for n in graph.getNeighbours(v):
                if graph.objects[v].getclass() == graph.objects[n].getclass() or n in points: continue
                points.append(n)
        differentClassPoinsDict[key] = points
    return differentClassPoinsDict


# remove edges, that do not connect to triangles
def removeSingleLines(graph, edgesDict, clusterDict, verticesDict, minimumValidIndex, newVal):
    for (i,j) in graph.getEdges():
        if graph[i,j][VALUE] < minimumValidIndex: continue
        merged = graph.getNeighbours(i) + graph.getNeighbours(j)
        merged.sort()
        k=0; found = 0;
        for k in range(len(merged)-1):
            if merged[k] == merged[k+1] and graph[i, merged[k]][VALUE] >= minimumValidIndex and graph[j, merged[k]][VALUE] >= minimumValidIndex:
                found = 1; break
        if not found:
            if newVal == None: graph[i,j] = None            # delete the edge
            else:
                graph[i,j][VALUE] = newVal   # set the edge value
                graph[i,j][CLUSTER] = -1
            # remove the edge from the edgesDict
            if edgesDict and clusterDict and verticesDict:
                index = max(clusterDict[j], clusterDict[i])
                if index == -1: continue
                if (i,j) in edgesDict[index]: edgesDict[index].remove((i,j))
                elif (j,i) in edgesDict[index]: edgesDict[index].remove((j,i))
                if clusterDict[i] != -1: verticesDict[clusterDict[i]].remove(i)
                if clusterDict[j] != -1: verticesDict[clusterDict[j]].remove(j)
                clusterDict[j] = clusterDict[i] = -1

# remove single lines and triangles that are now treated as their own clusters because they are probably insignificant
def removeSingleTrianglesAndLines(graph, edgesDict, clusterDict, verticesDict, newVal):
    for key in edgesDict.keys():
        if len(verticesDict[key]) < 4:
            for (i,j) in edgesDict[key]:
                graph[i,j][VALUE] = newVal
                graph[i,j][CLUSTER] = -1
                clusterDict[i] = -1; clusterDict[j] = -1;
            verticesDict.pop(key)
            edgesDict.pop(key)

# make clusters smaller by removing points on the hull that lie closer to points in opposite class
def removeDistantPointsFromClusters(graph, edgesDict, clusterDict, verticesDict, closureDict, newVal):
    for key in closureDict.keys():
        edges = closureDict[key]
        vertices = union([i for (i,j) in edges], [j for (i,j) in edges])

        for vertex in vertices:
            correctClass = int(graph.objects[vertex].getclass())
            correct = []; incorrect = []
            for n in graph.getNeighbours(vertex):
                if int(graph.objects[n].getclass()) == correctClass: correct.append(graph[vertex,n][DISTANCE])
                else: incorrect.append(graph[vertex,n][DISTANCE])

            # if the distance to the correct class value is greater than to the incorrect class -> remove the point from the
            if correct == [] or (len(incorrect) > 0 and min(correct) > min(incorrect)):
                # if closure will degenerate into a line -> remove the remaining vertices
                correctClassNeighbors = []
                for n in graph.getNeighbours(vertex):
                    if int(graph.objects[n].getclass()) == correctClass and graph[vertex,n][CLUSTER] != -1: correctClassNeighbors.append(n)
                for i in correctClassNeighbors:
                    for j in correctClassNeighbors:
                        if i==j: continue
                        if (i,j) in edges:
                            graph[i,j][CLUSTER] = -1
                            if newVal == None:  graph[i,j] = None
                            else:               graph[i,j][VALUE] = newVal
                            if i in verticesDict[key]: verticesDict[key].remove(i)
                            if j in verticesDict[key]: verticesDict[key].remove(j)
                            clusterDict[i] = -1; clusterDict[j] = -1
                            if (i,j) in edgesDict[key]: edgesDict[key].remove((i,j))
                            else: edgesDict[key].remove((j,i))
                            edges.remove((i,j))

                # remove the vertex from the closure
                for n in graph.getNeighbours(vertex):
                    if int(graph.objects[n].getclass()) != correctClass or graph[vertex,n][CLUSTER] == -1: continue
                    if newVal == None:  graph[vertex,n] = None
                    else:               graph[vertex,n][VALUE] = newVal
                    if (n, vertex) in edgesDict[graph[vertex,n][CLUSTER]]: edgesDict[graph[vertex,n][CLUSTER]].remove((n,vertex))
                    elif (vertex, n) in edgesDict[graph[vertex,n][CLUSTER]]: edgesDict[graph[vertex,n][CLUSTER]].remove((vertex, n))
                    graph[vertex,n][CLUSTER] = -1
                if clusterDict[vertex] != -1: verticesDict[key].remove(vertex)
                #verticesDict[clusterDict[vertex]].remove(vertex)
                clusterDict[vertex] = -1


# mark edges on the closure with the outerEdgeValue and edges inside a cluster with a innerEdgeValue
# algorithm: for each edge, find all triangles that contain this edge. if the number of triangles is 1, then this edge is on the closure.
# if there is a larger number of triangles, then they must all lie on the same side of this edge, otherwise this edge is not on the closure
def computeClosure(graph, edgesDict, minimumValidIndex, innerEdgeValue, outerEdgeValue, differentClassValue, tooDistantValue):
    closureDict = {}
    for key in edgesDict.keys(): closureDict[key] = []  # create dictionary where each cluster will contain all edges that lie on the closure
    for (i,j) in graph.getEdges():
        if graph[i,j][VALUE] < minimumValidIndex or graph[i,j][CLUSTER] == -1: continue
        merged = intersection(graph.getNeighbours(i), graph.getNeighbours(j))
        sameClassPoints = []; otherClassPoints = []
        for v in merged:
            if graph[i, v][VALUE] >= minimumValidIndex and graph[j, v][VALUE] >= minimumValidIndex: sameClassPoints.append(v)
            elif graph[i, v][VALUE] == graph[j, v][VALUE] == differentClassValue: otherClassPoints.append(v)
            elif graph[i,v][VALUE] == tooDistantValue or graph[j,v][VALUE] == tooDistantValue:
                for n in graph.getNeighbours(v):
                    if graph[v,n][VALUE] == differentClassValue and n not in otherClassPoints: otherClassPoints.append(n)
        outer = 1; outer2 = 0
        if sameClassPoints == []: outer = 0
        elif len(sameClassPoints) > 0:
            dir = getPosition(graph.objects[i][0].value, graph.objects[i][1].value, graph.objects[j][0].value, graph.objects[j][1].value, graph.objects[sameClassPoints[0]][0].value, graph.objects[sameClassPoints[0]][1].value)
            for val in sameClassPoints[1:]:
                dir2 = getPosition(graph.objects[i][0].value, graph.objects[i][1].value, graph.objects[j][0].value, graph.objects[j][1].value, graph.objects[val][0].value, graph.objects[val][1].value)
                if dir != dir2: outer = 0; break
            if otherClassPoints != []:   # test if a point from a different class is lying inside one of the triangles
                for o in otherClassPoints:
                    val = 0; nearerPoint = 0
                    for s in sameClassPoints:
                        # check if o is inside (i,j,s) triangle
                        val += pointInsideCluster (graph.objects, [(i,j), (j,s), (s,i)], graph.objects[o][0].value, graph.objects[o][1].value)
                        # check if another point from sameClassPoints is inside (i,j,o) triangle - if it is, then (i,j) definitely isn't on the closure
                        nearerPoint += pointInsideCluster(graph.objects, [(i,j), (j,o), (o,i)], graph.objects[s][0].value, graph.objects[s][1].value)
                    if val > 0 and nearerPoint == 0: outer2 = 1
        if outer + outer2 == 1:      # if outer is 0 or more than 1 than it is not an outer edge
            graph[i,j][VALUE] = outerEdgeValue
            closureDict[graph[i,j][CLUSTER]].append((i,j))
        else:
            graph[i,j][VALUE] = innerEdgeValue
    return closureDict

# produce a dictionary with edges, clusters and vertices. clusters are enumerated from 1 up. Each cluster has a different value.
# minimumValidValue - edges that have smaller value than this will be ignored and will represent boundary between different clusters
def enumerateClusters(graph, minimumValidValue):
    clusterIndex = 1
    verticesDict = {}
    clusterDict = {}
    edgesDict = {}
    for (i,j) in graph.getEdges(): graph[i,j][CLUSTER] = -1   # initialize class cluster
    for i in range(graph.nVertices): clusterDict[i] = -1

    for i in range(graph.nVertices):
        if graph.getNeighbours(i) == [] or clusterDict[i] != -1: continue
        edgesDict[clusterIndex] = []
        verticesDict[clusterIndex] = []
        verticesToSearch = [i]
        while verticesToSearch != []:
            current = verticesToSearch.pop()
            if clusterDict[current] != -1: continue
            clusterDict[current] = clusterIndex
            verticesDict[clusterIndex].append(current)
            for n in graph.getNeighbours(current):
                if graph[current,n][VALUE] < minimumValidValue: continue
                if clusterDict[n] == -1:
                    verticesToSearch.append(n)
                    edgesDict[clusterIndex].append((n, current))
                    graph[current, n][CLUSTER] = clusterIndex
        clusterIndex += 1
    return (edgesDict, clusterDict, verticesDict, clusterIndex-1)

# for edges that have too small number of vertices set edge value to insignificantEdgeValue
# for edges that fall apart because of the distance between points set value to removedEdgeValue
def computeAlphaShapes(graph, edgesDict, alphaShapesValue, insignificantEdgeValue, removedEdgeValue):
    for key in edgesDict.keys():
        edges = edgesDict[key]
        if len(edges) < 5:
            for (i,j) in edges: graph[i,j][VALUE] = insignificantEdgeValue
            continue

        # remove edges that are of lenghts average + alphaShapesValue * standard deviation
        lengths = [graph[i,j][DISTANCE] for (i,j) in edges]
        ave = sum(lengths) / len(lengths)
        std = sqrt(sum([(x-ave)*(x-ave) for x in lengths])/len(lengths))
        allowedDistance = ave + alphaShapesValue * std
        for index in range(len(lengths)):
            if lengths[index] > allowedDistance: graph[edges[index][0], edges[index][1]][VALUE] = removedEdgeValue


# alphashapes removed some edges. if after clustering two poins of the edge still belong to the same cluster, restore the deleted edges
def fixDeletedEdges(graph, edgesDict, clusterDict, deletedEdgeValue, repairValue, deleteValue):
    for (i,j) in graph.getEdges():
        if graph[i,j][VALUE] == deletedEdgeValue:
            if clusterDict[i] == clusterDict[j]:           # points still belong to the same cluster. restore the values
                graph[i,j][VALUE] = repairValue            # restore the value of the edge
                graph[i,j][CLUSTER] = clusterDict[i]       # reset the edge value
                edgesDict[clusterDict[i]].append((i,j))    # re-add the edge to the list of edges
            else:       # points belong to a different clusters. delete the values
                if deleteValue == None: graph[i,j] = None
                else:                   graph[i,j][VALUE] = deleteValue


# compute the area of the polygons that are not necessarily convex
# can be used to measure how good one cluster is. Currently is is not used
def computeAreas(graph, edgesDict, clusterDict, verticesDict, closureDict, outerEdgeValue):
    areaDict = {}

    for key in closureDict.keys():
        # first check if the closure is really a closure, by checking if it is connected into a circle
        # if it is not, remove this closure from all dictionaries
        vertices = [i for (i,j) in closureDict[key]] + [j for (i,j) in closureDict[key]]
        vertices.sort()
        valid = 1
        if len(vertices) % 2 != 0: valid = 0
        else:
            for i in range(len(vertices)/2):
                if vertices[2*i] != vertices[2*i+1]: valid = 0; break
        if not valid:
            print "found and ignored an invalid group", graph.objects.domain[0].name, graph.objects.domain[1].name, closureDict[key]
            areaDict[key] = 0
            continue

        # then compute its area size
        currArea = 0.0
        edges = closureDict[key] # select outer edges
        coveredEdges = []
        while len(coveredEdges) < len(edges):
            for (e1, e2) in edges:
                if (e1, e2) not in coveredEdges and (e2, e1) not in coveredEdges: break
            polygons = computePolygons(graph, (e1, e2), outerEdgeValue)
            for poly in polygons:
                currArea += computeAreaOfPolygon(graph, poly)       # TO DO: currently this is incorrect. we should first check if the polygon poly lies inside one of the other polygons. if true, than the area should be subtracted
                for i in range(len(poly)-2): coveredEdges.append((poly[i], poly[i+1]))
                coveredEdges.append((poly[0], poly[-1]))
        areaDict[key] = currArea
    return areaDict

# used when computing area of polygon
# for a given graph and starting edge return a set of polygons that represent the boundaries. in the simples case there will be only one polygon.
# However there can be also subpolygons and connected polygons
# this method is unable to guaranty correct computation of polygons in case there is an ambiguity - such as in case of 3 connected triangles where each shares two vertices with the other two triangles.
def computePolygons(graph, startingEdge, outerEdgeValue):
    waitingEdges = [startingEdge]
    takenEdges = []
    polygons = []
    while waitingEdges != []:
        (start, end) = waitingEdges.pop()
        if (start, end) in takenEdges or (end,start) in takenEdges: continue
        subpath = [start]
        while end != subpath[0]:
            neighbors = graph.getNeighbours(end)
            outerEdges = []
            for n in neighbors:
                if graph[end, n][VALUE] == outerEdgeValue and n != start and (n, end) not in takenEdges and (end, n) not in takenEdges:
                    outerEdges.append(n)
            if end in subpath:
                i = subpath.index(end)
                polygons.append(subpath[i:])
                subpath = subpath[:i]
            if outerEdges == []:
                print "there is a bug in computePolygons function", graph.objects.domain, startingEdge, end, neighbors
                break
            subpath.append(end)
            takenEdges.append((start, end))
            start = end; end = outerEdges[0]
            outerEdges.remove(end)
            if len(outerEdges) > 0:
                for e in outerEdges:
                    if (start, e) not in waitingEdges and (e, start) not in waitingEdges: waitingEdges.append((start, e))
        takenEdges.append((subpath[-1], subpath[0]))
        polygons.append(subpath)
    return polygons

# fast computation of the area of the polygon
# formula can be found at http://en.wikipedia.org/wiki/Polygon
def computeAreaOfPolygon(graph, polygon):
    polygon = polygon + [polygon[0]]
    tempArea = 0.0
    for i in range(len(polygon)-1):
        tempArea += graph.objects[polygon[i]][0]*graph.objects[polygon[i+1]][1] - graph.objects[polygon[i+1]][0]*graph.objects[polygon[i]][1]
    return abs(tempArea)

# does the point lie inside or on the edge of a cluster described with edges in closure
# algorithm from computational geometry in c (Jure Zabkar)
def pointInsideCluster(data, closure, xTest, yTest):
    # test if the point is the same as one of the points in the closure
    for (p1, p2) in closure:
        if data[p1][0] == xTest and data[p1][1] == yTest: return 1
        if data[p2][0] == xTest and data[p2][1] == yTest: return 1

    count = 0
    for (p2, p3) in closure:
        x1 = data[p2][0] - xTest; y1 = data[p2][1] - yTest
        x2 = data[p3][0] - xTest; y2 = data[p3][1] - yTest
        if (y1 > 0 and y2 <= 0) or (y2 > 0 and y1 <= 0):
            x = (x1 * y2 - x2 * y1) / (y2 - y1)
            if x > 0: count += 1
    return count % 2


# compute average distance between points inside each cluster
def computeAverageDistance(graph, edgesDict):
    ave_distDict = {}
    for key in edgesDict.keys():
        distArray = []
        for (v1,v2) in edgesDict[key]:
            d = sqrt((graph.objects[v1][0]-graph.objects[v2][0])*(graph.objects[v1][0]-graph.objects[v2][0]) + (graph.objects[v1][1]-graph.objects[v2][1])*(graph.objects[v1][1]-graph.objects[v2][1]))
            distArray.append(d)
        ave_distDict[key] = sum(distArray) / float(max(1, len(distArray)))
    return ave_distDict


# ############################################################################################################################################
# #############################      FUNCTIONS FOR COMPUTING ENLARGED CLOSURE     ############################################################
# ############################################################################################################################################

# return a list of vertices that are connected to vertex and have an edge value value
def getNeighboursWithValue(graph, vertex, value):
    ret = []
    for n in graph.getNeighbours(vertex):
        if graph[vertex, n][VALUE] == value: ret.append(n)
    return ret

# for neighbors of v find which left and right vertices will represent the boundary of the enlarged closure
# this is only called when a vertex has more than 2 edges that we know that lie on the closure. in this case we have to figure out which pairs of vertices in ns belong together
def getNextPoints(graph, v, ns, closure):
    ret = []
    status = {}
    while ns != []:
        e = ns.pop()
        x_e = 0.999*graph.objects[v][0] + 0.001*graph.objects[e][0]; y_e = 0.999*graph.objects[v][1] + 0.001*graph.objects[e][1]
        for e2 in ns:
            x_e2 = 0.999*graph.objects[v][0] + 0.001*graph.objects[e2][0]; y_e2 = 0.999*graph.objects[v][1] + 0.001*graph.objects[e2][1]
            intersect = 0
            for test in ns:
                if test == e2: continue
                intersect += lineIntersect((x_e, y_e), (x_e2, y_e2), (graph.objects[v][0], graph.objects[v][1]), (graph.objects[test][0], graph.objects[test][1]))
            status[(e,e2)] = intersect

    for (e,e2) in status.keys():
        if not status.has_key((e,e2)): continue
        if status[(e,e2)] == 0:
            ret.append((e,e2))
            for (f, f2) in status.keys():
                if f == e or f == e2 or f2 == e or f2 == e2:
                    status.pop((f,f2))

    for (e,e2) in status.keys():
        if not status.has_key((e,e2)): continue
        ret.append((e,e2))
        for (f, f2) in status.keys():
            if f == e or f == e2 or f2 == e or f2 == e2:
                status.pop((f,f2))
    return ret

# add val to the dictionary under the key key
def addPointsToDict(dictionary, key, val):
    if dictionary.has_key(key): dictionary[key] = dictionary[key] + [val]
    else: dictionary[key] = [val]

# return a list that possibly contains multiple lists. in each list there are vertices sorted in the way as they are ordered on the closure
def getSortedClosurePoints(graph, closure):
    dicts = []
    tempDicts = []
    tempD = {}

    pointDict = {}
    points = union([i for (i,j) in closure], [j for (i,j) in closure])
    for p in points:
        ns = getNeighboursWithValue(graph, p, 2)
        if len(ns) > 2:
            split = getNextPoints(graph, p, ns, closure)       # splited is made of [(l1, r1), (l2, r2), ...]
        else: split = [(ns[0], ns[1])]
        pointDict[p] = split

    lists = []
    while pointDict.keys() != []:
        for start in pointDict.keys():
            if len(pointDict[start]) == 1:
                currList = [start, pointDict[start][0][0]]
                last = pointDict[start][0][1]
                pointDict.pop(start)
                break
        lastAdded = currList[-1]
        while lastAdded != start:
            for (l,r) in pointDict[lastAdded]:
                if l == currList[-2]:
                    currList.append(r)
                    pointDict[lastAdded].remove((l,r))
                    if pointDict[lastAdded] == []: pointDict.pop(lastAdded)
                    lastAdded = r
                    break
                elif r == currList[-2]:
                    currList.append(l)
                    pointDict[lastAdded].remove((l,r))
                    if pointDict[lastAdded] == []: pointDict.pop(lastAdded)
                    lastAdded = l
                    break
        lists.append(currList[:-1])
    return lists

# we are trying to blown a cluster. we therefore compute the new position of point point that lies outside the current cluster
def getEnlargedPointPosition(graph, point, left, right, dist, closure):
    y_diff1 = graph.objects[point][0] - graph.objects[left][0]
    x_diff1 = - (graph.objects[point][1] - graph.objects[left][1])
    d = sqrt(x_diff1*x_diff1 + y_diff1*y_diff1)
    x_diff1 /= d; y_diff1 /= d

    y_diff2 = graph.objects[right][0] - graph.objects[point][0]
    x_diff2 = - (graph.objects[right][1] - graph.objects[point][1])
    d = sqrt(x_diff2*x_diff2 + y_diff2*y_diff2)
    x_diff2 /= d; y_diff2 /= d

    x_diff = x_diff1 + x_diff2; y_diff = y_diff1 + y_diff2
    d = sqrt(x_diff*x_diff + y_diff*y_diff)
    x_diff /= d; y_diff /= d
    x_diff *= dist; y_diff *= dist

    x = graph.objects[point][0] + x_diff*0.001
    y = graph.objects[point][1] + y_diff*0.001
    if pointInsideCluster(graph.objects, closure, 0.99*graph.objects[left][0] + 0.01*x, 0.99*graph.objects[left][1] + 0.01*y):
        return (graph.objects[point][0] - x_diff, graph.objects[point][1] - y_diff)
    else:
        return (graph.objects[point][0] + x_diff, graph.objects[point][1] + y_diff)


# this function computes a blown cluster. The cluster is enlarged for a half of average distance between edges inside the cluster
def enlargeClosure(graph, closure, aveDist):
    halfAveDist = aveDist / 2.0
    sortedClosurePoints = getSortedClosurePoints(graph, closure)
    merged = []
    for group in sortedClosurePoints:
        currMerged = []
        for i in range(len(group)):
            p = group[i]
            l = group[i-1]; r = group[(i+1)%len(group)]
            x, y = getEnlargedPointPosition(graph, p, l, r, halfAveDist, closure)
            currMerged.append((x,y))
        merged.append(currMerged)
    return merged



#test widget appearance
if __name__=="__main__":
    import sys
    a = QApplication(sys.argv)
    ow = ClusterOptimization()
    a.setMainWidget(ow)
    ow.show()
    a.exec_()