import orange, sys, random, statc
import orngVisFuncts, orngTest, orngStat
from math import sqrt
import os, operator
from math import sqrt
import numpy, time
from copy import copy, deepcopy
from orngLinProj import FreeViz
from orngScaleData import getVariableValuesSorted

# used for outlier detection
VIZRANK_POINT = 0
CLUSTER_POINT = 1
VIZRANK_MOSAIC = 2

# quality measure
CLASS_ACCURACY = 0
AVERAGE_CORRECT = 1
BRIER_SCORE = 2
AUC = 3
measuresDict = {CLASS_ACCURACY: "Classification accuracy", AVERAGE_CORRECT: "Average probability of correct classification",
                BRIER_SCORE: "Brier score", AUC: "Area under curve (AUC)"}

# testing method
LEAVE_ONE_OUT = 0
TEN_FOLD_CROSS_VALIDATION = 1
TEST_ON_LEARNING_SET = 2

# results in the list
ACCURACY = 0
OTHER_RESULTS = 1
LEN_TABLE = 2
ATTR_LIST = 3
TRY_INDEX = 4
GENERAL_DICT = 5

OTHER_ACCURACY = 0
OTHER_PREDICTIONS = 1
OTHER_DISTRIBUTION = 2

# evaluation algorithm
ALGORITHM_KNN = 0
ALGORITHM_HEURISTIC = 1

NUMBER_OF_INTERVALS = 6  # number of intervals to use when discretizing. used when using the very fast heuristic

# attrCont
CONT_MEAS_NONE = 0
CONT_MEAS_RELIEFF = 1
CONT_MEAS_S2N = 2
CONT_MEAS_S2NMIX = 3

# attrDisc
DISC_MEAS_NONE = 0
DISC_MEAS_RELIEFF = 1
DISC_MEAS_GAIN = 2
DISC_MEAS_GINI = 3

DETERMINISTIC_ALL = 0
GAMMA_ALL = 1
GAMMA_SINGLE = 2

PROJOPT_NONE = 0
PROJOPT_SPCA = 1
PROJOPT_PLS = 2

contMeasuresDiscClass = [("None", None), ("ReliefF", orange.MeasureAttribute_relief(k=10, m=50)),
                ("Signal to Noise Ratio", orngVisFuncts.S2NMeasure()), ("Signal to Noise OVA", orngVisFuncts.S2NMeasureMix())]

discMeasuresDiscClass = [("None", None), ("ReliefF", orange.MeasureAttribute_relief(k=10, m=50)),
                ("Gain ratio", orange.MeasureAttribute_gainRatio()), ("Gini index", orange.MeasureAttribute_gini())]

contMeasuresNoClass = [("None", None)]
discMeasuresNoClass = [("None", None)]

contMeasuresContClass = [("None", None)]
discMeasuresContClass = [("None", None)]


# array of testing methods. used by calling python's apply method depending on the value of self.testingMethod
testingMethods = [orngTest.leaveOneOut, orngTest.crossValidation, orngTest.learnAndTestOnLearnData]

# visualization methods
SCATTERPLOT = 1
RADVIZ = 2
LINEAR_PROJECTION = 3
POLYVIZ = 4
KNN_IN_ORIGINAL_SPACE = 10

# optimization type
EXACT_NUMBER_OF_ATTRS = 0
MAXIMUM_NUMBER_OF_ATTRS = 1

class VizRank:
    def __init__(self, visualizationMethod, graph = None):
        if not graph:
            if visualizationMethod == SCATTERPLOT:
                import orngScaleScatterPlotData
                graph = orngScaleScatterPlotData.orngScaleScatterPlotData()
            elif visualizationMethod == RADVIZ:
                import orngScaleLinProjData
                graph = orngScaleLinProjData.orngScaleLinProjData()
                graph.normalize_examples = 1
            elif visualizationMethod in [LINEAR_PROJECTION, KNN_IN_ORIGINAL_SPACE]:
                import orngScaleLinProjData
                graph = orngScaleLinProjData.orngScaleLinProjData()
                graph.normalize_examples = 0
            elif visualizationMethod == POLYVIZ:
                import orngScalePolyvizData
                graph = orngScalePolyvizData.orngScalePolyvizData()
                graph.normalize_examples = 1
            else:
                print "an invalid visualization method was specified. VizRank can not run."
                return

        random.seed(0)      # always use the same seed to make results repeatable
        self.graph = graph
        self.freeviz = FreeViz(graph)
        self.visualizationMethod = visualizationMethod

        self.results = []
        self.arguments = []                                 # a list of arguments

        self.kValue = 10
        self.percentDataUsed = 100
        self.qualityMeasure = AVERAGE_CORRECT
        self.qualityMeasureCluster = 0      ### TO DO: fix it
        self.qualityMeasureContClass = 0    ### TO DO: fix it
        self.testingMethod = TEN_FOLD_CROSS_VALIDATION
        self.optimizationType = MAXIMUM_NUMBER_OF_ATTRS
        self.attributeCount = 4
        self.evaluationAlgorithm = ALGORITHM_KNN
        self.attrCont = CONT_MEAS_RELIEFF
        self.attrDisc = DISC_MEAS_RELIEFF
        self.attrContNoClass = 0
        self.attrDiscNoClass = 0
        self.attrDiscContClass = 0
        self.attrContContClass = 0
        
        self.attrSubsetSelection = GAMMA_ALL                # how do we find attribute subsets to evaluate - deterministic according to attribute ranking score or using gamma distribution - if using gamma, do we want to evaluate all possible permutations of attributes or only one
        self.projOptimizationMethod = PROJOPT_NONE          # None, supervisedPCA, partial least square
        self.useExampleWeighting = 0                        # weight examples, so that the class that has a low number of examples will have higher weights
        self.evaluationData = {}
        self.evaluationData["triedCombinations"] = {}

        self.externalLearner = None                         # do we use knn or some external learner
        self.selectedClasses = []                           # which classes are we trying to separate
        self.learnerName = "VizRank Learner"
        #self.onlyOnePerSubset = 1                           # save only the best placement of attributes in radviz
        self.maxResultListLen = 100000                      # number of projections to store in a list
        self.abortCurrentOperation = 0
        self.minNumOfExamples = 0                           # if a dataset has less than this number of examples we don't consider that projection

        # when to stop evaluation. when first criterion holds, evaluation stops
        self.timeLimit = 0              # if greater than 0 then this is the number of minutes that VizRank will use to evaluate projections
        self.projectionLimit = 0        # if greater than 0 then this is the number of projections that will be evaluated with VizRank
        self.evaluatedProjectionsCount = 0

        # when to stop local optimization?
        self.optimizeTimeLimit = 0
        self.optimizeProjectionLimit = 0
        self.optimizedProjectionsCount = 0

        if visualizationMethod == SCATTERPLOT: self.parentName = "Scatterplot"
        elif visualizationMethod == RADVIZ:    self.parentName = "Radviz"
        elif visualizationMethod == LINEAR_PROJECTION:  self.parentName = "Linear Projection"
        elif visualizationMethod == POLYVIZ:            self.parentName = "Polyviz"

        self.argumentCount = 1              # number of arguments used when classifying
        #self.argumentValueFormula = 1       # how to compute argument value

        self.locOptOptimizeProjectionByPermutingAttributes = 1      # try to improve projection by switching pairs of attributes in a projection
        self.locOptAllowAddingAttributes = 0                        # do we allow increasing the number of visualized attributes
        self.locOptMaxAttrsInProj = 20                              # if self.locOptAllowAddingAttributes == 1 then what is the maximum number of attributes in a projection
        self.locOptAttrsToTry = 50                                 # number of best ranked attributes to try
        self.locOptProjCount = 20                                   # try to locally optimize this number of best ranked projections

        self.rankArgumentsByStrength = 0  # how do you want to compute arguments. if 0 then we go through the top ranked projection and classify. If 1 we rerank projections to projections with strong class prediction and use them for classification
        self.storeEachPermutation = 0       # do we want to save information for each fold when evaluating projection - used to compute VizRank's accuracy

        # 0 - set to sqrt(N)
        # 1 - set to N / c
        self.kValueFormula = 1
        self.autoSetTheKValue = 1       # automatically set the value k
        
        self.saveEvaluationResults = 0
        self.evaluationResults = {}


    def clearResults(self):
        self.results = []
        self.evaluationResults = {}
        self.evaluationData = {}    # clear all previous data about tested permutations and stuff
        self.evaluationData["triedCombinations"] = {}

    def clearArguments(self):
        self.arguments = []

    def removeTooSimilarProjections(self, allowedPercentOfEqualAttributes = 70):
        i=0
        while i < len(self.results):
            if self.results[i][TRY_INDEX] != -1 and self.existsABetterSimilarProjection(i, allowedPercentOfEqualAttributes):
                self.results.pop(i)
            else:
                i += 1

    # test if one of the projections in self.results[0:index] are similar to the self.results[index] projection
    def existsABetterSimilarProjection(self, index, allowedPercentOfEqualAttributes = 70):
        testAttrs = self.results[index][ATTR_LIST]
        for i in range(index):
            attrs = self.results[i][ATTR_LIST]
            equalAttrs = [attr in attrs for attr in testAttrs]
            if 100*sum(equalAttrs) > allowedPercentOfEqualAttributes * float(len(testAttrs)):
                return 1
        return 0

    def getkValue(self, kValueFormula = -1):
        if not self.graph.have_data: return 1
        if kValueFormula == -1:
            kValueFormula = self.kValueFormula
        if kValueFormula == 0 or not self.graph.data_has_discrete_class or self.graph.data_has_continuous_class:
            kValue = int(sqrt(len(self.graph.raw_data)))
        else:
            kValue = int(len(self.graph.raw_data) / max(1, len(self.graph.data_domain.classVar.values)))    # k = N / c (c = # of class values)
        return kValue

    def createkNNLearner(self, k = -1, kValueFormula = -1):
        if k == -1:
            if kValueFormula == -1 or not self.graph.have_data or len(self.graph.raw_data) == 0:
                kValue = self.kValue
            else:
                kValue = self.getkValue(kValueFormula)

            if self.percentDataUsed != 100:
                kValue = int(kValue * self.percentDataUsed / 100.0)
        else:
            kValue = k

        return orange.kNNLearner(k = kValue, rankWeight = 0, distanceConstructor = orange.ExamplesDistanceConstructor_Euclidean(normalize=0))


    def setData(self, data):
        self.clearResults()
        self.selectedClasses = []
        if self.__class__ == VizRank:
            self.graph.setData(data, self.graph.raw_subset_data)

        if not self.graph.data_has_discrete_class:
            return

        self.selectedClasses = range(len(self.graph.data_domain.classVar.values))

        if self.autoSetTheKValue:
            self.kValue = self.getkValue(self.kValueFormula)

        self.correctSettingsIfNecessary()

    # save subsetdata. first example from this dataset can be used with argumentation - it can find arguments for classifying the example to the possible class values
    def setSubsetData(self, subData):
        if self.__class__ == VizRank:
            self.graph.setData(self.graph.raw_data, subData)
        self.clearArguments()

    def getEvaluatedAttributes(self):        
        if self.graph.data_has_discrete_class:
            return orngVisFuncts.evaluateAttributesDiscClass(self.graph.raw_data, contMeasuresDiscClass[self.attrCont][1], discMeasuresDiscClass[self.attrDisc][1])
        elif self.graph.data_has_continuous_class:
            return orngVisFuncts.evaluateAttributesContClass(self.graph.raw_data, contMeasuresContClass[self.attrContContClass][1], discMeasuresContClass[self.attrDiscContClass][1])
        else:
            return orngVisFuncts.evaluateAttributesNoClass(self.graph.raw_data, contMeasuresNoClass[self.attrContNoClass][1], discMeasuresNoClass[self.attrDiscNoClass][1])
        

    # return a function that is appropriate to find the best projection in a list in respect to the selected quality measure
    def getMaxFunct(self):
        if self.graph.data_has_discrete_class and self.qualityMeasure == BRIER_SCORE: return min
        else: return max

    def addResult(self, accuracy, other_results, lenTable, attrList, tryIndex, generalDict = {}, results=None):
        self.insertItem(self.findTargetIndex(accuracy), accuracy, other_results, lenTable, attrList, tryIndex, generalDict)

    # use bisection to find correct index
    def findTargetIndex(self, accuracy):
        funct = self.getMaxFunct()
        top = 0; bottom = len(self.results)

        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if funct(accuracy, self.results[mid][ACCURACY]) == accuracy: bottom = mid
            else: top = mid

        if len(self.results) == 0: return 0
        if funct(accuracy, self.results[top][ACCURACY]) == accuracy:
            return top
        else:
            return bottom

    # insert new result - give parameters: accuracy of projection, number of examples in projection and list of attributes.
    def insertItem(self, index, accuracy, other_results, lenTable, attrList, tryIndex, generalDict = {}, updateStatusBar = 0):
        if index < self.maxResultListLen:
            self.results.insert(index, (accuracy, other_results, lenTable, attrList, tryIndex, generalDict))


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

    # kNNEvaluate - evaluate class separation in the given projection using a heuristic or k-NN method
    def kNNComputeAccuracy(self, table):
        # select a subset of the data if necessary
        if self.percentDataUsed != 100:
            indices = orange.MakeRandomIndices2(table, 1.0-float(self.percentDataUsed)/100.0)
            testTable = table.select(indices)
        else:
            testTable = table

        if len(testTable) == 0: return 0, 0

        if self.evaluationAlgorithm == ALGORITHM_KNN or self.externalLearner:
            if self.externalLearner: learner = self.externalLearner
            else:                    learner = self.createkNNLearner(); weight = 0

            if self.useExampleWeighting and testTable.domain.classVar and testTable.domain.classVar.varType == orange.VarTypes.Discrete:
                testTable, weightID = orange.Preprocessor_addClassWeight(testTable, equalize=1)
                results = apply(testingMethods[self.testingMethod], [[learner], (testTable, weightID)])
            else:
                results = apply(testingMethods[self.testingMethod], [[learner], testTable])

            if self.saveEvaluationResults:
                self.evaluationResults = results
                #self.classifier = 

            # compute classification success using selected measure
            if testTable.domain.classVar.varType == orange.VarTypes.Discrete:
                return self.computeAccuracyFromResults(testTable, results)

            # for continuous class we can't compute brier score and classification accuracy
            else:
                val = 0.0
                if not results.results or not results.results[0].probabilities[0]: return 0, 0
                for res in results.results:  val += res.probabilities[0].density(res.actualClass)
                if len(results.results) > 0: val/= float(len(results.results))
                return 100.0*val, (100.0*val)

        # ###############################
        # do we want to use very fast heuristic
        # ###############################
        elif self.evaluationAlgorithm == ALGORITHM_HEURISTIC:
            # if input attributes are continuous (may be discrete for evaluating scatterplots, where we dicretize the whole domain...)
            if testTable.domain[0].varType == orange.VarTypes.Continuous and testTable.domain[1].varType == orange.VarTypes.Continuous:
                discX = orange.EquiDistDiscretization(testTable.domain[0], testTable, numberOfIntervals = NUMBER_OF_INTERVALS)
                discY = orange.EquiDistDiscretization(testTable.domain[0], testTable, numberOfIntervals = NUMBER_OF_INTERVALS)
                testTable = testTable.select([discX, discY, testTable.domain.classVar])

            currentClassDistribution = [int(v) for v in orange.Distribution(testTable.domain.classVar, testTable)]
            prediction = [0.0 for i in range(len(testTable.domain.classVar.values))]

            # create a new attribute that is a cartesian product of the two visualized attributes
            nattr = orange.EnumVariable(values=[str(i) for i in range(NUMBER_OF_INTERVALS*NUMBER_OF_INTERVALS)])
            nattr.getValueFrom = orange.ClassifierByLookupTable2(nattr, testTable.domain[0], testTable.domain[1])
            for i in range(len(nattr.getValueFrom.lookupTable)): nattr.getValueFrom.lookupTable[i] = i

            for dist in orange.ContingencyAttrClass(nattr, testTable):
                dist = list(dist)
                if sum(dist) == 0: continue
                m = max(dist)
                prediction[dist.index(m)] += m * m / float(sum(dist))

            prediction = [val*100.0 for val in prediction]             # turn prediction array into percents
            acc = sum(prediction) / float(max(1, len(testTable)))               # compute accuracy for all classes
            val = 0.0; s = 0.0
            for index in self.selectedClasses:                          # compute accuracy for selected classes
                val += prediction[index]
                s += currentClassDistribution[index]
            for i in range(len(prediction)):
                prediction[i] /= float(max(1, currentClassDistribution[i]))    # turn to probabilities
            return val/float(max(1,s)), (acc, prediction, currentClassDistribution)
        else:
            return 0, 0     # in case of an invalid value


    def computeAccuracyFromResults(self, table, results):
        prediction = [0.0 for i in range(len(table.domain.classVar.values))]
        countsByFold =  [0 for i in range(results.numberOfIterations)]

        if self.qualityMeasure == AVERAGE_CORRECT:
            for res in results.results:
                if not res.probabilities[0]: continue
                prediction[res.actualClass] += res.probabilities[0][res.actualClass]
                countsByFold[res.iterationNumber] += 1
            prediction = [val*100.0 for val in prediction]

        elif self.qualityMeasure == BRIER_SCORE:
            #return orngStat.BrierScore(results)[0], results
            for res in results.results:
                if not res.probabilities[0]: continue
                prediction[res.actualClass] += sum([prob*prob for prob in res.probabilities[0]]) - 2*res.probabilities[0][res.actualClass] + 1
                countsByFold[res.iterationNumber] += 1

        elif self.qualityMeasure == CLASS_ACCURACY:
            #return 100*orngStat.CA(results)[0], results
            for res in results.results:
                prediction[res.actualClass] += res.classes[0]==res.actualClass
                countsByFold[res.iterationNumber] += 1
            prediction = [val*100.0 for val in prediction]
        elif self.qualityMeasure == AUC:
            aucResult = orngStat.AUC(results)
            if aucResult:
                return aucResult[0], None
            else:
                return 0, None

        # compute accuracy only for classes that are selected as interesting. other class values do not participate in projection evaluation
        acc = sum(prediction) / float(max(1, len(results.results)))                 # accuracy over all class values
        classes = self.selectedClasses or range(len(self.graph.data_domain.classVar.values))
        val = sum([prediction[index] for index in classes])    # accuracy over all selected classes

        currentClassDistribution = [int(v) for v in orange.Distribution(table.domain.classVar, table)]
        s = sum([currentClassDistribution[index] for index in classes])

        prediction = [prediction[i] / float(max(1, currentClassDistribution[i])) for i in range(len(prediction))] # turn to probabilities
        
        return val/max(1, float(s)), (acc, prediction, list(currentClassDistribution))


    # Argumentation functions
    def findArguments(self, example):
        self.clearArguments()
        if not self.graph.have_data or not self.graph.data_has_class or len(self.results) == 0:
            if len(self.results) == 0: print 'To classify an example using VizRank you first have to evaluate some projections.'
            return orange.MajorityLearner(self.graph.raw_data)(example, orange.GetBoth)

        self.arguments = [[] for i in range(len(self.graph.data_domain.classVar.values))]
        vals = [0.0 for i in range(len(self.arguments))]

        if self.rankArgumentsByStrength == 1:
            for index in range(min(len(self.results), self.argumentCount + 50)):
                classValue, dist = self.computeClassificationForExample(index, example, kValue = len(self.graph.raw_data))
                if classValue and dist:
                    for i in range(len(self.arguments)):
                        self.arguments[i].insert(self.getArgumentIndex(dist[i], i), (dist[i], dist, self.results[index][ATTR_LIST], index))

            for i in range(len(self.arguments)):
                arr = self.arguments[i]
                arr.sort()
                arr.reverse()
                arr = arr[:self.argumentCount]
                self.arguments[i] = arr
                vals[i] = sum([arg[0] for arg in arr])
        else:
            usedArguments = 0; index = 0
            while usedArguments < self.argumentCount and index < len(self.results):
                classValue, dist = self.computeClassificationForExample(index, example, kValue = self.getkValue(kValueFormula = 0))
                if classValue and dist:
                    for i in range(len(self.arguments)):
                        self.arguments[i].insert(self.getArgumentIndex(dist[i], i), (dist[i], dist, self.results[index][ATTR_LIST], index))
                        vals[i] += dist[i]
                    usedArguments += 1
                index += 1

        suma = sum(vals)
        if suma == 0:
            dist = orange.Distribution(self.graph.data_domain.classVar.name, self.graph.raw_data)
            vals = [dist[i] for i in range(len(dist))]; suma = sum(vals)

        classValue = example.domain.classVar[vals.index(max(vals))]
        dist = orange.DiscDistribution([val/float(suma) for val in vals])
        dist.variable = self.graph.data_domain.classVar
        return classValue, dist


    def computeClassificationForExample(self, projectionIndex, example, kValue = -1):
        (accuracy, other_results, lenTable, attrList, tryIndex, generalDict) = self.results[projectionIndex]

        if 1 in [example[attr].isSpecial() for attr in attrList]: return None, None

        attrIndices = [self.graph.attribute_name_index[attr] for attr in attrList]
        attrVals = [self.graph.scale_example_value(example, ind) for ind in attrIndices]

        table = self.graph.create_projection_as_example_table(attrIndices, settingsDict = generalDict)
        [xTest, yTest] = self.graph.get_projected_point_position(attrIndices, attrVals, settingsDict = generalDict)

        learner = self.externalLearner or self.createkNNLearner(k = kValue)
        if self.useExampleWeighting: table, weightID = orange.Preprocessor_addClassWeight(table, equalize=1)
        else: weightID = 0

        classifier = learner(table, weightID)
        classVal, dist = classifier(orange.Example(table.domain, [xTest, yTest, "?"]), orange.GetBoth)
        return classVal, dist


    def getArgumentIndex(self, value, classValue):
        top = 0; bottom = len(self.arguments[classValue])
        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if max(value, self.arguments[classValue][mid][0]) == value: bottom = mid
            else: top = mid

        if len(self.arguments[classValue]) == 0: return 0
        if max(value, self.arguments[classValue][top][0]) == value:  return top
        else:                                                        return bottom

    def correctSettingsIfNecessary(self):
        if not self.graph.have_data: return
        # check if we have discrete attributes. if yes, then make sure we are not using s2nMix measure and GAMMA_SINGLE
        if orange.VarTypes.Discrete in [attr.varType for attr in self.graph.data_domain.attributes]:
            if self.attrCont == CONT_MEAS_S2NMIX:           self.attrCont = CONT_MEAS_S2N
            if self.attrSubsetSelection == GAMMA_SINGLE:    self.attrSubsetSelection = GAMMA_ALL

    def isEvaluationCanceled(self):
        stop = 0
        if self.timeLimit > 0: stop = (time.time() - self.startTime) / 60 >= self.timeLimit
        if self.projectionLimit > 0: stop = stop or self.evaluatedProjectionsCount >= self.projectionLimit
        return stop

    def isOptimizationCanceled(self):
        stop = 0
        if self.optimizeTimeLimit > 0: stop = (time.time() - self.startTime) / 60 >= self.optimizeTimeLimit
        if self.optimizeProjectionLimit > 0: stop = stop or self.optimizedProjectionsCount >= self.optimizeProjectionLimit
        return stop


    # get a new subset of attributes. if attributes are not evaluated yet then evaluate them and save info to evaluationData dict.
    def selectNextAttributeSubset(self, minLength, maxLength):
        z = self.evaluationData.get("z", minLength-1)
        u = self.evaluationData.get("u", minLength-1)
        self.evaluationData["combinations"] = []
        self.evaluationData["index"] = 0

        # if we use heuristic to find attribute orders
        if self.attrCont == CONT_MEAS_S2NMIX or self.attrSubsetSelection == GAMMA_SINGLE:
            if not self.evaluationData.has_key("attrs"):
                attributes, attrsByClass = orngVisFuncts.findAttributeGroupsForRadviz(self.graph.raw_data, orngVisFuncts.S2NMeasureMix())
                attributes = [self.graph.attribute_name_index[name] for name in attributes]
                attrsByClass = [[self.graph.attribute_name_index[name] for name in arr] for arr in attrsByClass]
                self.evaluationData["attrs"] = (attributes, attrsByClass)
            else:
                attributes, attrsByClass = self.evaluationData["attrs"]

            if z >= len(attributes): return None      # did we already try all the attributes
            numClasses = len(self.graph.data_domain.classVar.values)
            if self.attrSubsetSelection in [GAMMA_ALL, GAMMA_SINGLE]:
                combinations = self.getAttributeSubsetUsingGammaDistribution(u+1)
            else:
                combinations = orngVisFuncts.combinations(range(z), u)
                for i in range(len(combinations))[::-1]:
                    comb = combinations[i] + [z]
                    counts = [0] * numClasses
                    for ind in comb: counts[ind%numClasses] += 1
                    if max(counts) - min(counts) > 1:
                        combinations.pop(i)     # ignore combinations that don't have approximately the same number of attributes for each class value
                        continue
                    attrList = [[] for c in range(numClasses)]
                    for ind in comb: attrList[ind % numClasses].append(attributes[ind])
                    combinations[i] = attrList

        # no heuristic. try all combinations of a group of attributes
        else:
            if not self.evaluationData.has_key("attrs"):
                # evaluate attributes
                evaluatedAttributes = self.getEvaluatedAttributes()
                attributes = [self.graph.attribute_name_index[name] for name in evaluatedAttributes]
                self.evaluationData["attrs"] = attributes
                self.totalPossibilities = 0

                # build list of indices for permutations of different number of attributes
                permutationIndices = {}
                for i in range(minLength, maxLength+1):
                    if i > len(attributes): continue        # if we don't have enough attributes
                    if self.projOptimizationMethod != 0 or self.visualizationMethod == KNN_IN_ORIGINAL_SPACE:
                        permutationIndices[i] = [range(i)]
                    else:
                        permutationIndices[i] = orngVisFuncts.generateDifferentPermutations(range(i))
                    self.totalPossibilities += orngVisFuncts.combinationsCount(i, len(attributes)) * len(permutationIndices[i])
##                sys.stderr.write("selectNextAttributeSubset " + str(permutationIndices.keys()) + "\n")
                self.evaluationData["permutationIndices"] = permutationIndices
            else:
                attributes = self.evaluationData["attrs"]

            # do we have enough attributes at all?
            if len(attributes) < u+1:
                combinations = []
            else:
                # if we don't want to use any heuristic
                if self.attrCont == CONT_MEAS_NONE and self.attrDisc == DISC_MEAS_NONE:
                    combination = []
                    while len(combination) < u+1:
                        v = random.randint(0, len(self.graph.data_domain.attributes)-1)
                        if v not in combination: combination.append(v)
                    combinations = [combination]
                elif self.attrSubsetSelection == DETERMINISTIC_ALL:
                    if z >= len(attributes): return None      # did we already try all the attributes
                    combinations = orngVisFuncts.combinations(attributes[:z], u)
                    map(list.append, combinations, [attributes[z]] * len(combinations))     # append the z-th attribute to all combinations in the list
                elif self.attrSubsetSelection in [GAMMA_ALL, GAMMA_SINGLE]:
                    combinations = self.getAttributeSubsetUsingGammaDistribution(u+1)

        # update values for the number of attributes
        u += 1
        self.evaluationData["u"] = (u >= maxLength and minLength-1) or u
        if self.attrSubsetSelection == DETERMINISTIC_ALL:
            self.evaluationData["z"] = (u >= maxLength and z+1) or z

        self.evaluationData["combinations"] = combinations
        return combinations

    # use gamma distribution to select a subset of attrCount attributes. if we want to use heuristic to find attribute order then
    # apply gamma distribution on attribute lists for each class value.
    # before returning a subset of attributes also test if this subset was already tested. if yes, then try to generate a new subset (repeat this max 50 times)
    def getAttributeSubsetUsingGammaDistribution(self, attrCount):
        maxTries = 100
        triedDict = self.evaluationData.get("triedCombinations", {})
        projCountWidth = len(triedDict.keys()) / 1000

        if self.attrCont == CONT_MEAS_S2NMIX or self.attrSubsetSelection == GAMMA_SINGLE:
            numClasses = len(self.graph.data_domain.classVar.values)
            attributes, attrsByClass = self.evaluationData["attrs"]

            for i in range(maxTries):
                attrList = [[] for c in range(numClasses)]; attrs = []
                tried = 0
                while len(attrs) < min(attrCount, len(self.graph.data_domain.attributes)):
                    ind = tried%numClasses
                    #ind = random.randint(0, numClasses-1)       # warning: this can generate uneven groups for each class value!!!
                    attr = attrsByClass[ind][int(random.gammavariate(1, 5 + i/10 + projCountWidth))%len(attrsByClass[ind])]
                    if attr not in attrList[ind]:
                        attrList[ind].append(attr)
                        attrs.append(attr)
                    tried += 1
                attrs.sort()
                if not triedDict.has_key(tuple(attrs)) and len(attrs) == attrCount:
                    self.evaluationData["triedCombinations"][tuple(attrs)] = 1     # this is not the best, since we don't want to save used combinations since we only test one permutation
                    #return [filter(None, attrList)]        # problem: using filter removes value 0 from the array, which means that the attribute ranked as best wont be in the projections
                    return [attrList]
        else:
            attributes = self.evaluationData["attrs"]
            for i in range(maxTries):
                attrList = []
                while len(attrList) < min(attrCount, len(attributes)):
                    attr = attributes[int(random.gammavariate(1,5 + (len(attributes)/1000) + projCountWidth))%len(attributes)]
                    if attr not in attrList:
                        attrList.append(attr)
                attrList.sort()
                if not triedDict.has_key(tuple(attrList)):
                    triedDict[tuple(attrList)] = 1
                    #return [filter(None, attrList)]        # problem: using filter removes value 0 from the array, which means that the attribute ranked as best wont be in the projections
                    return [attrList]
        return None

    # generate possible permutations of the current attribute subset. use evaluationData dict to find which attribute subset to use.
    def getNextPermutations(self):
        combinations = self.evaluationData["combinations"]
        index  = self.evaluationData["index"]
        if not combinations or index >= len(combinations):
            return None     # did we test all the projections

        combination = combinations[index]
        permutations = []

        if self.attrCont == CONT_MEAS_S2NMIX or self.attrSubsetSelection == GAMMA_SINGLE:
            # if we don't want to test all placements then we only create a permutation of groups and attributes in each group
            if self.attrSubsetSelection == GAMMA_SINGLE:
                permutations = [reduce(operator.add, combination)]
                usedPerms = {tuple(permutations[0]):1}
                for c in range(10):
                    combination = [[group.pop(random.randint(0, len(group)-1)) for num in range(len(group))] for group in [combination.pop(random.randint(0, len(combination)-1)) for i in range(len(combination))]]
                    comb = reduce(operator.add, combination)
                    if not usedPerms.has_key(tuple(comb)):
                        usedPerms[tuple(comb)] = 1
                        permutations.append(comb)

            # create only one permutation, because its all we need
            elif self.projOptimizationMethod != 0 or self.visualizationMethod == KNN_IN_ORIGINAL_SPACE:
                permutations.append(reduce(operator.add, combination))
            else:
                for proj in orngVisFuncts.createProjections(len(self.graph.data_domain.classVar.values), sum([len(group) for group in combination])):
                    try: permutations.append([combination[i][j] for (i,j) in proj])
                    except: pass
        else:
            permutationIndices = self.evaluationData["permutationIndices"]
##            sys.stderr.write("getNextPermutations " + str(permutationIndices.keys()) + "\n")
            permutations = [[combination[val] for val in ind] for ind in permutationIndices[len(combination)]]

        self.evaluationData["index"] = index + 1
        return permutations

    def computeTotalHeight(self, node):
        if node.branches: 
            return node.height * (node.last - node.first) + sum([self.computeTotalHeight(n) for n in node.branches])
        else:
            return node.height

    def evaluateProjection(self, data):
        if self.graph.data_has_discrete_class:
            return self.kNNComputeAccuracy(data)
        elif self.graph.data_has_continuous_class:
            return 0
        else:
            matrix = orange.SymMatrix(len(data))
            matrix.setattr('items', data)
            dist = orange.ExamplesDistanceConstructor_Euclidean(data)
            for i in range(len(data)):
                for j in range(i+1):
                    matrix[i, j] = dist(data[i], data[j])
            root = orange.HierarchicalClustering(matrix, linkage = orange.HierarchicalClustering.Ward, overwriteMatrix = 0)
            val = self.computeTotalHeight(root)
            return val, (val)
            

    # ##########################################################################
    # MAIN FUNCTION FOR EVALUATING PROJECTIONS
    # ##########################################################################
    def evaluateProjections(self, clearPreviousProjections = 1):
        random.seed(0)      # always use the same seed to make results repeatable
        if not self.graph.have_data: return 0
        
        # TO DO: remove the following line when we add support for cont class
        if not self.graph.data_has_discrete_class: return 0
        self.correctSettingsIfNecessary()
        if self.timeLimit == self.projectionLimit == 0 and self.__class__.__name__ == "VizRank":
            print "Evaluation of projections was started without any time or projection restrictions. To prevent an indefinite projection evaluation a time limit of 2 hours was set."
            self.timeLimit = 2 * 60

        self.startTime = time.time()

        if clearPreviousProjections:
            self.evaluatedProjectionsCount = 0
            self.optimizedProjectionsCount = 0
            self.evaluationData = {}            # clear all previous data about tested permutations and stuff
            self.evaluationData["triedCombinations"] = {}
            self.clearResults()

        self.clearArguments()
        maxFunct = self.getMaxFunct()
        
        if self.__class__ != VizRank:
            from PyQt4.QtGui import qApp

#        if not self.graph.data_has_discrete_class:
#            print "Projections can be evaluated only for data with a discrete class."
#            return 0

        if self.visualizationMethod == SCATTERPLOT:
            evaluatedAttributes = self.getEvaluatedAttributes()
            contVars = [orange.FloatVariable(attr.name) for attr in self.graph.data_domain.attributes]
            attrCount = len(self.graph.data_domain.attributes)

            count = len(evaluatedAttributes)*(len(evaluatedAttributes)-1)/2
            strCount = orngVisFuncts.createStringFromNumber(count)
            
            for i in range(len(evaluatedAttributes)):
                attr1 = self.graph.attribute_name_index[evaluatedAttributes[i]]
                for j in range(i):
                    attr2 = self.graph.attribute_name_index[evaluatedAttributes[j]]
                    self.evaluatedProjectionsCount += 1
                    if self.isEvaluationCanceled():
                        return self.evaluatedProjectionsCount

                    table = self.graph.create_projection_as_example_table([attr1, attr2])
                    if len(table) < self.minNumOfExamples: continue
                    accuracy, other_results = self.evaluateProjection(table)
                    generalDict = {"Results": self.evaluationResults} if self.saveEvaluationResults else {}
                    self.addResult(accuracy, other_results, len(table), [self.graph.data_domain[attr1].name, self.graph.data_domain[attr2].name], self.evaluatedProjectionsCount, generalDict=generalDict)

                    if self.__class__ != VizRank:
                        self.setStatusBarText("Evaluated %s/%s projections..." % (orngVisFuncts.createStringFromNumber(self.evaluatedProjectionsCount), strCount))
                        self.parentWidget.progressBarSet(100.0*self.evaluatedProjectionsCount/max(1,float(count)))

        # #################### RADVIZ, LINEAR_PROJECTION  ################################
        elif self.visualizationMethod in (RADVIZ, LINEAR_PROJECTION, POLYVIZ, KNN_IN_ORIGINAL_SPACE):
            if self.projOptimizationMethod != 0:
                self.freeviz.useGeneralizedEigenvectors = 1
                self.graph.normalize_examples = 0

            # variables and domain for the table
            domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), orange.EnumVariable(self.graph.data_domain.classVar.name, values = getVariableValuesSorted(self.graph.data_domain.classVar))])
            minLength = (self.optimizationType == EXACT_NUMBER_OF_ATTRS and self.attributeCount) or 3
            maxLength = self.attributeCount
            classListFull = self.graph.original_data[self.graph.data_class_index]

            # each call to selectNextAttributeSubset gets a new combination of attributes in a range from minLength to maxLength. if we return None for a given number of attributes this
            # doesn't mean yet that there are no more possible combinations. it may be just that we wanted a combination of 6 attributes in a domain with 4 attributes. therefore we have
            # to try maxLength-minLength+1 times and if we fail every time then there are no more valid projections

            newProjectionsExist = 1
            while newProjectionsExist:
                for experiment in range(maxLength-minLength+1):
                    if self.selectNextAttributeSubset(minLength, maxLength): break
                    newProjectionsExist = 0
                permutations = self.getNextPermutations()
                while permutations:
                    attrIndices = permutations[0]

                    # if we use SPCA, PLS or KNN_IN_ORIGINAL_SPACE
                    if self.projOptimizationMethod != 0 or self.visualizationMethod == KNN_IN_ORIGINAL_SPACE:
                        if self.visualizationMethod == KNN_IN_ORIGINAL_SPACE:
                            table = self.graph.raw_data.select([self.graph.data_domain[attr] for attr in attrIndices] + [self.graph.data_domain.classVar] )
                            xanchors, yanchors = self.graph.create_xanchors(len(attrIndices)), self.graph.create_yanchors(len(attrIndices))
                            attrNames = [self.graph.data_domain[attr].name for attr in attrIndices]
                        else:
                            projections = self.freeviz.findProjection(self.projOptimizationMethod, attrIndices, set_anchors = 0, percentDataUsed = self.percentDataUsed)
                            if projections != None:
                                xanchors, yanchors, (attrNames, newIndices) = projections
                                table = self.graph.create_projection_as_example_table(newIndices, domain = domain, XAnchors = xanchors, YAnchors = yanchors)
                        if len(table) < self.minNumOfExamples: continue
                        self.evaluatedProjectionsCount += 1
                        accuracy, other_results = self.evaluateProjection(table)
                        generalDict = {"XAnchors": list(xanchors), "YAnchors": list(yanchors), "Results": self.evaluationResults} if self.saveEvaluationResults else {"XAnchors": list(xanchors), "YAnchors": list(yanchors)}
                        self.addResult(accuracy, other_results, len(table), attrNames, self.evaluatedProjectionsCount, generalDict = generalDict)
                        if self.isEvaluationCanceled(): return self.evaluatedProjectionsCount
                        if self.__class__ != VizRank:
                            self.setStatusBarText("Evaluated %s projections..." % (orngVisFuncts.createStringFromNumber(self.evaluatedProjectionsCount)))
                    else:
                        XAnchors = self.graph.create_xanchors(len(attrIndices))
                        YAnchors = self.graph.create_yanchors(len(attrIndices))
                        validData = self.graph.get_valid_list(attrIndices)
                        if numpy.sum(validData) >= self.minNumOfExamples:
                            classList = numpy.compress(validData, classListFull)
                            selectedData = numpy.compress(validData, numpy.take(self.graph.no_jittering_scaled_data, attrIndices, axis = 0), axis = 1)
                            sum_i = self.graph._getSum_i(selectedData)

                            tempList = []

                            # for every permutation compute how good it separates different classes
                            for permutation in permutations:
                                if self.evaluatedProjectionsCount % 10 == 0 and self.isEvaluationCanceled():
                                    continue

                                table = self.graph.create_projection_as_example_table(permutation, validData = validData, classList = classList, sum_i = sum_i, XAnchors = XAnchors, YAnchors = YAnchors, domain = domain)
                                accuracy, other_results = self.evaluateProjection(table)

                                # save the permutation
                                if self.storeEachPermutation:
                                    generalDict = {"Results": self.evaluationResults} if self.saveEvaluationResults else {}
                                    self.addResult(accuracy, other_results, len(table), [self.graph.attribute_names[i] for i in permutation], self.evaluatedProjectionsCount, generalDict)
                                else:
                                    tempList.append((accuracy, other_results, len(table), [self.graph.attribute_names[i] for i in permutation]))

                                self.evaluatedProjectionsCount += 1
                                if self.__class__ != VizRank:
                                    self.setStatusBarText("Evaluated %s projections..." % (orngVisFuncts.createStringFromNumber(self.evaluatedProjectionsCount)))
                                    qApp.processEvents()        # allow processing of other events

                            if not self.storeEachPermutation and len(tempList) > 0:   # return only the best attribute placements
                                (acc, other_results, lenTable, attrList) = maxFunct(tempList)
                                generalDict = {"Results": self.evaluationResults} if self.saveEvaluationResults else {}
                                self.addResult(acc, other_results, lenTable, attrList, self.evaluatedProjectionsCount, generalDict=generalDict)

                        if self.isEvaluationCanceled():
                            return self.evaluatedProjectionsCount

                    permutations = self.getNextPermutations()
        else:
            print "unknown visualization method"

        return self.evaluatedProjectionsCount

    def getProjectionQuality(self, attrList, useAnchorData = 0):
        if not self.graph.have_data: return 0.0, None
        table = self.graph.create_projection_as_example_table([self.graph.attribute_name_index[attr] for attr in attrList], useAnchorData = useAnchorData)
        return self.evaluateProjection(table)


    def insertTempProjection(self, projections, acc, attrList):
        if len(projections) == 0: return [(acc, attrList)]

        top = 0; bottom = len(projections)
        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if max(acc, projections[mid][0]) == acc: bottom = mid
            else: top = mid

        if max(acc, projections[top][0]) == acc: projections.insert(top, (acc, attrList))
        else:                                    projections.insert(bottom, (acc, attrList))

    # ##########################################################################
    # FUNCTION FOR OPTIMIZING BEST PROJECTIONS
    # ##########################################################################
    def optimizeBestProjections(self, restartWhenImproved = 1):
        random.seed(0)      # always use the same seed to make results repeatable
        count = min(len(self.results), self.locOptProjCount)
        if not count: return
        self.correctSettingsIfNecessary()
        self.optimizedProjectionsCount = 0
        """
        if self.optimizeTimeLimit == self.optimizeProjectionLimit == 0:
            print "Optimization of projections was started without any time or projection restrictions. To prevent an indefinite projection optimization a time limit of 2 hours was set."
            self.optimizeProjectionLimit = 2 * 60
        """

        if self.__class__ != VizRank:
            from PyQt4.QtGui import qApp

        attrs = [self.results[i][ATTR_LIST] for i in range(count)]                                   # create a list of attributes that are in the top projections
        attrs = [[self.graph.attribute_name_index[name] for name in projection] for projection in attrs]    # find indices from the attribute names
        accuracys = [self.getProjectionQuality(self.results[i][ATTR_LIST])[0] for i in range(count)]
        projections = [(accuracys[i], attrs[i]) for i in range(len(accuracys))]

        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), orange.EnumVariable(self.graph.data_domain.classVar.name, values = getVariableValuesSorted(self.graph.data_domain.classVar))])
        attributes = [self.graph.attribute_name_index[name] for name in self.getEvaluatedAttributes()[:self.locOptAttrsToTry]]
        self.startTime = time.time()
        lenOfAttributes = len(attributes)
        maxFunct = self.getMaxFunct()

        if self.visualizationMethod == SCATTERPLOT:
            classListFull = self.graph.original_data[self.graph.data_class_index]

            tempDict = {}
            projIndex = 0
            while len(projections) > 0:
                (accuracy, projection) = projections.pop(0)
                projIndex -= 1

                significantImprovement = 0
                strTotalAtts = orngVisFuncts.createStringFromNumber(lenOfAttributes)
                for (attrIndex, attr) in enumerate(attributes):
                    if attr in projection: continue
                    testProjections = []
                    if not tempDict.has_key((projection[0], attr)) and not tempDict.has_key((attr, projection[0])): testProjections.append([projection[0], attr])
                    if not tempDict.has_key((projection[1], attr)) and not tempDict.has_key((attr, projection[1])): testProjections.append([attr, projection[1]])

                    for testProj in testProjections:
                        table = self.graph.create_projection_as_example_table(testProj, domain = domain)
                        if len(table) < self.minNumOfExamples: continue
                        acc, other_results = self.evaluateProjection(table)
                        if hasattr(self, "setStatusBarText") and self.optimizedProjectionsCount % 10 == 0:
                            self.setStatusBarText("Evaluated %s projections. Last accuracy was: %2.2f%%" % (orngVisFuncts.createStringFromNumber(self.optimizedProjectionsCount), acc))
                        if acc > accuracy:
                            self.addResult(acc, other_results, len(table), [self.graph.attribute_names[i] for i in testProj], projIndex)
                            self.insertTempProjection(projections, acc, testProj)
                            tempDict[tuple(testProj)] = 1
                            if min(acc, accuracy) != 0 and max(acc, accuracy) > 1.005 *min(acc, accuracy):  significantImprovement = 1

                        self.optimizedProjectionsCount += 1
                        if self.__class__ != VizRank:
                            qApp.processEvents()        # allow processing of other events
                        if self.optimizedProjectionsCount % 10 == 0 and self.isOptimizationCanceled():
                            return self.optimizedProjectionsCount
                    if significantImprovement: break

        # #################### RADVIZ, LINEAR_PROJECTION  ################################
        elif self.visualizationMethod in (RADVIZ, LINEAR_PROJECTION, POLYVIZ):
            numClasses = len(self.graph.data_domain.classVar.values)

            classListFull = self.graph.original_data[self.graph.data_class_index]
            newProjDict = {}
            projIndex = 0

            while len(projections) > 0:
                (accuracy, projection) = projections.pop(0)
                projIndex -= 1

                # first try to use the attributes in the projection and evaluate only different permutations of these attributes
                if self.locOptOptimizeProjectionByPermutingAttributes == 1 and self.projOptimizationMethod == 0:
                    bestProjection = projection; tempProjection = projection
                    bestAccuracy = accuracy; tempAccuracy = accuracy
                    triedPermutationsDict = {}
                    failedConsecutiveTries = 0
                    tries = 0
                    XAnchors = self.graph.create_xanchors(len(projection))
                    YAnchors = self.graph.create_yanchors(len(projection))
                    validData = self.graph.get_valid_list(projection)
                    classList = numpy.compress(validData, classListFull)
                    while failedConsecutiveTries < 5 and tries < 50:
                        #newProj = orngVisFuncts.switchTwoElements(tempProjection, nrOfTimes = 3)
                        newProj = orngVisFuncts.switchTwoElementsInGroups(tempProjection, numClasses, 3)
                        tries += 1
                        if triedPermutationsDict.has_key(str(newProj)):
                            failedConsecutiveTries += 1
                        else:
                            failedConsecutiveTries = 0
                            triedPermutationsDict[str(newProj)] = 1

                            table = self.graph.create_projection_as_example_table(newProj, validData = validData, classList = classList, XAnchors = XAnchors, YAnchors = YAnchors, domain = domain)
                            if len(table) < self.minNumOfExamples: continue
                            acc, other_results = self.evaluateProjection(table)
                            self.optimizedProjectionsCount += 1
                            if self.__class__ != VizRank:
                                qApp.processEvents()        # allow processing of other events
                            if self.isOptimizationCanceled(): return self.optimizedProjectionsCount
                            if hasattr(self, "setStatusBarText") and self.optimizedProjectionsCount % 10 == 0:
                                self.setStatusBarText("Evaluated %s projections. Last accuracy was: %2.2f%%" % (orngVisFuncts.createStringFromNumber(self.optimizedProjectionsCount), acc))
                            if acc > bestAccuracy:
                                bestAccuracy = acc
                                bestProjection = newProj
                                #self.addResult(acc, other_results, len(table), [self.graph.attribute_names[i] for i in newProj], -1, {})
                            if acc > tempAccuracy or acc > 0.99 * tempAccuracy:
                                tempProjection = newProj
                                tempAccuracy = acc
                    projection = bestProjection
                    accuracy = bestAccuracy

                # take best projection and try to replace one of the attributes with a new attribute
                # when you can't further improve projections this way try adding a new attribute to the projection
                # in the first step try to find a better projection by substituting an existent attribute with a new one
                # in the second step try to find a better projection by adding a new attribute to the circle
                significantImprovement = 0
                for iteration in range(2):
                    if iteration == 1 and not self.locOptAllowAddingAttributes: continue    # if we are not allowed to increase the number of visualized attributes
                    if (len(projection) + iteration > self.locOptMaxAttrsInProj): continue
                    strTotalAtts = orngVisFuncts.createStringFromNumber(lenOfAttributes)
                    for (attrIndex, attr) in enumerate(attributes):
                        if attr in projection: continue
                        if significantImprovement and restartWhenImproved: break        # if we found a projection that is significantly better than the currently best projection then restart the search with this projection
                        tempList = []

                        # SPCA, PLS
                        if self.projOptimizationMethod != 0:
                            if iteration == 0:  # replace one attribute in each projection with attribute attr
                                testProjections = [copy(projection) for i in range(len(projection))]
                                for i in range(len(testProjections)): testProjections[i][len(projection)-1-i] = attr
                            elif iteration == 1: testProjections = [projection + [attr]]

                            for proj in testProjections:
                                proj.sort()
                                if newProjDict.has_key(str(proj)): continue
                                newProjDict[str(proj)] = 1
                                xanchors, yanchors, (attrNames, newIndices) = self.freeviz.findProjection(self.projOptimizationMethod, proj, set_anchors = 0, percentDataUsed = self.percentDataUsed)
                                table = self.graph.create_projection_as_example_table(newIndices, domain = domain, XAnchors = xanchors, YAnchors = yanchors)
                                if len(table) < self.minNumOfExamples: continue
                                self.optimizedProjectionsCount += 1
                                acc, other_results = self.evaluateProjection(table)

                                tempList.append((acc, other_results, len(table), newIndices, {"XAnchors": xanchors, "YAnchors": yanchors}))
                                if self.storeEachPermutation:
                                    self.addResult(acc, other_results, len(table), attrNames, projIndex, generalDict = {"XAnchors": xanchors, "YAnchors": yanchors})

                                if self.__class__ != VizRank:
                                    qApp.processEvents()        # allow processing of other events
                                if self.isOptimizationCanceled(): return self.optimizedProjectionsCount

                        # ordinary radviz projections
                        else:
                            testProjections = [copy(projection) for i in range(len(projection))]
                            if iteration == 0:  # replace one attribute in each projection with attribute attr
                                count = len(projection)
                                for i in range(count): testProjections[i][i] = attr
                            elif iteration == 1:
                                count = len(projection) + 1
                                for i in range(count-1): testProjections[i].insert(i, attr)

                            XAnchors = self.graph.create_xanchors(count)
                            YAnchors = self.graph.create_yanchors(count)
                            validData = self.graph.get_valid_list(testProjections[0])
                            classList = numpy.compress(validData, classListFull)

                            for testProj in testProjections:
                                if newProjDict.has_key(str(testProj)): continue
                                newProjDict[str(testProj)] = 1

                                table = self.graph.create_projection_as_example_table(testProj, validData = validData, classList = classList, XAnchors = XAnchors, YAnchors = YAnchors, domain = domain)
                                if len(table) < self.minNumOfExamples: continue
                                acc, other_results = self.evaluateProjection(table)
                                if hasattr(self, "setStatusBarText") and self.optimizedProjectionsCount % 10 == 0: self.setStatusBarText("Evaluated %s projections. Last accuracy was: %2.2f%%" % (orngVisFuncts.createStringFromNumber(self.optimizedProjectionsCount), acc))
                                if acc > accuracy:
                                    tempList.append((acc, other_results, len(table), testProj, {}))
                                if self.storeEachPermutation:
                                    self.addResult(acc, other_results, len(table), [self.graph.attribute_names[i] for i in testProj], projIndex, {})

                                self.optimizedProjectionsCount += 1
                                if self.__class__ != VizRank:
                                    qApp.processEvents()        # allow processing of other events
                                if self.isOptimizationCanceled(): return self.optimizedProjectionsCount

                        # return only the best attribute placements
                        if len(tempList) == 0: continue     # can happen if the newProjDict already had all the projections that we tried
                        (acc, other_results, lenTable, attrList, generalDict) = maxFunct(tempList)
                        if acc > 1.005*accuracy:
                            self.insertTempProjection(projections, acc, attrList)
                            self.addResult(acc, other_results, lenTable, [self.graph.attribute_names[i] for i in attrList], projIndex , generalDict)
                            if hasattr(self, "setStatusBarText"): self.setStatusBarText("Found a better projection with accuracy: %2.2f%%" % (acc))
                        if accuracy != 0 and acc > 1.01 * accuracy:  significantImprovement = 1

        else:
            print "unknown visualization method"

        return self.optimizedProjectionsCount

    # ##############################################################
    # Loading and saving projection files
    # ##############################################################

    # save the list into a file - filename can be set if you want to call this function without showing the dialog
    def save(self, name, results = None, count = 1000):
        # take care of extension
        if os.path.splitext(name)[1].lower() != ".proj": name = name + ".proj"

        if not results: results = self.results
        self.abortCurrentOperation = 0

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        # open, write and save file
        file = open(name, "wt")

        attrs = ["kValue", "percentDataUsed", "qualityMeasure", "testingMethod", "parentName", "evaluationAlgorithm", "useExampleWeighting", "projOptimizationMethod", "attrSubsetSelection", "optimizationType", "attributeCount", "attrDisc", "attrCont", "timeLimit", "projectionLimit"]
        dict = {}
        for attr in attrs: dict[attr] = self.__dict__.get(attr)
        dict["dataCheckSum"] = self.graph.raw_data.checksum()
        dict["totalProjectionsEvaluated"] = self.evaluatedProjectionsCount + self.optimizedProjectionsCount  # let's also save the total number of projections that we evaluated in order to get this list

        file.write("%s\n%s\n" % (str(dict), str(self.selectedClasses)))

        i=0
        for i in range(len(results)):
            if i >= count: break

            (acc, other_results, lenTable, attrList, tryIndex, generalDict) = results[i]

            s = "(%.3f, (" % (acc)
            for val in other_results:
                if type(val) == float: s += "%.3f ," % val
                elif type(val) == list:
                    s += "["
                    for el in val:
                        if type(el) == float: s += "%.3f, " % (el)
                        elif type(el) == int: s += "%d, " % (el)
                        else: s += "%s, " % str(el)
                    if s[-2] == ",": s = s[:-2]
                    s += "], "
            if s[-2] == ",": s = s[:-2]
            s += "), %d, %s, %d, %s)" % (lenTable, str(attrList), tryIndex, str(generalDict).replace("\n     ", "")) # be sure to remove \n in XAnchors and YAnchors otherwise load doesn't work
            file.write(s + "\n")

            if self.abortCurrentOperation: break
            if hasattr(self, "setStatusBarText"):
                self.setStatusBarText("Saved %s projections" % (orngVisFuncts.createStringFromNumber(i)))

        file.flush()
        file.close()
        self.abortCurrentOperation = 0
        return i

    # load projections from a file
    def load(self, name, ignoreCheckSum = 1, maxCount = -1):
        self.clearResults()
        self.clearArguments()
        self.abortCurrentOperation = 0

        file = open(name, "rt")
        settings = eval(file.readline()[:-1])
        if settings.get("parentName", "").lower() != self.parentName.lower():
            if self.__class__ != VizRank:
                QMessageBox.critical( self, "Optimization Dialog", 'Unable to load projection file. It was saved for %s method'%(settings["parentName"]), QMessageBox.Ok)
            else:
                print 'Unable to load projection file. It was saved for %s method' % (settings["parentName"])
            file.close()
            return [], 0

        if settings.has_key("dataCheckSum") and settings["dataCheckSum"] != self.graph.raw_data.checksum():
            if not ignoreCheckSum and self.__class__.__name__ == "OWVizRank":
                if QMessageBox.information(self, 'VizRank', 'The current data set has a different checksum than the data set that was used to evaluate projections in this file.\nDo you want to continue loading anyway, or cancel?','Continue','Cancel', '', 0,1):
                    file.close()
                    return [], 0
            else:
                print "The data set has a different checksum than the data set that was used in projection evaluation. Projection might be invalid but the file will be loaded anyway..."

        for key in settings.keys():
            setattr(self, key, settings[key])

        # find if it was computed for specific class values
        selectedClasses = eval(file.readline()[:-1])

        if self.__class__ != VizRank:
            from PyQt4.QtGui import qApp

        count = 0
        for line in file.xreadlines():
            (acc, other_results, lenTable, attrList, tryIndex, generalDict) = eval(line)
            VizRank.insertItem(self, count, acc, other_results, lenTable, attrList, tryIndex, generalDict)
            count+=1
            if maxCount != -1 and count >= maxCount: break
            if self.abortCurrentOperation: break
            if count % 100 == 0 and hasattr(self, "setStatusBarText"):
                self.setStatusBarText("Loaded %s projections" % (orngVisFuncts.createStringFromNumber(count)))
                qApp.processEvents()        # allow processing of other events
        file.close()

        self.abortCurrentOperation = 0

        # update loaded results
        return selectedClasses, count

    # remove results that have tryIndex > topProjectionIndex
    def reduceResults(self, topProjectionIndex):
        results = self.results
        self.clearResults()
        i=0
        for (accuracy, other_results, lenTable, attrList, tryIndex, generalDict) in results:
            if tryIndex <= topProjectionIndex:
                self.insertItem(i, accuracy, other_results, lenTable, attrList, tryIndex, generalDict)
                i += 1


# ###############################################################################################################################################
# ######           VIZRANK OUTLIERS            ##############################################################################################
# ###############################################################################################################################################
class VizRankOutliers:
    def __init__(self, vizrank, dialogType):
        self.vizrank = vizrank
        self.dialogType = dialogType

        self.data = None
        self.results = None

        self.projectionIndices = []
        self.matrixOfPredictions = None
        self.graphMatrix = None
        self.evaluatedExamples = []
        self.projectionCount = 20

        if self.dialogType == VIZRANK_POINT:
            self.ATTR_LIST = ATTR_LIST
            self.ACCURACY = ACCURACY
        elif self.dialogType == VIZRANK_MOSAIC:
            import orngMosaic
            self.ATTR_LIST = orngMosaic.ATTR_LIST
            self.ACCURACY = orngMosaic.SCORE


    def setResults(self, data, results):
        self.data = data
        self.results = results
        self.matrixOfPredictions = None


    def evaluateProjections(self, qApp = None):
        if self.dialogType == VIZRANK_POINT:
            graph = self.vizrank.graph

        if not self.results or not self.data: return

        projCount = min(int(self.projectionCount), len(self.results))
        classCount = max(len(self.data.domain.classVar.values), 1)
        existing = 0
        if self.matrixOfPredictions != None:
            existing = numpy.shape(self.matrixOfPredictions)[0]/classCount
            if existing < projCount:
                self.matrixOfPredictions = numpy.resize(self.matrixOfPredictions, (projCount*classCount, len(self.data)))
            elif existing > projCount:
                self.matrixOfPredictions = self.matrixOfPredictions[0:classCount*projCount,:]
        else:
            self.matrixOfPredictions = -100 * numpy.ones((projCount*classCount, len(self.data)), numpy.float)

        # compute the matrix of predictions
        results = self.results[existing:min(len(self.results), projCount)]
        index = 0
        for result in results:
            if self.dialogType == VIZRANK_POINT:
                acc, other, tableLen, attrList, tryIndex, generalDict = result
                attrIndices = [graph.attribute_name_index[attr] for attr in attrList]
                validDataIndices = graph.get_valid_indices(attrIndices)
                table = graph.create_projection_as_example_table(attrIndices, settingsDict = generalDict)    # TO DO: this does not work with polyviz!!!
                acc, probabilities = self.vizrank.kNNClassifyData(table)

            elif self.dialogType == VIZRANK_MOSAIC:
                from orngCI import FeatureByCartesianProduct
                acc, attrList, tryIndex, other = result
                probabilities = numpy.zeros((len(self.data), len(self.data.domain.classVar.values)), numpy.float)
                newFeature, quality = FeatureByCartesianProduct(self.data, attrList)
                dist = orange.ContingencyAttrClass(newFeature, self.data)
                data = self.data.select([newFeature, self.data.classVar])     # create a dataset that has only this new feature and class info
                clsVals = len(self.data.domain.classVar.values)
                validDataIndices = range(len(data))
                for i, ex in enumerate(data):
                    try:
                        prob = dist[ex[0]]
                        for j in range(clsVals):
                            probabilities[i][j] = prob[j] / max(1, float(sum(prob.values())))
                    except:
                        validDataIndices.remove(i)

            #self.matrixOfPredictions[(existing + index)*classCount:(existing + index +1)*classCount] = numpy.transpose(probabilities)
            probabilities = numpy.transpose(probabilities)
            for i in range(classCount):
                numpy.put(self.matrixOfPredictions[(existing + index)*classCount + i], validDataIndices, probabilities[i])

            index += 1
            if hasattr(self, "setStatusBarText"):
                self.setStatusBarText("Evaluated %s/%s projections..." % (orngVisFuncts.createStringFromNumber(existing + index), orngVisFuncts.createStringFromNumber(projCount)))
                self.widget.progressBarSet(100.0*(index)/max(1, float(projCount-existing)))
            if qApp:
                qApp.processEvents()

        # generate a sorted list of (probability, exampleIndex, classDistribution)
        projCount = min(int(self.projectionCount), len(self.results))
        self.evaluatedExamples = []
        for exIndex in range(len(self.data)):
            matrix = numpy.transpose(numpy.reshape(self.matrixOfPredictions[:, exIndex], (projCount, classCount)))
            valid = numpy.where(matrix[int(self.data[exIndex].getclass())] != -100, 1, 0)
            data = numpy.compress(valid, matrix[int(self.data[exIndex].getclass())])
            if len(data): aveAcc = numpy.sum(data) / float(len(data))
            else:         aveAcc = 0
            classPredictions = []
            for ind, val in enumerate(self.data.domain.classVar.values):
                data = numpy.compress(valid, matrix[ind])
                if len(data): acc = numpy.sum(data) / float(len(data))
                else:         acc = 0
                classPredictions.append((acc, val))
            self.evaluatedExamples.append((aveAcc, exIndex, classPredictions))
        self.evaluatedExamples.sort()

    # take the self.evaluatedExamples list and find examples where probability of the "correct" class is lower than probability of some other class
    # change class value of such examples to class value that has the highest probability
    def changeClassToMostProbable(self):
        if not self.data or not self.evaluatedExamples or len(self.evaluatedExamples) != len(self.data):
            print "no data or outliers not found yet. Run evaluateProjections() first."
            return

        correctedData = orange.ExampleTable(self.data)
        for (aveAcc, exInd, classPredictions) in self.evaluatedExamples:
            (acc, clsVal) = max(classPredictions)
            correctedData[exInd].setclass(clsVal)
        return correctedData


# ###############################################################################################################################################
# ######       VIZRANK LEARNERS, CLASSIFIERS       ##############################################################################################
# ###############################################################################################################################################

# class that represents kNN classifier that classifies examples based on top evaluated projections
class VizRankClassifier(orange.Classifier):
    def __init__(self, vizrank, data):
        self.VizRank = vizrank

        if self.VizRank.__class__.__name__ == "OWVizRank":
            self.VizRank.parentWidget.setData(data)
            self.VizRank.parentWidget.handleNewSignals()
            self.VizRank.timeLimit = self.VizRank.evaluationTime
            if self.VizRank.optimizeBestProjection:
                self.VizRank.optimizeTimeLimit = self.VizRank.optimizeBestProjectionTime
            else:
                self.VizRank.optimizeTimeLimit = 0
        else:
            self.VizRank.setData(data)

        self.VizRank.evaluateProjections()

        # do we want to optimize current projection. if yes then spend the same amount of time to optimize it
        if self.VizRank.optimizeTimeLimit > 0 or self.VizRank.optimizeProjectionLimit:
            self.VizRank.optimizeBestProjections()
            self.VizRank.removeTooSimilarProjections()

        #if self.VizRank.__class__.__name__ == "OWVizRank": del self.VizRank.useTimeLimit


    # for a given example run argumentation and find out to which class it most often fall
    def __call__(self, example, returnType = orange.GetBoth):
        if self.VizRank.__class__.__name__ == "OWVizRank":
            table = orange.ExampleTable(example.domain)
            table.append(example)
            self.VizRank.parentWidget.setSubsetData(table)       # show the example is we use the widget
            self.VizRank.parentWidget.handleNewSignals()
            classVal, dist = self.VizRank.findArguments(example, 0, 0)
        else:
            classVal, dist = self.VizRank.findArguments(example)

        if returnType == orange.GetBoth: return classVal, dist
        else:                            return classVal


# #############################################################################
# learner that builds VizRankClassifier
class VizRankLearner(orange.Learner):
    def __init__(self, visualizationMethod = SCATTERPLOT, vizrank = None, graph = None):
        if not vizrank:
            vizrank = VizRank(visualizationMethod, graph)
        self.VizRank = vizrank
        self.name = self.VizRank.learnerName


    def __call__(self, examples, weightID = 0):
        return VizRankClassifier(self.VizRank, examples)



#test widget
if __name__=="__main__":
    data = orange.ExampleTable(r"E:\Development\Python23\Lib\site-packages\Orange\Datasets\UCI\wine.tab")
    #data = orange.ExampleTable(r"E:\Development\Python23\Lib\site-packages\Orange\Datasets\microarray\cancer\leukemia.tab")
    """
    vizrank = VizRank(LINEAR_PROJECTION)
    vizrank.setData(data)
    vizrank.optimizationType = EXACT_NUMBER_OF_ATTRS    # MAXIMUM_NUMBER_OF_ATTRS,  EXACT_NUMBER_OF_ATTRS
    vizrank.attributeCount = 10
    vizrank.attrCont = CONT_MEAS_S2NMIX
    vizrank.projOptimizationMethod = 0
    vizrank.useExampleWeighting = 0
    vizrank.attrSubsetSelection = GAMMA_SINGLE
    vizrank.timeLimit = 1
    vizrank.evaluateProjections()
    """
    data = orange.ExampleTable(r"E:\Development\Python23\Lib\site-packages\Orange\datasets\Imatch\irski podatki\merged\merged-all.tab")
    vizrank = VizRank(RADVIZ)
    vizrank.setData(data)
    vizrank.attributeCount = 6
    vizrank.optimizationType = MAXIMUM_NUMBER_OF_ATTRS    # MAXIMUM_NUMBER_OF_ATTRS,  EXACT_NUMBER_OF_ATTRS
    #vizrank.attrSubsetSelection = GAMMA_SINGLE
    vizrank.attrSubsetSelection = DETERMINISTIC_ALL

    #vizrank.attrCont = CONT_MEAS_S2N
    vizrank.attrCont = CONT_MEAS_S2NMIX

    #vizrank.storeEachPermutation = 1
    #vizrank.load(r"E:\Development\Python23\Lib\site-packages\Orange\Datasets\microarray\cancer\leukemia - Radviz - test.proj")
    #vizrank.computeVizRanksAccuracy()
    vizrank.timeLimit = 10
    vizrank.evaluateProjections()
    #vizrank.findArguments(data[0])

