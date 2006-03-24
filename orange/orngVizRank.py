import orange, sys, random, statc
import orngVisFuncts, orngTest, orngStat
from math import sqrt
import os, orange
from math import sqrt
import Numeric, time
from copy import copy, deepcopy
from orngLinProj import FreeViz

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
ALGORITHM_FISHER = 1
ALGORITHM_HEURISTIC = 2

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

contMeasures = [("None", None), ("ReliefF", orange.MeasureAttribute_relief(k=10, m=50)),
                ("Signal to Noise Ratio", orngVisFuncts.S2NMeasure()), ("Signal to Noise OVA", orngVisFuncts.S2NMeasureMix())]
discMeasures = [("None", None), ("ReliefF", orange.MeasureAttribute_relief(k=10, m=50)),
                ("Gain ratio", orange.MeasureAttribute_gainRatio()), ("Gini index", orange.MeasureAttribute_gini())]


# array of testing methods. used by calling python's apply method depending on the value of self.testingMethod
testingMethods = [orngTest.leaveOneOut, orngTest.crossValidation, orngTest.learnAndTestOnLearnData]

# visualization methods
SCATTERPLOT = 1
RADVIZ = 2
LINEAR_PROJECTION = 3

# optimization type
EXACT_NUMBER_OF_ATTRS = 0
MAXIMUM_NUMBER_OF_ATTRS = 1

import orngScaleScatterPlotData
import orngScaleLinProjData

class VizRank:
    def __init__(self, visualizationMethod, graph = None):
        if not graph:
            if visualizationMethod == SCATTERPLOT:
                graph = orngScaleScatterPlotData.orngScaleScatterPlotData()
            elif visualizationMethod == RADVIZ:
                graph = orngScaleLinProjData.orngScaleLinProjData()
                graph.normalizeExamples = 1
                graph.scalingByVariance = 0
            elif visualizationMethod == LINEAR_PROJECTION:
                graph = orngScaleLinProjData.orngScaleLinProjData()
                graph.normalizeExamples = 0
                graph.scalingByVariance = 1
            else:
                print "an invalid visualization method was specified. VizRank can not run."
                return

        random.seed(0)      # always use the same seed to make results repeatable
        self.graph = graph
        self.freeviz = FreeViz(graph)
        self.visualizationMethod = visualizationMethod

        self.data = None
        self.results = []
        self.arguments = []                                 # a list of arguments
                
        self.kValue = 10
        self.percentDataUsed = 100
        self.qualityMeasure = AVERAGE_CORRECT
        self.testingMethod = TEN_FOLD_CROSS_VALIDATION
        self.optimizationType = MAXIMUM_NUMBER_OF_ATTRS
        self.attributeCount = 4
        self.evaluationAlgorithm = ALGORITHM_KNN
        self.attrCont = CONT_MEAS_RELIEFF
        self.attrDisc = DISC_MEAS_RELIEFF
        self.attrSubsetSelection = DETERMINISTIC_ALL        # how do we find attribute subsets to evaluate - deterministic according to attribute ranking score or using gamma distribution - if using gamma, do we want to evaluate all possible permutations of attributes or only one
        self.useSupervisedPCA = 0                           # use the supervisedPCA
        self.useExampleWeighting = 0                        # weight examples, so that the class that has a low number of examples will have higher weights
        
        self.evaluationTime = 2                             # how many minutes do we want to try to find top projections
        self.optimizeBestProjection = 0                     # do we want to try to locally improve the best projections
        self.optimizeBestProjectionTime = 2                 # how many minutes do we want to try to locally optimize the best projections
        self.externalLearner = None                         # do we use knn or some external learner
        self.selectedClasses = []                           # which classes are we trying to separate
        self.learnerName = "VizRank Learner"
        #self.onlyOnePerSubset = 1                           # save only the best placement of attributes in radviz
        self.maxResultListLen = 100000                      # number of projections to store in a list
        self.saveResultsFromFolds = 0                       # do we want to save information for each fold when evaluating projection - used to compute VizRank's accuracy
        self.abortCurrentOperation = 0
        
        if visualizationMethod == SCATTERPLOT: self.parentName = "Scatterplot"
        elif visualizationMethod == RADVIZ:    self.parentName = "Radviz"
        elif visualizationMethod == LINEAR_PROJECTION: self.parentName = "Linear Projection"

        self.argumentCount = 1              # number of arguments used when classifying 
        #self.argumentValueFormula = 1       # how to compute argument value

        self.locOptOptimizeProjectionByPermutingAttributes = 1      # try to improve projection by switching pairs of attributes in a projection
        self.locOptMaxAttrsInProj = 20                              # maximum number of attributes in projection
        self.locOptAttrsToTry = 100                                 # number of best ranked attributes to try 
        self.locOptProjCount = 20                                   # consider this number of best ranked projections
        self.attributeNameIndex = {}                                # dict with indices to attributes
                
        self.datasetName = ""
        
        # 0 - set to sqrt(N)
        # 1 - set to N / c
        self.kValueFormula = 1
        self.autoSetTheKValue = 1       # automatically set the value k


    def clearResults(self):
        self.results = []

    def clearArguments(self):
        self.arguments = []

    def removeTooSimilarProjections(self, allowedPercentOfEqualAttributes = 70):
        i=0
        while i < len(self.results):
            if self.existsABetterSimilarProjection(i, allowedPercentOfEqualAttributes):
                self.results.pop(i)
            else:
                i += 1

    # test if one of the projections in self.results[0:index] are similar to the self.results[index] projection
    def existsABetterSimilarProjection(self, index, allowedPercentOfEqualAttributes = 70):
        testAttrs = self.results[index][ATTR_LIST]
        for i in range(index):
            attrs = self.results[i][ATTR_LIST]
            #if len(attrs) != len(testAttrs): continue
            #diffArr = [testAttrs[j] != attrs[j] for j in range(len(attrs))]
            #if sum(diffArr) < int((len(attrs)+4) * 0.20): return 1
            equalAttrs = [attr in attrs for attr in testAttrs]
            if 100*sum(equalAttrs)/float(len(testAttrs)) > allowedPercentOfEqualAttributes:
                return 1
        return 0


    def createkNNLearner(self, k = -1, kValueFormula = -1):
        if k == -1:
            if kValueFormula == -1 or not self.data or len(self.data) == 0:
                kValue = self.kValue
            elif kValueFormula == 0 or not self.data.domain.classVar or self.data.domain.classVar.varType != orange.VarTypes.Discrete:
                kValue = int(sqrt(len(self.data)))
            else:
                kValue = int(len(self.data) / len(self.data.domain.classVar.values))    # k = N / c (c = # of class values)
            if self.percentDataUsed != 100:
                kValue = int(kValue * self.percentDataUsed / 100.0)
        else:
            kValue = k
            
        return orange.kNNLearner(k = kValue, rankWeight = 0, distanceConstructor = orange.ExamplesDistanceConstructor_Euclidean(normalize=0))
        

    def setData(self, data):
        self.data = data

        self.clearResults()
        self.clearArguments()
        self.graph.setData(data)
        
        self.selectedClasses = []
        if self.data and self.data.domain.classVar and self.data.domain.classVar.varType == orange.VarTypes.Discrete:
            self.selectedClasses = range(len(self.data.domain.classVar.values))

        if not data: return

        if self.autoSetTheKValue:
            if self.kValueFormula == 0 or not data.domain.classVar or data.domain.classVar.varType == orange.VarTypes.Continuous:
                self.kValue = int(sqrt(len(data)))                                 # k = sqrt(N)
            elif self.kValueFormula == 1:
                self.kValue = int(len(data) / len(data.domain.classVar.values))    # k = N / c (c = # of class values)

        self.attributeNameIndex = self.graph.attributeNameIndex
        self.correctSettingsIfNecessary()

    # save subsetdata. first example from this dataset can be used with argumentation - it can find arguments for classifying the example to the possible class values
    def setSubsetData(self, subData):
        self.subsetData = subData
        self.graph.setSubsetData(subData)
        self.clearArguments()
        
    def getEvaluatedAttributes(self):
        return orngVisFuncts.evaluateAttributes(self.data, contMeasures[self.attrCont][1], discMeasures[self.attrDisc][1])

    # return a function that is appropriate to find the best projection in a list in respect to the selected quality measure
    def getMaxFunct(self):
        if self.data.domain.classVar.varType == orange.VarTypes.Discrete and self.qualityMeasure != BRIER_SCORE: return max
        else: return min

    def addResult(self, accuracy, other_results, lenTable, attrList, tryIndex, generalDict = {}):
        funct = self.qualityMeasure != BRIER_SCORE and max or min
        self.insertItem(self.findTargetIndex(accuracy, funct), accuracy, other_results, lenTable, attrList, tryIndex, generalDict)

    # use bisection to find correct index
    def findTargetIndex(self, accuracy, funct):
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

    """
    def computeVizRanksAccuracy(self, showResults = 1):
        if not self.results: return

        if not self.saveResultsFromFolds or self.results[0][GENERAL_DICT].get("resultsByFolds", {}) == {}:
            if self.__class__.__name__ == "OWVizRank":
                import qt; qt.QMessageBox.information(self, 'VizRank', "To compute VizRank's accuracy you have to check 'Save results for each fold' checkbox and then evaluate projections.", 'OK')
            else: print "To compute VizRank's accuracy you have to set VizRank's variable saveResultsFromFolds to 1 and then evaluate projections."
            return

        if hasattr(self, "setStatusBarText"): self.setStatusBarText("Examining evaluated projections...")
        nrFolds = len(self.results[0][GENERAL_DICT]["resultsByFolds"])
        folds = [[] for i in range(nrFolds)]
        for i in range(len(self.results)):
            resultsByFolds = self.results[i][GENERAL_DICT]["resultsByFolds"]
            s = sum(resultsByFolds)
            for j in range(nrFolds):
                folds[j].append((s-resultsByFolds[j], i))

        if self.useExampleWeighting:
            data, weightID = orange.Preprocessor_addClassWeight(self.data, equalize=1)
        else: data , weightID = orngTest.demangleExamples(self.data)
        basevalue = self.data.domain.classVar.baseValue

        indices = orange.MakeRandomIndicesCV(data, nrFolds, stratified = orange.MakeRandomIndices.StratifiedIfPossible)
        testResults = orngTest.ExperimentResults(nrFolds, ["VizRank"], list(data.domain.classVar.values), weightID!=0, basevalue)
        testResults.results = [orngTest.TestedExample(indices[i], int(data[i].getclass()), 1, data[i].getweight(weightID)) for i in range(len(data))]

        # save the data set and results
        fullData = self.data
        fullResults = self.results

        for fold in range(nrFolds):
            learnset = self.data.selectref(indices, fold, negate=1)
            testset  = self.data.selectref(indices, fold, negate=0)
            if not len(learnset): continue
            self.graph.setSubsetData(None)
            self.graph.setData(learnset)
            self.graph.setSubsetData(testset)
            self.data = learnset

            #self.graph.setData(learnset, keepMinMaxVals = 1)
            folds[fold].sort(); folds[fold].reverse()
            self.results = [self.results[index] for (val, index) in folds[fold][:self.argumentCount+100]]
                        
            for i in range(len(data)):
                if indices[i] != fold: continue
                classValue, dist = VizRank.findArguments(self, data[i])
                testResults.results[i].setResult(0, classValue, dist)

            self.data = fullData
            self.results = fullResults
            if hasattr(self, "setStatusBarText"): self.setStatusBarText("Evaluated %d/%d folds..." % (fold, nrFolds ))

        self.graph.setData(fullData)
        if hasattr(self, "parentWidget"): self.parentWidget.showSelectedAttributes()
        if hasattr(self, "setStatusBarText"): self.setStatusBarText("")

        acc, other_results, resultsByFolds = self.computeAccuracyFromResults(self.data, testResults)

        if showResults and self.__class__.__name__ == "OWVizRank":
            s = "%s of the VizRank on this list of evaluated projections is %.3f" % (measuresDict[self.qualityMeasure], acc)
            if self.qualityMeasure in (CLASS_ACCURACY, AVERAGE_CORRECT): s += "%"
            import qt
            qt.QMessageBox.information(self, "VizRank's accuracy", s, 'OK')
        return acc
    """

    # kNNEvaluate - evaluate class separation in the given projection using a heuristic or k-NN method
    def kNNComputeAccuracy(self, table):
        # select a subset of the data if necessary
        if self.percentDataUsed != 100:
            indices = orange.MakeRandomIndices2(table, 1.0-float(self.percentDataUsed)/100.0)
            testTable = table.select(indices)
        else:
            testTable = table

        if len(testTable) == 0: return 0,0

        if self.evaluationAlgorithm == ALGORITHM_KNN or self.externalLearner:
            if self.externalLearner: learner = self.externalLearner
            else:                    learner = self.createkNNLearner(); weight = 0

            if self.useExampleWeighting:
                testTable, weightID = orange.Preprocessor_addClassWeight(testTable, equalize=1)
                results = apply(testingMethods[self.testingMethod], [[learner], (testTable, weightID)])
            else:
                results = apply(testingMethods[self.testingMethod], [[learner], testTable])
            
            # compute classification success using selected measure
            if testTable.domain.classVar.varType == orange.VarTypes.Discrete:
                return self.computeAccuracyFromResults(testTable, results)
                
            # for continuous class we can't compute brier score and classification accuracy
            else:
                val = 0.0
                if not results.results or not results.results[0].probabilities[0]: return 0,0
                for res in results.results:  val += res.probabilities[0].density(res.actualClass)
                val/= float(len(results.results))
                return 100.0*val, (100.0*val), None

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
            nattr = orange.EnumVariable(values=['i' for i in range(NUMBER_OF_INTERVALS*NUMBER_OF_INTERVALS)])
            nattr.getValueFrom = orange.ClassifierByLookupTable2(nattr, testTable.domain[0], testTable.domain[1])
            for i in range(NUMBER_OF_INTERVALS*NUMBER_OF_INTERVALS): nattr.getValueFrom.lookupTable[i] = i
            
            for dist in orange.ContingencyAttrClass(nattr, testTable):
                dist = list(dist)
                if sum(dist) == 0: continue
                m = max(dist)
                prediction[dist.index(m)] += m * m / float(sum(dist))

            prediction = [val*100.0 for val in prediction]             # turn prediction array into percents
            acc = sum(prediction) / float(len(testTable))               # compute accuracy for all classes
            val = 0.0; s = 0.0
            for index in self.selectedClasses:                          # compute accuracy for selected classes
                val += prediction[index]; s += currentClassDistribution[index]
            for i in range(len(prediction)): prediction[i] /= float(currentClassDistribution[i])    # turn to probabilities
            return val/float(s), (acc, prediction, currentClassDistribution), None


        elif self.evaluationAlgorithm == ALGORITHM_FISHER:
            val = OWVisTools.computeFisherQuality(testTable)
            return val, None, None

    def computeAccuracyFromResults(self, table, results):
        prediction = [0.0 for i in range(len(table.domain.classVar.values))]
        resultsByFolds = [0.0 for i in range(results.numberOfIterations)]
        countsByFold =  [0 for i in range(results.numberOfIterations)]

        if self.qualityMeasure == AVERAGE_CORRECT:
            for res in results.results:
                prediction[res.actualClass] += res.probabilities[0][res.actualClass]
                resultsByFolds[res.iterationNumber] += res.probabilities[0][res.actualClass]
                countsByFold[res.iterationNumber] += 1
            prediction = [val*100.0 for val in prediction]

        elif self.qualityMeasure == BRIER_SCORE:
            #return orngStat.BrierScore(results)[0], results
            for res in results.results:
                val = 0
                for prob in res.probabilities[0]: val += prob*prob
                val = val - 2*res.probabilities[0][res.actualClass] + 1
                prediction[res.actualClass] += val
                resultsByFolds[res.iterationNumber] += val
                countsByFold[res.iterationNumber] += 1
            
        elif self.qualityMeasure == CLASS_ACCURACY:
            #return 100*orngStat.CA(results)[0], results
            for res in results.results:
                prediction[res.actualClass] += res.classes[0]==res.actualClass
                resultsByFolds[res.iterationNumber] += res.classes[0]==res.actualClass
                countsByFold[res.iterationNumber] += 1
            prediction = [val*100.0 for val in prediction]
        elif self.qualityMeasure == AUC:
            return orngStat.AROCFromCDT(orngStat.computeCDT(results)[0])[7], None, None
            
        # compute accuracy only for classes that are selected as interesting. other class values do not participate in projection evaluation
        acc = sum(prediction) / float(len(results.results))                 # accuracy over all class values
        val = sum([prediction[index] for index in self.selectedClasses])    # accuracy over all selected classes

        currentClassDistribution = [int(v) for v in orange.Distribution(table.domain.classVar, table)]
        s = sum([currentClassDistribution[index] for index in self.selectedClasses])

        prediction = [prediction[i] / float(max(1, currentClassDistribution[i])) for i in range(len(prediction))] # turn to probabilities
        resultsByFolds = [resultsByFolds[i] / float(max(countsByFold[i],1)) for i in range(results.numberOfIterations)]

        return val/float(s), (acc, prediction, list(currentClassDistribution)), resultsByFolds
        #return 100.0*statc.mean(resultsByFolds), (acc, prediction, list(currentClassDistribution)), resultsByFolds
        

    # Argumentation functions
    def findArguments(self, example):
        self.clearArguments()
        if not self.data or not self.data.domain or not self.data.domain.classVar:
            return orange.MajorityLearner(self.data)(example, orange.GetBoth)

        self.arguments = [[] for i in range(len(self.data.domain.classVar.values))]
                
        if len(self.results) == 0:
            print 'To classify an example using VizRank you first have to evaluate some projections.'
            return orange.MajorityLearner(self.data)(example, orange.GetBoth)

        usedArguments = 0; index = 0
        vals = [0.0 for i in range(len(self.arguments))]
        while usedArguments < self.argumentCount and index < len(self.results):
            (accuracy, other_results, lenTable, attrList, tryIndex, generalDict) = self.results[index]

            if 1 in [example[attr].isSpecial() for attr in attrList]: index+=1; continue
            attrIndices = [self.attributeNameIndex[attr] for attr in attrList]
            attrVals = [self.graph.scaleExampleValue(example, ind) for ind in attrIndices]
            table = self.graph.createProjectionAsExampleTable(attrIndices, settingsDict = generalDict)
            [xTest, yTest] = self.graph.getProjectedPointPosition(attrIndices, attrVals, settingsDict = generalDict)
            
            learner = self.externalLearner or self.createkNNLearner(kValueFormula = 0)
            if self.useExampleWeighting: table, weightID = orange.Preprocessor_addClassWeight(table, equalize=1)
            else: weightID = 0
            classifier = learner(table, weightID)
            (classValue, dist) = classifier(orange.Example(table.domain, [xTest, yTest, "?"]), orange.GetBoth)
            classValue = int(classValue)
            #if max(dist) < 0.8: index += 1; continue        # ignore projections that don't have a reliable prediction
                        
            for i in range(len(self.arguments)):
                self.arguments[i].insert(self.getArgumentIndex(dist[i], i), (None, dist[i], dist, attrList, index))
                vals[i] += dist[i]

            index += 1; usedArguments += 1

        suma = sum(vals)
        if suma == 0:
            dist = orange.Distribution(self.data.domain.classVar.name, self.data)
            vals = [dist[i] for i in range(len(dist))]; suma = sum(vals)

        classValue = example.domain.classVar[vals.index(max(vals))]
        dist = orange.DiscDistribution([val/float(suma) for val in vals])
        dist.variable = self.data.domain.classVar
        return classValue, dist


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
        if not self.data: return
        # check if we have discrete attributes. if yes, then make sure we are not using s2nMix measure and GAMMA_SINGLE
        if 1 in [self.data.domain[attr].varType == orange.VarTypes.Discrete for attr in self.data.domain.attributes]:
            if self.attrCont == CONT_MEAS_S2NMIX:           self.attrCont = CONT_MEAS_S2N
            if self.attrSubsetSelection == GAMMA_SINGLE:    self.attrSubsetSelection = GAMMA_ALL

    def isOptimizationCanceled(self):
        return (time.time() - self.startTime) / 60 >= self.optimizeBestProjectionTime

    def isEvaluationCanceled(self):
        return (time.time() - self.startTime) / 60 >= self.evaluationTime


    # get a new subset of attributes. if attributes are not evaluated yet then evaluate them and save info to evaluationData dict.
    def selectNextAttributeSubset(self, minLength, maxLength):
        z = self.evaluationData.get("z", minLength-1)
        u = self.evaluationData.get("u", minLength-1)
        self.evaluationData["combinations"] = []
        self.evaluationData["index"] = 0

        # if we use heuristic to find attribute orders
        if self.attrCont == CONT_MEAS_S2NMIX or self.attrSubsetSelection == GAMMA_SINGLE:
            if not self.evaluationData.has_key("attrs"):
                attributes, attrsByClass = orngVisFuncts.findAttributeGroupsForRadviz(self.data, orngVisFuncts.S2NMeasureMix())
                attributes = [self.attributeNameIndex[name] for name in attributes]
                attrsByClass = [[self.attributeNameIndex[name] for name in arr] for arr in attrsByClass]
                self.evaluationData["attrs"] = (attributes, attrsByClass)
            else:
                attributes, attrsByClass = self.evaluationData["attrs"]

            if z >= len(attributes): return None      # did we already try all the attributes
            numClasses = len(self.data.domain.classVar.values)
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
                evaluatedAttributes = orngVisFuncts.evaluateAttributes(self.data, contMeasures[self.attrCont][1], discMeasures[self.attrDisc][1])
                attributes = [self.attributeNameIndex[name] for name in evaluatedAttributes]
                self.evaluationData["attrs"] = attributes
                self.totalPossibilities = 0
                
                # build list of indices for permutations of different number of attributes
                permutationIndices = {}
                for i in range(minLength, maxLength+1):
                    if self.useSupervisedPCA:
                        permutationIndices[i] = [range(i)]
                    else:
                        permutationIndices[i] = orngVisFuncts.generateDifferentPermutations(range(i))
                    self.totalPossibilities += orngVisFuncts.combinationsCount(i, len(attributes)) * len(permutationIndices[i])
                self.evaluationData["permutationIndices"] = permutationIndices
            else:
                attributes = self.evaluationData["attrs"]

            if z >= len(attributes): return None      # did we already try all the attributes

            if self.attrSubsetSelection == DETERMINISTIC_ALL:
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
        maxTries = 50
        triedDict = self.evaluationData.get("triedCombinations", {})
        
        if self.attrCont == CONT_MEAS_S2NMIX or self.attrSubsetSelection == GAMMA_SINGLE:
            numClasses = len(self.data.domain.classVar.values)
            attributes, attrsByClass = self.evaluationData["attrs"]
            for i in range(maxTries):
                attrList = [[] for c in range(numClasses)]; attrs = []
                placed = 0; tried = 0
                while placed < min(attrCount, len(self.data.domain.attributes)):
                    ind = tried%numClasses
                    found = 0
                    while not found:
                        attr = attrsByClass[ind][int(random.gammavariate(1,10 + len(attrsByClass[ind])/2000))%len(attrsByClass[ind])]
                        if attr not in attrList[ind]:
                            attrList[ind].append(attr); placed += 1; attrs.append(attr); found = 1
                    tried += 1
                attrs.sort()
                if not triedDict.has_key(tuple(attrs)) and len(attrs) == attrCount:
                    triedDict[tuple(attrs)] = 1
                    self.evaluationData["triedCombinations"] = triedDict
                    return [attrList]
        else:
            attributes = self.evaluationData["attrs"]
            for i in range(maxTries):
                attrList = []
                placed = 0; 
                while placed < attrCount:
                    attr = attributes[int(random.gammavariate(1,10 + len(attributes)/1000))%len(attributes)]
                    if attr not in attrList:
                        attrList.append(attr); placed += 1
                attrList.sort()
                if not triedDict.has_key(tuple(attrList)):
                    triedDict[tuple(attrList)] = 1
                    self.evaluationData["triedCombinations"] = triedDict
                    return [attrList]
        return None

    # generate possible permutations of the current attribute subset. use evaluationData dict to find which attribute subset to use. 
    def getNextPermutations(self):
        combinations, index = self.evaluationData["combinations"], self.evaluationData["index"]
        if not combinations or index >= len(combinations):
            return None     # did we test all the projections
        combination = combinations[index]
        permutations = []

        if self.attrCont == CONT_MEAS_S2NMIX or self.attrSubsetSelection == GAMMA_SINGLE:
            # if we don't want to test all placements then we only create a permutation of groups and attributes in each group
            if self.attrSubsetSelection == GAMMA_SINGLE:
                newCombinations = [combination]
                for c in range(10):
                    tempCombination = deepcopy(newCombinations[-1])
                    newCombination = []
                    for i in range(len(tempCombination)):
                        group = tempCombination.pop(random.randint(0, len(tempCombination)-1))
                        newGroup = []
                        for j in range(len(group)):
                            newGroup.append(group.pop(random.randint(0, len(group)-1)))
                        newCombination.append(newGroup)
                    newCombinations.append(newCombination)

                for comb in newCombinations:
                    allAttrs = []
                    for group in comb: allAttrs += group
                    permutations.append(allAttrs)

                """
                newCombination = []
                for i in range(len(combination)):
                    group = combination.pop(random.randint(0, len(combination)-1))
                    newGroup = []
                    for j in range(len(group)):
                        newGroup.append(group.pop(random.randint(0, len(group)-1)))
                    newCombination.append(newGroup)
                combination = newCombination
                """
                
            # create only one permutation, because its all we need
            elif self.useSupervisedPCA:       
                allAttrs = []
                for l in combination: allAttrs += l
                permutations.append(allAttrs)
            else:
                for proj in orngVisFuncts.createProjections(len(self.data.domain.classVar.values), sum([len(group) for group in combination])):
                    try: permutations.append([combination[i][j] for (i,j) in proj])
                    except: pass
        else:
            permutationIndices = self.evaluationData["permutationIndices"]
            for ind in permutationIndices[len(combination)]:
                permutations.append([combination[val] for val in ind])          # try to optimize using map (e.g. map(list.__getitem__, combination, ind)

        self.evaluationData["index"] = index + 1
        return permutations


    def evaluateProjections(self):
        random.seed(0)      # always use the same seed to make results repeatable
        if not self.data: return
        self.correctSettingsIfNecessary()
        evaluatedProjections = 0
        self.startTime = time.time()
        self.evaluationData = {}            # clear all previous data about tested permutations and stuff

        maxFunct = self.getMaxFunct()
        self.clearResults()
        self.clearArguments()
        if self.__class__.__name__ == "OWVizRank":
            from qt import qApp, QMessageBox
            if self.attributeCount >= 10 and not (self.useSupervisedPCA) and self.attrSubsetSelection != GAMMA_SINGLE and QMessageBox.critical(self, 'VizRank', 'You chose to evaluate projections with a high number of attributes. Since VizRank has to evaluate different placements\nof these attributes there will be a high number of projections to evaluate. Do you still want to proceed?','Continue','Cancel', '', 0,1):
                return
            self.disableControls()
            self.parentWidget.progressBarInit()
        
        if self.visualizationMethod == SCATTERPLOT:
            evaluatedAttributes = orngVisFuncts.evaluateAttributes(self.data, contMeasures[self.attrCont][1], discMeasures[self.attrDisc][1])
            contVars = [orange.FloatVariable(attr.name) for attr in self.data.domain.attributes]
            contDomain = orange.Domain(contVars + [self.data.domain.classVar])
            attrCount = len(self.data.domain.attributes)

            count = len(evaluatedAttributes)*(len(evaluatedAttributes)-1)/2
            strCount = orngVisFuncts.createStringFromNumber(count)
            
            for i in range(len(evaluatedAttributes)):
                for j in range(i):
                    attr1 = self.attributeNameIndex[evaluatedAttributes[j]]; attr2 = self.attributeNameIndex[evaluatedAttributes[i]]
                    evaluatedProjections += 1
                    if self.isEvaluationCanceled():
                        self.finishEvaluation(evaluatedProjections)
                        return
                    
                    table = self.graph.createProjectionAsExampleTable([attr1, attr2])
                    accuracy, other_results, resultsByFolds = self.kNNComputeAccuracy(table)
                    self.addResult(accuracy, other_results, len(table), [self.data.domain[attr1].name, self.data.domain[attr2].name], evaluatedProjections, (resultsByFolds and {"resultsByFolds" : resultsByFolds}) or {})
                    
                    if self.__class__.__name__ == "OWVizRank":
                        self.setStatusBarText("Evaluated %s/%s projections..." % (orngVisFuncts.createStringFromNumber(evaluatedProjections), strCount))
                        self.parentWidget.progressBarSet(100.0*evaluatedProjections/float(count))

        # #################### RADVIZ, LINEAR_PROJECTION  ################################
        elif self.visualizationMethod in (RADVIZ, LINEAR_PROJECTION):
            if self.useSupervisedPCA:
                self.freeviz.useGeneralizedEigenvectors = 1
                self.graph.normalizeExamples = 0
                
            # replace attribute names with indices in domain - faster searching
            classIndex = self.attributeNameIndex[self.data.domain.classVar.name]            

            # variables and domain for the table
            domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.data.domain.classVar])
            minLength = (self.optimizationType == EXACT_NUMBER_OF_ATTRS and self.attributeCount) or 3
            maxLength = self.attributeCount
            classListFull = Numeric.transpose(self.data.toNumeric("c")[0])[0]

            # each call to selectNextAttributeSubset gets a new combination of attributes in a range from minLength to maxLength. if we return None for a given number of attributes this
            # doesn't mean yet that there are no more possible combinations. it may be just that we wanted a combination of 6 attributes in a domain with 4 attributes. therefore we have
            # to try maxLength-minLength+1 times and if we fail every time then there are no more valid projections

            newProjectionsExist = 1
            while newProjectionsExist:
                for experiment in range(maxLength-minLength+1):     
                    if self.selectNextAttributeSubset(minLength, maxLength) != None: break
                    newProjectionsExist = 0         
                permutations = self.getNextPermutations()
                while permutations:
                    attrIndices = permutations[0]
                       
                    if self.useSupervisedPCA:
                        xanchors, yanchors, (attrNames, newIndices) = self.freeviz.findSPCAProjection(attrIndices, setGraphAnchors = 0)
                        table = self.graph.createProjectionAsExampleTable(newIndices, settingsDict = {"domain": domain, "XAnchors": xanchors, "YAnchors": yanchors})
                        evaluatedProjections += 1
                        accuracy, other_results, resultsByFolds = self.kNNComputeAccuracy(table)
                        self.addResult(accuracy, other_results, len(table), attrNames, evaluatedProjections, generalDict = {"XAnchors": xanchors, "YAnchors": yanchors})  # here we don't save resultsByFolds since this would be cheating - models would be different for each fold.
                        if self.isEvaluationCanceled(): self.finishEvaluation(evaluatedProjections); return
                        if self.__class__.__name__ == "OWVizRank": self.setStatusBarText("Evaluated %s projections..." % (orngVisFuncts.createStringFromNumber(evaluatedProjections)))
                    else:
                        XAnchors = self.graph.createXAnchors(len(attrIndices))
                        YAnchors = self.graph.createYAnchors(len(attrIndices))
                        validData = self.graph.getValidList(attrIndices)
                        classList = Numeric.compress(validData, classListFull)
                        selectedData = Numeric.compress(validData, Numeric.take(self.graph.noJitteringScaledData, attrIndices))
                        sum_i = self.graph._getSum_i(selectedData)

                        tempList = []

                        # for every permutation compute how good it separates different classes
                        for permutation in permutations:
                            if self.isEvaluationCanceled():
                                self.finishEvaluation(evaluatedProjections)
                                return
                            
                            table = self.graph.createProjectionAsExampleTable(permutation, settingsDict = {"validData": validData, "classList": classList, "sum_i": sum_i, "XAnchors": XAnchors, "YAnchors": YAnchors, "domain": domain})
                            accuracy, other_results, resultsByFolds = self.kNNComputeAccuracy(table)
                            
                            # save the permutation
                            if self.saveResultsFromFolds:
                                self.addResult(accuracy, other_results, len(table), [self.graph.attributeNames[i] for i in permutation], evaluatedProjections, (resultsByFolds and {"resultsByFolds" : resultsByFolds}) or {})
                            else:
                                tempList.append((accuracy, other_results, len(table), [self.graph.attributeNames[i] for i in permutation]))

                            evaluatedProjections += 1
                            if self.__class__.__name__ == "OWVizRank":
                                self.setStatusBarText("Evaluated %s projections..." % (orngVisFuncts.createStringFromNumber(evaluatedProjections)))
                                qApp.processEvents()        # allow processing of other events

                        if not self.saveResultsFromFolds and len(tempList) > 0:   # return only the best attribute placements
                            (acc, other_results, lenTable, attrList) = maxFunct(tempList)
                            self.addResult(acc, other_results, lenTable, attrList, evaluatedProjections)

                    permutations = self.getNextPermutations()  

        self.finishEvaluation(evaluatedProjections)
    

    def finishEvaluation(self, evaluatedProjections):
        if self.__class__.__name__ == "OWVizRank":
            secs = time.time() - self.startTime
            self.setStatusBarText("Finished evaluation (evaluated %s projections in %d min, %d sec)" % (orngVisFuncts.createStringFromNumber(evaluatedProjections), secs/60, secs%60))
            self.parentWidget.progressBarFinished()
            self.enableControls()
            self.finishedAddingResults()
            from qt import qApp;  qApp.processEvents()
            if self.parentWidget: self.parentWidget.showSelectedAttributes()
            

    def getProjectionQuality(self, attrList, useAnchorData = 0):
        if not self.data: return 0.0, None
        table = self.graph.createProjectionAsExampleTable([self.attributeNameIndex[attr] for attr in attrList], settingsDict = {"useAnchorData": useAnchorData})
        return self.kNNComputeAccuracy(table)


    def optimizeBestProjections(self, restartWhenImproved = 1):
        random.seed(0)      # always use the same seed to make results repeatable
        count = min(len(self.results), self.locOptProjCount)
        if not count: return
        self.correctSettingsIfNecessary()
        attrLists = [self.results[i][ATTR_LIST] for i in range(count)]                                   # create a list of attributes that are in the top projections
        attrLists = [[self.attributeNameIndex[name] for name in projection] for projection in attrLists]    # find indices from the attribute names
        accuracys = [self.getProjectionQuality(self.results[i][ATTR_LIST])[0] for i in range(count)]
        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.data.domain.classVar])
        attributes = [self.attributeNameIndex[name] for name in orngVisFuncts.evaluateAttributes(self.data, contMeasures[self.attrCont][1], discMeasures[self.attrDisc][1])[:self.locOptAttrsToTry]]
        self.startTime = time.time()
        evaluatedProjections = 0
        lenOfAttributes = len(attributes)
        maxFunct = self.getMaxFunct()

        if self.__class__.__name__ == "OWVizRank": 
            self.disableControls()
            from qt import qApp
        
        if self.visualizationMethod == SCATTERPLOT:
            classIndex = self.attributeNameIndex[self.data.domain.classVar.name]
            classListFull = Numeric.transpose(self.data.toNumeric("c")[0])[0]

            for i in range(len(attrLists)):
                projection = attrLists[i]
                accuracy = accuracys[i]
                optimizedProjection = 1
                
                while optimizedProjection:
                    significantImprovement = 0
                    
                    strTotalAtts = orngVisFuncts.createStringFromNumber(lenOfAttributes)
                    listOfCandidates = []
                    for (attrIndex, attr) in enumerate(attributes):
                        if attr in projection: continue
                        if significantImprovement and restartWhenImproved: break        # if we found a projection that is significantly better than the currently best projection then restart the search with this projection

                        projections = [[projection[0], attr], [attr, projection[1]]]
                        
                        tempList = []
                        for testProj in projections:
                            table = self.graph.createProjectionAsExampleTable(testProj, settingsDict = {"domain": domain})
                            acc, other_results, resultsByFolds = self.kNNComputeAccuracy(table)
                            
                            tempList.append((acc, other_results, len(table), testProj))  # save the permutation
                            if self.saveResultsFromFolds:
                                self.addResult(acc, other_results, len(table), testProj, 0, (resultsByFolds and {"resultsByFolds" : resultsByFolds}) or {})

                            evaluatedProjections += 1
                            if self.__class__ != VizRank: qApp.processEvents()        # allow processing of other events
                            if self.isOptimizationCanceled():
                                self.finishEvaluation(evaluatedProjections)
                                return

                        # return only the best attribute placements
                        (acc, other_results, lenTable, attrList) = maxFunct(tempList)
                        if not self.saveResultsFromFolds and min(acc, accuracy)/max(acc, accuracy) > 0.98:  # if the found projection is at least 98% as good as the one optimized, add it to the list of projections anyway
                            self.addResult(acc, other_results, lenTable, [self.graph.attributeNames[i] for i in attrList], 1)
                        
                        if maxFunct(acc, accuracy) == acc:
                            listOfCandidates.append((acc, attrList))
                            if hasattr(self, "setStatusBarText"): self.setStatusBarText("Found a better projection with accuracy: %2.2f%%" % (acc))
                            if max(acc, accuracy)/min(acc, accuracy) > 1.0001: optimizedProjection = 1
                            if max(acc, accuracy)/min(acc, accuracy) > 1.005:  significantImprovement = 1
                        elif hasattr(self, "setStatusBarText"): self.setStatusBarText("Evaluated %s projections (attribute %s/%s). Last accuracy was: %2.2f%%" % (orngVisFuncts.createStringFromNumber(evaluatedProjections), orngVisFuncts.createStringFromNumber(attrIndex), strTotalAtts, acc))
                            

                    # select the best new projection and say this is now our new projection to optimize    
                    if len(listOfCandidates) > 0:
                        (accuracy, projection) = maxFunct(listOfCandidates)
                        if hasattr(self, "setStatusBarText"):  self.setStatusBarText("Increased accuracy to %2.2f%%" % (accuracy))

        # #################### RADVIZ, LINEAR_PROJECTION  ################################
        elif self.visualizationMethod in (RADVIZ, LINEAR_PROJECTION):
            numClasses = len(self.data.domain.classVar.values)

            classListFull = Numeric.transpose(self.data.toNumeric("c")[0])[0]
            newProjDict = {}

            for i in range(len(attrLists)):
                projection = attrLists[i]
                accuracy = accuracys[i]
                optimizedProjection = 1

                if self.locOptOptimizeProjectionByPermutingAttributes == 1 and not self.useSupervisedPCA:
                    bestProjection = projection; tempProjection = projection
                    bestAccuracy = accuracy; tempAccuracy = accuracy
                    triedPermutationsDict = {}
                    failedConsecutiveTries = 0
                    XAnchors = self.graph.createXAnchors(len(projection))
                    YAnchors = self.graph.createYAnchors(len(projection))
                    validData = self.graph.getValidList(projection)
                    classList = Numeric.compress(validData, classListFull)
                    while failedConsecutiveTries < 30:
                        #newProj = orngVisFuncts.switchTwoElements(tempProjection, nrOfTimes = 3)
                        newProj = orngVisFuncts.switchTwoElementsInGroups(tempProjection, numClasses, 3)
                        if triedPermutationsDict.has_key(str(newProj)):
                            failedConsecutiveTries += 1
                        else:
                            failedConsecutiveTries = 0
                            triedPermutationsDict[str(newProj)] = 1
                            
                            table = self.graph.createProjectionAsExampleTable(newProj, settingsDict = {"validData": validData, "classList": classList, "XAnchors": XAnchors, "YAnchors": YAnchors, "domain": domain})
                            acc, other_results, resultsByFolds = self.kNNComputeAccuracy(table)
                            evaluatedProjections += 1
                            if self.__class__ != VizRank: qApp.processEvents()        # allow processing of other events
                            if self.isOptimizationCanceled(): self.finishEvaluation(evaluatedProjections); return
                            self.addResult(acc, other_results, len(table), [self.graph.attributeNames[i] for i in newProj], 0, {})
                            self.setStatusBarText("Evaluated %s projections. Last accuracy was: %2.2f%%" % (orngVisFuncts.createStringFromNumber(evaluatedProjections), acc))
                            if acc > bestAccuracy:
                                bestAccuracy = acc
                                bestProjection = newProj
                            if acc > tempAccuracy or acc / tempAccuracy > 0.99:
                                tempProjection = newProj
                                tempAccuracy = acc
                    projection = bestProjection
                    accuracy = bestAccuracy
                
                while optimizedProjection:
                    optimizedProjection = 0
                    significantImprovement = 0
                    
                    # in the first step try to find a better projection by substituting an existent attribute with a new one
                    # in the second step try to find a better projection by adding a new attribute to the circle
                    for iteration in range(2):
                        if (len(projection) + iteration > self.locOptMaxAttrsInProj): continue    
                        if iteration == 1 and optimizedProjection: continue             # if we already found a better projection with replacing an attribute then don't try to add a new atribute
                        strTotalAtts = orngVisFuncts.createStringFromNumber(lenOfAttributes)
                        listOfCandidates = []
                        for (attrIndex, attr) in enumerate(attributes):
                            if attr in projection: continue
                            if significantImprovement and restartWhenImproved: break        # if we found a projection that is significantly better than the currently best projection then restart the search with this projection
                            tempList = []

                            # supervised PCA
                            if self.useSupervisedPCA:
                                if iteration == 0:  # replace one attribute in each projection with attribute attr
                                    #projections = [copy(projection) for i in range(max(1, len(projection)/3))]
                                    projections = [copy(projection) for i in range(len(projection))]
                                    for i in range(len(projections)): projections[i][len(projection)-1-i] = attr
                                elif iteration == 1: projections = [projection + [attr]]

                                for proj in projections:
                                    if newProjDict.has_key(str(proj)): continue
                                    newProjDict[str(proj)] = 1
                                    xanchors, yanchors, (attrNames, newIndices) = self.freeviz.findSPCAProjection(proj, setGraphAnchors = 0)
                                    table = self.graph.createProjectionAsExampleTable(newIndices, settingsDict = {"domain": domain, "XAnchors": xanchors, "YAnchors": yanchors})
                                    evaluatedProjections += 1
                                    acc, other_results, resultsByFolds = self.kNNComputeAccuracy(table)
                                    
                                    tempList.append((acc, other_results, len(table), newIndices, {"XAnchors": xanchors, "YAnchors": yanchors}))
                                    if self.saveResultsFromFolds:
                                        self.addResult(acc, other_results, len(table), attrNames, 0, {"XAnchors": xanchors, "YAnchors": yanchors, "resultsByFolds" : resultsByFolds})
    
                                    #self.addResult(acc, other_results, len(table), attrNames, evaluatedProjections, generalDict = {"XAnchors": xanchors, "YAnchors": yanchors})
                                    if self.__class__ != VizRank: qApp.processEvents()        # allow processing of other events
                                    if self.isOptimizationCanceled(): self.finishEvaluation(evaluatedProjections); return

                            # ordinary radviz projections
                            else:
                                projections = [copy(projection) for i in range(len(projection))]
                                if iteration == 0:  # replace one attribute in each projection with attribute attr
                                    count = len(projection)
                                    for i in range(count): projections[i][i] = attr
                                elif iteration == 1:
                                    count = len(projection) + 1
                                    for i in range(count-1): projections[i].insert(i, attr)

                                XAnchors = self.graph.createXAnchors(count)
                                YAnchors = self.graph.createYAnchors(count)
                                validData = self.graph.getValidList(projections[0])
                                classList = Numeric.compress(validData, classListFull)
                                
                                for testProj in projections:
                                    if newProjDict.has_key(str(testProj)): continue
                                    newProjDict[str(testProj)] = 1
                                    
                                    table = self.graph.createProjectionAsExampleTable(testProj, settingsDict = {"validData": validData, "classList": classList, "XAnchors": XAnchors, "YAnchors": YAnchors, "domain": domain})
                                    acc, other_results, resultsByFolds = self.kNNComputeAccuracy(table)
                                    tempList.append((acc, other_results, len(table), testProj, {}))
                                    if self.saveResultsFromFolds:
                                        self.addResult(acc, other_results, len(table), [self.graph.attributeNames[i] for i in testProj], 0, (resultsByFolds and {"resultsByFolds" : resultsByFolds}) or {})

                                    evaluatedProjections += 1
                                    if self.__class__ != VizRank: qApp.processEvents()        # allow processing of other events
                                    if self.isOptimizationCanceled(): self.finishEvaluation(evaluatedProjections); return

                            # return only the best attribute placements
                            if len(tempList) == 0: continue     # can happen if the newProjDict already had all the projections that we tried
                            (acc, other_results, lenTable, attrList, generalDict) = maxFunct(tempList)

                            if not self.saveResultsFromFolds and min(acc, accuracy)/max(acc, accuracy) > 0.99:   # if the found projection is at least 99% as good as the one optimized, add it to the list of projections anyway
                                self.addResult(acc, other_results, lenTable, [self.graph.attributeNames[i] for i in attrList], 0, generalDict = generalDict)
                            
                            if maxFunct(acc, accuracy) == acc:
                                listOfCandidates.append((acc, attrList))
                                if hasattr(self, "setStatusBarText"): self.setStatusBarText("Found a better projection with accuracy: %2.2f%%" % (acc))
                                if max(acc, accuracy)/min(acc, accuracy) > 1.001:  optimizedProjection = 1
                                if max(acc, accuracy)/min(acc, accuracy) > 1.005:  significantImprovement = 1
                            elif hasattr(self, "setStatusBarText"): self.setStatusBarText("Evaluated %s projections (attribute %s/%s). Last accuracy was: %2.2f%%" % (orngVisFuncts.createStringFromNumber(evaluatedProjections), orngVisFuncts.createStringFromNumber(attrIndex), strTotalAtts, acc))
                                
                                

                        # select the best new projection and say this is now our new projection to optimize    
                        if len(listOfCandidates) > 0:
                            (accuracy, projection) = maxFunct(listOfCandidates)
                            if hasattr(self, "setStatusBarText"): self.setStatusBarText("Increased accuracy to %2.2f%%" % (accuracy))

        self.finishEvaluation(evaluatedProjections)

    # ##############################################################
    # Loading and saving projection files
    # ##############################################################

    # save the list into a file - filename can be set if you want to call this function without showing the dialog
    def save(self, name, results = None, count = 1000):
        # take care of extension
        if os.path.splitext(name)[1] != ".proj": name = name + ".proj"

        if not results: results = self.results

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        # open, write and save file
        file = open(name, "wt")
        
        attrs = ["kValue", "percentDataUsed", "qualityMeasure", "testingMethod", "parentName", "evaluationAlgorithm", "useExampleWeighting", "useSupervisedPCA", "saveResultsFromFolds", "attrSubsetSelection", "optimizationType", "attributeCount", "attrDisc", "attrCont", "evaluationTime"]
        dict = {}
        for attr in attrs: dict[attr] = self.__dict__[attr]
        dict["dataCheckSum"] = self.data.checksum()
        
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
            s += "), %d, %s, %d, %s)" % (lenTable, str(attrList), tryIndex, generalDict)
            file.write(s + "\n")

            if self.abortCurrentOperation: break
            if hasattr(self, "setStatusBarText"):
                self.setStatusBarText("Saved %s projections" % (orngVisFuncts.createStringFromNumber(i)))

        file.flush()
        file.close()
        return i

    # load projections from a file
    def load(self, name, ignoreCheckSum = 1):
        self.clearResults()
        self.clearArguments()

        if self.__class__.__name__ == "OWVizRank":
            import qt
        
        file = open(name, "rt")
        settings = eval(file.readline()[:-1])
        if settings.get("parentName", "").lower() != self.parentName.lower():
            if self.__class__.__name__ == "OWVizRank":
                qt.QMessageBox.critical( None, "Optimization Dialog", 'Unable to load projection file. It was saved for %s method'%(settings["parentName"]), qt.QMessageBox.Ok)
            else: print 'Unable to load projection file. It was saved for %s method' % (settings["parentName"])
            file.close()
            return [], 0

        if not ignoreCheckSum and settings.has_key("dataCheckSum") and settings["dataCheckSum"] != self.data.checksum():
            if self.__class__.__name__ == "OWVizRank":
                if qt.QMessageBox.information(self, 'VizRank', 'The current data set has a different checksum than the data set that was used to evaluate projections in this file.\nDo you want to continue loading anyway, or cancel?','Continue','Cancel', '', 0,1):
                    file.close()
                    return [], 0
            else: print "'The current data set has a different checksum than the data set that was used to evaluate projections in this file. Continuing loading the file anyway..."
                
        if hasattr(self, "setSettings"): self.setSettings(settings)

        # find if it was computed for specific class values        
        selectedClasses = eval(file.readline()[:-1])
        
        count = 0
        for line in file.xreadlines():
            (acc, other_results, lenTable, attrList, tryIndex, generalDict) = eval(line)
            VizRank.insertItem(self, count, acc, other_results, lenTable, attrList, tryIndex, generalDict)
            count+=1
            if self.abortCurrentOperation: break
            if count % 100 == 0 and hasattr(self, "setStatusBarText"):
                self.setStatusBarText("Loaded %s projections" % (orngVisFuncts.createStringFromNumber(count)))
                qt.qApp.processEvents()        # allow processing of other events
        file.close()

        # update loaded results
        return selectedClasses, count


# ###############################################################################################################################################
# ######       VIZRANK LEARNERS, CLASSIFIERS       ##############################################################################################
# ###############################################################################################################################################

# class that represents kNN classifier that classifies examples based on top evaluated projections
class VizRankClassifier(orange.Classifier):
    def __init__(self, vizrank, data):
        self.VizRank = vizrank

        if self.VizRank.__class__.__name__ == "OWVizRank":
            self.VizRank.parentWidget.cdata(data)
            self.VizRank.useTimeLimit = 1                
        else:
            self.VizRank.setData(data)

        self.VizRank.evaluateProjections()
        
        # do we want to optimize current projection. if yes then spend the same amount of time to optimize it            
        if self.VizRank.optimizeBestProjection:
            self.VizRank.optimizeBestProjections()
            self.VizRank.removeTooSimilarProjections()

        if self.VizRank.__class__.__name__ == "OWVizRank": del self.VizRank.useTimeLimit


    # for a given example run argumentation and find out to which class it most often fall        
    def __call__(self, example, returnType = orange.GetBoth):
        if self.VizRank.__class__.__name__ == "OWVizRank":
            table = orange.ExampleTable(example.domain)
            table.append(example)
            self.VizRank.parentWidget.subsetdata(table)       # show the example is we use the widget
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
    vizrank.useSupervisedPCA = 0
    vizrank.useExampleWeighting = 0
    vizrank.attrSubsetSelection = GAMMA_SINGLE
    vizrank.evaluationTime = 1
    vizrank.evaluateProjections()
    """
    vizrank = VizRank(RADVIZ)
    vizrank.setData(data)
    vizrank.attributeCount = 6
    vizrank.optimizationType = MAXIMUM_NUMBER_OF_ATTRS    # MAXIMUM_NUMBER_OF_ATTRS,  EXACT_NUMBER_OF_ATTRS
    vizrank.attrSubsetSelection = GAMMA_SINGLE
    vizrank.attrCont = CONT_MEAS_S2N
    #vizrank.saveResultsFromFolds = 1
    #vizrank.load(r"E:\Development\Python23\Lib\site-packages\Orange\Datasets\microarray\cancer\leukemia - Radviz - test.proj")
    #vizrank.computeVizRanksAccuracy()
    vizrank.evaluationTime = 0.2
    vizrank.evaluateProjections()
    #vizrank.findArguments(data[0])
    
