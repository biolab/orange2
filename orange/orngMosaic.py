from orngCI import FeatureByCartesianProduct, FeatureByIM
from orngEvalAttr import MeasureAttribute_Distance, MeasureAttribute_MDL, mergeAttrValues
from orngCN2 import CN2UnorderedLearner
from orngContingency import Entropy
from math import sqrt, log, e
from orngScaleData import getVariableValuesSorted, discretizeDomain
import orange, orngVisFuncts
import time, operator, numpy, sys
import orngTest, orngStat, statc
from copy import copy
from orngMisc import LimitedCounter

# quality measures
CHI_SQUARE = 0
CHI_SQUARE_CLASS = 1
CRAMERS_PHI_CLASS = 2
INFORMATION_GAIN = 3
GAIN_RATIO = 4
DISTANCE_MEASURE = 5
MDL = 6
INTERACTION_GAIN = 7
AVERAGE_PROBABILITY_OF_CORRECT_CLASSIFICATION = 8
GINI_INDEX = 9
CN2_RULES = 10

# conditional probability estimation
RELATIVE = 0
LAPLACE = 1
M_ESTIMATE = 2

# optimization type
EXACT_NUMBER_OF_ATTRS = 0
MAXIMUM_NUMBER_OF_ATTRS = 1

# attrCont
MEAS_NONE = 0
MEAS_RELIEFF = 1
MEAS_GAIN_RATIO = 2
MEAS_GINI = 3
MEAS_DISTANCE = 4
MEAS_MDL = 5

# items in the results list
SCORE = 0
ATTR_LIST = 1
TRY_INDEX = 2
EXTRA_INFO = 3

CROSSVALIDATION = 0
PROPORTION_TEST = 1

AO_CROSSVALIDATION = 0
AO_TESTONTRAIN = 1

# classification method
MOS_TOPPROJ = 0
MOS_SEMINAIVE = 1
MOS_COMBINING = 2

discMeasures = [("None", None), ("ReliefF", orange.MeasureAttribute_relief(k=10, m=50)), ("Gain ratio", orange.MeasureAttribute_gainRatio()), ("Gini index", orange.MeasureAttribute_gini()), ("Distance", MeasureAttribute_Distance()), ("Minimum description length", MeasureAttribute_MDL())]

def norm_factor(p):
    max = 10.
    min = -10.
    z = 0.
    eps = 0.001
    while ((max-min)>eps):
        pval = statc.zprob(z)
        if pval>p:
            max = z
        if pval<p:
            min = z
        z = (max + min)/2.
    return z


class orngMosaic:
    def __init__(self):
        self.attributeCount = 2
        self.optimizationType = MAXIMUM_NUMBER_OF_ATTRS
        self.qualityMeasure = MDL
        self.attrDisc = MDL
        self.percentDataUsed = 100

        self.ignoreTooSmallCells = 1    # when computing chi-square and kramer's phi, ignore cells with less than 5 elements

        self.classificationMethod = MOS_SEMINAIVE
        self.testingMethod = CROSSVALIDATION
        self.attributeOrderTestingMethod = AO_TESTONTRAIN
        self.mValue = 2.0
        self.probabilityEstimation = M_ESTIMATE
        self.learnerName = "Mosaic Classifier"
        self.saveSettingsList = ["attrDisc", "qualityMeasure", "percentDataUsed"]       # this is the default list of settings to save when saving interesting projections. change the list to get different behavior

        # classification
        self.clsTau = 0.8               # semi naive bayes parameter
        self.clsTopProjCount = 10       # number of top projections considered in classification
        self.classConfidence = 90       # parameter in the combining way of classification

        self.timeLimit = 0              # if greater than 0 then this is the number of minutes that VizRank will use to evaluate projections
        self.projectionLimit = 0        # if greater than 0 then this is the number of projections that will be evaluated with VizRank
        self.evaluatedProjectionsCount = 0

        self.evaluatedAttributes = None   # save last evaluated attributes
        self.cancelOptimization = 0           # used to stop attribute and value order
        self.cancelEvaluation = 0
        self.cancelTreeBuilding = 0        # used in mosaic tree building
        self.attributeDistributions = {}

        self.data = None
        self.results = []
        self.shownResults = []
        self.attrLenDict = {}

    def clearResults(self):
        self.results = []

    def setData(self, data, removeUnusedValues = 0):
        self.evaluatedAttributes = None
        self.aprioriDistribution = None
        self.aprioriProbabilities = None
        self.attributeNameIndex = {}
        self.attributeDistributions = {}
        self.logits = {}
        self.arguments = {}
        self.classVals = []
        self.cvIndices = None
        self.contingencies = {}
        self.clearResults()

        if data and (len(data) == 0 or len(data.domain) == 0):        # if we don't have any examples or attributes then this is not a valid data set
            data = None

        self.data = discretizeDomain(data, removeUnusedValues)
        if not self.data:
            return

        self.attributeNameIndex = dict([(self.data.domain[i].name, i) for i in range(len(self.data.domain))])

        if self.data.domain.classVar:
            self.classVals = [val for val in self.data.domain.classVar.values]
            self.aprioriDistribution  = orange.Distribution(self.data.domain.classVar.name, self.data)
            self.aprioriProbabilities = [nrOfCases / float(max(1, len(self.data))) for nrOfCases in self.aprioriDistribution]

            s = sum(self.aprioriDistribution)
            for i in range(len(self.aprioriDistribution)):
                if s == 0:
                    p = 1
                elif (self.aprioriDistribution[i]/s) > 0 and (self.aprioriDistribution[i]/s) < 1:
                    p = (self.aprioriDistribution[i]/s) / (1-(self.aprioriDistribution[i]/s))
                elif (self.aprioriDistribution[i]/s) == 0:
                    p = 0.0001
                elif (self.aprioriDistribution[i]/s) == 1:
                    p = 99999.99
                self.logits[self.classVals[i]] = log(p)


    # given a dataset return a list of attributes where attribute are sorted by their decreasing importance for class discrimination
    def getEvaluatedAttributes(self, data):
        if not data.domain.classVar or data.domain.classVar.varType != orange.VarTypes.Discrete:
            if self.__class__.__name__ != "orngMosaic":
                from PyQt4.QtGui import QMessageBox
                QMessageBox.information( None, "Mosaic Dialog", 'In order to be able to find interesing projections the data set has to have a discrete class.', QMessageBox.Ok + QMessageBox.Default)
            return []
        if self.evaluatedAttributes:
            return self.evaluatedAttributes

        if self.__class__.__name__ != "orngMosaic":
            from PyQt4.QtGui import qApp, QWidget
            from PyQt4.QtCore import Qt
            self.setStatusBarText("Evaluating attributes...")
            qApp.setOverrideCursor(Qt.WaitCursor)

        try:
            # evaluate attributes using the selected attribute measure
            self.evaluatedAttributes = orngVisFuncts.evaluateAttributesDiscClass(data, None, discMeasures[self.attrDisc][1])
            # remove attributes with only one value
            for attr in self.evaluatedAttributes[::-1]:
                if data.domain[attr].varType == orange.VarTypes.Discrete and len(data.domain[attr].values) < 2:
                    self.evaluatedAttributes.remove(attr)
        except:
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # print the exception

        if self.__class__.__name__ != "orngMosaic":
            self.setStatusBarText("")
            qApp.restoreOverrideCursor()

        if not self.evaluatedAttributes: return []
        else:                            return self.evaluatedAttributes


    # create a dictionary with all possible pairs of "combination-of-attr-values" : count
    def getConditionalDistributions(self, data, attrs):
        dict = {}
        for i in range(1, len(attrs)+1):
            newFeature, quality = FeatureByCartesianProduct(data, attrs[:i])
            dist = orange.Distribution(newFeature, data)
            for val in newFeature.values:
                dict[val] = dist[val]

        if data.domain.classVar:
            cont = orange.ContingencyAttrClass(newFeature, data)
            clsValues = data.domain.classVar.values
            for val in newFeature.values:
                for classV in clsValues:
                    dict[val+"-"+classV] = cont[val][classV]
        return dict

    def getContingencys(self, attrs):
        conts = {}
        for attr in attrs:
            conts[attr] = self.contingencies.get(attr, orange.ContingencyAttrClass(attr, self.data))

        self.contingencies.update(conts)
        return conts


    def isEvaluationCanceled(self):
        if self.cancelEvaluation:  return 1

        stop = 0
        if self.timeLimit > 0: stop = (time.time() - self.startTime) / 60 >= self.timeLimit
        if self.projectionLimit > 0: stop = stop or self.evaluatedProjectionsCount >= self.projectionLimit
        return stop

    def isOptimizationCanceled(self):
        if self.cancelOptimization:  return 1

        stop = 0
        if self.timeLimit > 0: stop = (time.time() - self.startTime) / 60 >= self.timeLimit
        if self.projectionLimit > 0: stop = stop or self.evaluatedProjectionsCount >= self.projectionLimit
        return stop


    #
    # PROJECTIONS EVALUATION
    #
    def evaluateProjections(self):
        self.cancelEvaluation = 0
        self.evaluatedProjectionsCount = 0
        if not self.data or len(self.data) == 0 or (not self.classVals and self.qualityMeasure != CHI_SQUARE): return
        fullData = self.data

        try:
            if self.percentDataUsed != 100:
                self.data = fullData.select(orange.MakeRandomIndices2(fullData, 1.0-float(self.percentDataUsed)/100.0))

            self.clearResults()

            if len(self.data) == 0:
                self.data = fullData
                self.setStatusBarText("The dataset is empty. Unable to evaluate projections...")
                self.finishEvaluation(0)
                return

            if self.__class__.__name__ != "orngMosaic":
                self.disableControls()
                self.visualizationWidget.progressBarInit()
                from PyQt4.QtGui import qApp
                self.qApp = qApp

            self.startTime = time.time()
            triedPossibilities = 0

            maxLength = self.attributeCount
            if self.optimizationType == 0: minLength = self.attributeCount
            else:                          minLength = 1

            # generate cn2 rules and show projections that have
            if self.qualityMeasure == CN2_RULES:
                ruleFinder = orange.RuleBeamFinder()
                ruleFinder.evaluator = orange.RuleEvaluator_Laplace()
                ruleFinder.ruleStoppingValidator = orange.RuleValidator_LRS(alpha=0.2, min_coverage=0, max_rule_complexity = 4)
                ruleFinder.validator = orange.RuleValidator_LRS(alpha=0.05, min_coverage=0, max_rule_complexity=4)
                ruleFinder.ruleFilter = orange.RuleBeamFilter_Width(width=5)

                learner = CN2UnorderedLearner()
                learner.ruleFinder = ruleFinder
                learner.coverAndRemove = orange.RuleCovererAndRemover_Default()

                if self.__class__.__name__ != "orngMosaic":
                    from OWCN2 import CN2ProgressBar
                    learner.progressCallback = CN2ProgressBar(self.visualizationWidget)

                classifier = learner(self.data)

                self.dictResults = {}
                for rule in classifier.rules:
                    conds = rule.filter.conditions
                    domain = rule.filter.domain
                    attrs = [domain[c.position].name for c in conds]
                    if len(attrs) > self.attributeCount or (self.optimizationType == EXACT_NUMBER_OF_ATTRS and len(attrs) != self.attributeCount):
                        continue
                    sortedAttrs = copy(attrs); sortedAttrs.sort()
                    vals = [domain[c.position].values[int(c.values[0])] for c in conds]
                    self.dictResults[tuple(sortedAttrs)] = self.dictResults.get(tuple(sortedAttrs), []) + [(rule.quality, attrs, vals)]

                for key in self.dictResults.keys():
                    el = self.dictResults[key]
                    score = sum([e[0] for e in el]) / float(len(el))
                    self.insertItem(score, el[0][1], self.findTargetIndex(score), 0, extraInfo = el)

            else:
                if self.qualityMeasure == CHI_SQUARE:
                    evaluatedAttrs = [attr.name for attr in self.data.domain.variables]
                else:
                    evaluatedAttrs = self.getEvaluatedAttributes(self.data)
                    
                if evaluatedAttrs == []:
                    self.data = fullData
                    self.finishEvaluation(0)
                    return

                # total number of possible projections
                totalPossibilities = 0
                for i in range(minLength, maxLength+1):
                    totalPossibilities += orngVisFuncts.combinationsCount(i, len(evaluatedAttrs))
                totalStr = orngVisFuncts.createStringFromNumber(totalPossibilities)

                self.cvIndices = None
                for z in range(len(evaluatedAttrs)):
                    for u in range(minLength-1, maxLength):
                        combinations = orngVisFuncts.combinations(evaluatedAttrs[:z], u)

                        for attrList in combinations:
                            triedPossibilities += 1

                            attrs = [evaluatedAttrs[z]] + attrList
                            diffVals = reduce(operator.mul, [max(1, len(self.data.domain[attr].values)) for attr in attrs])
                            if diffVals > 200: continue     # we cannot efficiently deal with projections with more than 200 different values

                            val = self._Evaluate(attrs)
                            ind = self.findTargetIndex(val)
                            start = ind
                            if ind > 0 and self.results[ind-1][0] == val:
                                ind -= 1
                            while ind > 0 and self.results[ind-1][0] == val and len(attrs) < len(self.results[ind-1][1]):
                                ind -= 1
                            while ind < len(self.results) and self.results[ind][0] == val and len(attrs) > len(self.results[ind][1]):
                                ind += 1

                            self.insertItem(val, attrs, ind, triedPossibilities)
                            self.evaluatedProjectionsCount += 1

                            if self.__class__.__name__ != "orngMosaic":
                                self.visualizationWidget.progressBarSet(100.0*triedPossibilities/float(totalPossibilities))
                                self.setStatusBarText("Evaluated %s/%s visualizations..." % (orngVisFuncts.createStringFromNumber(triedPossibilities), totalStr))
                            if hasattr(self, "qApp"):
                                self.qApp.processEvents()        # allow processing of other events

                            if self.isEvaluationCanceled():
                                self.data = fullData
                                self.finishEvaluation(triedPossibilities)
                                return
        except:
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # print the exception

        self.data = fullData
        self.finishEvaluation(triedPossibilities)


    def finishEvaluation(self, evaluatedProjections):
        self.attrLenDict = dict([(i,1) for i in range(self.attributeCount+1)])
        
        if self.__class__.__name__ != "orngMosaic":
            secs = time.time() - self.startTime
            self.setStatusBarText("Evaluation stopped (evaluated %s projections in %d min, %d sec)" % (orngVisFuncts.createStringFromNumber(evaluatedProjections), secs/60, secs%60))
            self.visualizationWidget.progressBarFinished()
            self.enableControls()
            self.finishedAddingResults()


    def _Evaluate(self, attrs):
        newFeature, quality = FeatureByCartesianProduct(self.data, attrs)

        retVal = -1
        if self.qualityMeasure in [CHI_SQUARE]:
            dists = []
            for attr in attrs:
                if not self.attributeDistributions.has_key(attr):
                    self.attributeDistributions[attr] = orange.Distribution(attr, self.data)
                dists.append(self.attributeDistributions[attr]) 
                        
            # create cartesian product of selected attributes and compute contingency 
            dXY = orange.Distribution(newFeature, self.data)   # distribution of the merged attribute
            
            # compute chi-square
            retVal = 0.0
            domain = self.data.domain
            for vs in LimitedCounter([len(domain[attr].values) for attr in attrs]):
                expected = len(self.data) * reduce(lambda x, y: x*y, [dists[i][v]/float(len(self.data)) for i,v in enumerate(vs)])
                actual = dXY["-".join([domain[attrs[i]].values[v] for i, v in enumerate(vs)])]
                if expected == 0: continue
                pearson2 = (actual - expected)*(actual - expected) / expected
                retVal += pearson2
                
            
                
        if self.qualityMeasure in [CHI_SQUARE_CLASS, CRAMERS_PHI_CLASS]:
            aprioriSum = sum(self.aprioriDistribution)
            retVal = 0.0

            for dist in orange.ContingencyAttrClass(newFeature, self.data):
                for i in range(len(self.aprioriDistribution)):
                    expected = sum(dist) * (self.aprioriDistribution[i] / float(aprioriSum))
                    if self.ignoreTooSmallCells and expected < 5: continue       # statisticians advise ignoring terms that have less than 5 expected examples, since they can significantly disturb chi-square
                    if not expected: continue                                    # prevent division by zero
                    retVal += (dist[i] - expected)**2 / expected

            if self.qualityMeasure == CRAMERS_PHI_CLASS:
                vals = min(len(newFeature.values), len(self.data.domain.classVar.values))-1
                if vals:
                    retVal = sqrt(retVal / (len(self.data) * vals))

        elif self.qualityMeasure == DISTANCE_MEASURE:
            retVal = MeasureAttribute_Distance(newFeature, self.data)

        elif self.qualityMeasure == MDL:
            retVal = MeasureAttribute_MDL(newFeature, self.data)

        elif self.qualityMeasure == GINI_INDEX:
            retVal = orange.MeasureAttribute_gini(newFeature, self.data)

        elif self.qualityMeasure == INFORMATION_GAIN:
            retVal = orange.MeasureAttribute_info(newFeature, self.data)
            classEntropy = Entropy(numpy.array([val for val in self.aprioriDistribution]))
            if classEntropy:
                retVal = retVal * 100.0 / classEntropy
        
        elif self.qualityMeasure == GAIN_RATIO:
            retVal = orange.MeasureAttribute_gainRatio(newFeature, self.data)

        elif self.qualityMeasure == INTERACTION_GAIN:
            new = orange.MeasureAttribute_info(newFeature, self.data)
            gains = [orange.MeasureAttribute_info(attr, self.data) for attr in attrs]
            retVal = new - sum(gains)

            classEntropy = Entropy(numpy.array([val for val in self.aprioriDistribution]))
            if classEntropy:
                retVal = retVal * 100.0 / classEntropy

        elif self.qualityMeasure == AVERAGE_PROBABILITY_OF_CORRECT_CLASSIFICATION:
            d = self.data.select([newFeature, self.data.domain.classVar])     # create a dataset that has only this new feature and class info

            if not self.cvIndices:
                if self.testingMethod == PROPORTION_TEST:
                    pick = orange.MakeRandomIndices2(stratified = orange.MakeRandomIndices.StratifiedIfPossible, p0 = 0.7, randomGenerator = 0)
                    self.cvIndices = [pick(d) for i in range(10)]
                elif self.testingMethod == CROSSVALIDATION:
                    ind = orange.MakeRandomIndicesCV(d, 10, randomGenerator = 0, stratified = orange.MakeRandomIndices.StratifiedIfPossible)
                    self.cvIndices = [[val == i for val in ind] for i in range(10)]

            acc = 0.0; count = 0
            for ind in self.cvIndices:
                learnset = d.selectref(ind, 0)
                testset = d.selectref(ind, 1)
                learnDist = orange.Distribution(d.domain.classVar, learnset)
                newFeatureDist = orange.Distribution(newFeature, testset)
                learnConts = orange.ContingencyAttrClass(newFeature, learnset)
                testConts  = orange.ContingencyAttrClass(newFeature, testset)
                for val in testConts.keys():
                    s = sum(learnConts[val])
                    if not s: continue
                    learnClassProb = [v/float(s) for v in learnConts[val]]      # class distribution for each class value (on learning set)
                    testClassDist = [v for v in testConts[val]]                 # number of examples for each class value (on testing set)
                    for i in range(len(testClassDist)):
                        acc   += learnClassProb[i] * testClassDist[i]
                        count += testClassDist[i]
            retVal = 100*acc / max(1, float(count))

        del newFeature, quality
        return retVal

    #
    # ARGUMENTATION FUNCTIONS
    #
    def findArguments(self, example = None):
        self.arguments = dict([(val, []) for val in self.classVals])

        if not example or len(self.shownResults) == 0: return None, None
        if not (self.data and self.data.domain.classVar and self.logits and self.classVals): return None, None

        if self.__class__.__name__ != "orngMosaic":
            from PyQt4.QtGui import qApp

        usedArguments = 0
        for index in range(len(self.shownResults)):
            (score, attrList, tryIndex, extraInfo) = self.shownResults[index]
            args = self.evaluateArgument(example, attrList, score)      # call evaluation of a specific projection
            if not args: continue

            if self.classificationMethod == MOS_TOPPROJ and usedArguments >= self.clsTopProjCount:
                break

            for val in self.classVals:
                pos = self.getArgumentIndex(args[val][0], val)
                self.arguments[val].insert(pos, (args[val][0], score, attrList, index, args[val][1]))
                if self.__class__.__name__ != "orngMosaic" and val == str(self.classValueList.currentText()):
                    self.insertArgument(args[val][0], args[val][1], attrList, pos)
                    qApp.processEvents()
            usedArguments += 1

        predictions = []
        if self.classificationMethod == MOS_TOPPROJ:
            for val in self.classVals:
                predictions.append(sum([v[0] for v in self.arguments[val]]))

        elif self.classificationMethod == MOS_SEMINAIVE:
            self.argumentation_SemiNaive()      # remove combinations of attributes that are not dependent
            for val in self.classVals:
                v = self.aprioriDistribution[val]/float(len(self.data))
                for arg in self.arguments[val]:
                    v *= arg[0]
                predictions.append(v)

        elif self.classificationMethod == MOS_COMBINING:
            self.argumentation_Combining()
            for val in self.classVals:
                value = self.logits[val] + sum([v[0] for v in self.arguments[val]])
                predictions.append(e**value / (1 + e**value))       # use predictions from all arguments to classify an example

        classValue = self.data.domain.classVar[predictions.index(max(predictions))]
        if sum(predictions) == 0:
            predictions = [1]*len(predictions)
        dist = orange.DiscDistribution([val/float(sum(predictions)) for val in predictions])
        dist.variable = self.data.domain.classVar
        return classValue, dist


    # for a given example and a projection evaluate arguments for each class value
    def evaluateArgument(self, example, attrList, score):
        attrVals = [example[attr] for attr in attrList]
        if "?" in attrVals: return None      # the testExample has a missing value at one of the visualized attributes

        #subData = orange.Preprocessor_take(self.data, values = dict([(self.data.domain[attr], example[attr]) for attr in attrList]))
        attrVals = [example[attr] for attr in attrList]
        if "?" in attrVals: return None      # the testExample has a missing value at one of the visualized attributes
        subData = self.getDataSubset(attrList, attrVals)
        if not subData or len(subData) == 0: return None

        lenData = len(self.data)
        lenSubData = len(subData)

        arguments = {}

        actualProbabilities = self.estimateClassProbabilities(self.data, example, attrList, subData)    # estimate probabilities for the example and given attribute list

        if self.classificationMethod in [MOS_TOPPROJ, MOS_SEMINAIVE]:
            d = {}
            eps = 0.0
            for i in range(len(self.aprioriDistribution)):
                conts = self.getContingencys(attrList)
                P_Ci = self.aprioriDistribution[i]/float(lenData)
                prob = 1.0
                for attr in attrList:
                    prob *= self.contingencies[attr][example[attr]][i] / float(max(1,sum(self.contingencies[attr][example[attr]].values())))
                eps += P_Ci * abs( actualProbabilities[i]/float(lenSubData) - (prob / P_Ci) )

            for i in range(len(self.aprioriDistribution)):
                arguments[self.classVals[i]] = (actualProbabilities[i], (eps, lenSubData))

        elif self.classificationMethod == MOS_COMBINING:
            for i in range(len(self.aprioriDistribution)):
                # compute log odds of P(c|ai)/P(c)
                val = 0
                if (1-actualProbabilities[i]) * self.aprioriProbabilities[i] > 0.0:
                    val = (actualProbabilities[i] * (1-self.aprioriProbabilities[i])) / ((1-actualProbabilities[i]) * self.aprioriProbabilities[i])
                    if val > 0: val = log(val)
                    else: val = -999.99
                else: val = 999.99

                # compute confidence interval
                approxZero = 0.0001
                aprioriPc0 = max(approxZero, self.aprioriProbabilities[i]);
                aprioriPc1 = max(approxZero, 1-self.aprioriProbabilities[i])
                actualPc0  = max(approxZero, actualProbabilities[i]);
                actualPc1  = max(approxZero, 1-actualProbabilities[i])
                part1 = 1 / (len(self.data) * aprioriPc0 * aprioriPc1)
                part2 = 1 / max(approxZero, lenSubData  * actualPc0  * actualPc1)
                #error = sqrt(max(0, part2 - part1)) * 1.64
                if self.classConfidence == 0: error = 0.
                else:                         error = sqrt(max(0, part2 - part1)) * norm_factor(1-((1-float(self.classConfidence)/100.)/2.))

                arguments[self.classVals[i]] = (val, error)

        return arguments


    # probability estimation function
    def estimateClassProbabilities(self, data, example, attrList, subData = None, subDataDistribution = None, aprioriDistribution = None, probabilityEstimation = -1, mValue = -1):
        if probabilityEstimation == -1: probabilityEstimation = self.probabilityEstimation
        if aprioriDistribution == None: aprioriDistribution = self.aprioriDistribution
        if mValue == -1:                mValue = self.mValue

        if not subData:
            attrVals = [example[attr] for attr in attrList]
            if "?" in attrVals: return None      # the testExample has a missing value at one of the visualized attributes
            subData = self.getDataSubset(attrList, attrVals)
            #subData = orange.Preprocessor_take(data, values = dict([(data.domain[attr], example[attr]) for attr in attrList]))
        if not subDataDistribution:
            subDataDistribution = orange.Distribution(data.domain.classVar.name, subData)

        lenData = len(data)
        lenSubData = len(subData)

        # estimate probabilities
        if probabilityEstimation == RELATIVE:
            if not len(subData):
                self.printVerbose("OWMosaicOptimization: Empty data subset. Unable to compute relative frequency.")
                return [0.0 for i in range(len(aprioriDistribution))]      # prevent division by zero
            actualProbabilities = [ subDataDistribution[index] / float(lenSubData) for index in range(len(aprioriDistribution))]      # P(c_i | a_k) / P(c_i)

        elif probabilityEstimation == LAPLACE:
            actualProbabilities = []
            for index in range(len(aprioriDistribution)):
                actualProbabilities.append((subDataDistribution[index]+1) / float(lenSubData+len(subDataDistribution)))      # (r+1 / n+c) / P(c_i)

        elif probabilityEstimation == M_ESTIMATE:
            actualProbabilities = []
            for index in range(len(aprioriDistribution)):
                n = subDataDistribution[index]
                pa = aprioriDistribution[index]/float(sum(aprioriDistribution))
                actualProbabilities.append((pa * mValue + n) / float(lenSubData + mValue))       # p = (pa*m+n)/(N+m)

        # just to check if the math is ok
        s = sum(actualProbabilities)
        if "%.3f" % s != "1.000":
            print "Strange, huh. Probabilities don't sum to 1, but to", s, actualProbabilities

        return actualProbabilities


    def argumentation_Combining(self):
        if not self.arguments or not self.arguments.values(): return

        for classValue in self.arguments.keys():
            # create a dict for arguments of different lengths. each dict has arguments of this length and corresponding scores
            argList = {1: {}, 2: {}, 3:{}, 4:{}}
            arguments = self.arguments[classValue]
            for i in range(len(arguments)):
                attrs = arguments[i][2]
                attrs.sort()
                argList[len(attrs)][tuple(attrs)] = (arguments[i][0], arguments[i][4], i)       # save arg val, error and arg index

            if len(argList[1]) == 0: return     # in case that we only evaluated projections with EXACTLY X attributes we cannot remove weak arguments

            for count in [4,3,2]:
                args = argList[count]

                candidates = []     # candidate projections for deleting
                for key in args.keys():
                    splits = orngVisFuncts.getPossibleSplits(list(key))
                    for split in splits:
                        vals = [argList[len(v)].get(tuple(v), [None, None, None])[0] for v in split]
                        errors = [argList[len(v)].get(tuple(v), [None, None, None])[1] for v in split]
                        # if None in vals or abs(sum(vals)) >= abs(args[key][0]):       # this condition is not enough, because some elements in vals have + and some - signs which makes a combination better than individual attributes
                        if None in vals:        # some attrs have already been used
                            args.pop(key)
                            break
                        sameSign = (sum([abs(val) == val for val in vals]) in [0, len(vals)])   # do all values in vals have the same sign
                        partialVals = sum([abs(val) for val in vals] + errors)        # vals may have all - or all + values. errors are all +. to sum, we must make all vals +.
                        complexVal = abs(args[key][0]) - args[key][1]       # value and error of the combination of attributes
                        if not sameSign or partialVals >= complexVal:
                            args.pop(key)   # remove the combination of attributes because a split exists, that produces a more important argument
                            break
                    if args.has_key(key):
                        candidates.append((args[key][0], key, splits))

                candidates.sort()
                candidates.reverse()
                for val, attrs, splits in candidates:
                    vals = []
                    for split in splits:
                        vals += [argList[len(v)].get(tuple(v)) for v in split]
                    if None in vals:
                        args.pop(attrs)
                        continue       # we have already used some of the attributes in split for a better projection

                    # obviously we have found a very good projection of attributes and we still have all the attributes needed
                    # we now have to remove other projections that use these attributes
                    for split in splits:
                        for part in split:
                            if argList[len(part)].has_key(tuple(part)):
                                argList[len(part)].pop(tuple(part))

            #print [len(argList[1]), len(argList[2]), len(argList[3])]
            #assert False not in [a[a.keys()[i]] == a.values()[i] for i in range(len(a.keys()))]
            indicesToKeep = [val[2] for val in argList[1].values() + argList[2].values() + argList[3].values() + argList[4].values()]

            # we remove all arguments that are not in indicesToKeep
            for i in range(len(arguments))[::-1]:       # we remove all arguments that are not in indicesToKeep
                if i not in indicesToKeep:
                    arguments.pop(i)


    # compute probability that the combination of attribute values in attrValues is significantly different from being independent
    # see Kononenko: Semi-naive Bayes
    def argumentation_SemiNaive(self):
        if not self.arguments or not self.arguments.values(): return

        for classValue in self.arguments.keys():
            # create a dict for arguments of different lengths. each dict has arguments of this length and corresponding scores
            argList = {1: {}, 2: {}, 3:{}, 4:{}}
            arguments = self.arguments[classValue]
            for i in range(len(arguments)):
                attrs = arguments[i][2]
                argList[len(attrs)][tuple(attrs)] = (arguments[i][4], i)

            if len(argList[1]) == 0: return     # in case that we only evaluated projections with EXACTLY X attributes we cannot remove weak arguments

            for count in [4,3,2]:
                for key in argList[count].keys():
                    eps, Nab = argList[count][key][0]
                    v = float(4*eps*eps*Nab)
                    if v == 0 or 1/v > 1 - self.clsTau:
                        argList[count].pop(key)

            indicesToKeep = [val[1] for val in argList[1].values() + argList[2].values() + argList[3].values() + argList[4].values()]

            for i in range(len(arguments))[::-1]:       # we remove all arguments that are not in indicesToKeep
                if i not in indicesToKeep:
                    arguments.pop(i)



    def getInteractionImportanceProbability(self, attrList, attrValues, subData = None, contingencies = None, aprioriDistribution = None, subDataDistribution = None):
        assert len(attrList) == len(attrValues)

        if not subData:
            #subData = orange.Preprocessor_take(self.data, values = dict([(self.data.domain[attrList[i]], attrValues[i]) for i in range(len(attrList))]))
            subData = self.getDataSubset(attrList, attrValues)
        if not aprioriDistribution: aprioriDistribution = self.aprioriDistribution
        if not subDataDistribution:
            subDataDistribution = orange.Distribution(self.data.domain.classVar.name, subData)
        """
        if not contingencies:
            for i in range(len(attrList)):
                temp = self.data.filter({attrList[i]: attrValues[i]})
                contingencies
            contingencies = [orange.ContingencyAttrClass(attr, self.data) for attr in attrList]
        """

        lenSubData = len(subData)
        lenData = len(self.data)
        eps = 0.0
        for i in range(len(aprioriDistribution)):
            product = 1.0
            for j in range(len(attrList)):
                N_ci_aj = self.classDistributionsForExample[attrList[j]][i]
                N_aj = sum(self.classDistributionsForExample[attrList[j]])
                product *= N_ci_aj / N_aj       # P(ci|aj)

            eps += (aprioriDistribution[i]/lenData) * abs((subDataDistribution[i]/lenSubData) - (product*lenData)/aprioriDistribution[i])
            #eps += abs((subDataDistribution[i]/lenSubData) - (product*lenData)/aprioriDistribution[i])

        if eps*lenSubData == 0: return 9999999
        ret = 1/(4*eps*eps*lenSubData)
        #if ret > 1 or ret < 0.0:
        #    print "invalid probability", eps, ret
        #assert ret >= 0.0
        #assert ret <= 1.0
        return ret

    def getArgumentIndex(self, value, classValue):
        if len(self.arguments[classValue]) == 0: return 0

        top = 0; bottom = len(self.arguments[classValue])
        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if max(value, self.arguments[classValue][mid][0]) == value: bottom = mid
            else: top = mid

        if max(value, self.arguments[classValue][top][0]) == value:  return top
        else:                                                        return bottom


    #######
    # code for evaluating different placements of a set of attributes by the separation of different classes in the graph
    #######
    def evaluateAttributeOrder(self, attrs, valueOrder, conditions, revert, domain = None):
        if not domain:
            domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.data.domain.classVar])
            self.weightID = orange.newmetaid()
            domain.addmeta(self.weightID, orange.FloatVariable("ExampleWeight"))
        projData = orange.ExampleTable(domain)
        projData.domain.classVar.distributed = True

        triedIndices = [0]*(len(attrs))
        maxVals = [len(val) for val in valueOrder]
        xpos = 0; ypos = 0
        while triedIndices[0] < maxVals[0]:
            vals = [valueOrder[i][triedIndices[i]] for i in range(len(attrs))]
            combVal = reduce(operator.concat, map(operator.concat, [vals[i] for i in revert], ["-"]*len(vals)))[:-1]
            if conditions[combVal][0] > 0:
                #projData.append([xpos, ypos, conditions[combVal][1]])
                projData.append([xpos, ypos, projData.domain.classVar.values[0]])
                val = orange.Value(projData.domain.classVar, conditions[combVal][1])
                val.svalue = conditions[combVal][2]
                projData[-1][-1] = val
                projData[-1].setmeta("ExampleWeight", conditions[combVal][0]/max(1,len(self.data))) # set weight of the rectangle

            triedIndices[-1] += 1
            for i in range(1, len(attrs))[::-1]:
                if triedIndices[i] >= maxVals[i]:
                    triedIndices[i-1] += 1
                    triedIndices[i] = 0

            xpos = triedIndices[-1]
            ypos = len(attrs) > 1 and 2 * triedIndices[-2]
            if len(attrs) > 2: xpos += (2 + maxVals[-1]) * triedIndices[-3] # add the space of 3 between each different value of third attribute
            if len(attrs) > 3: ypos += (4 + maxVals[-2]) * triedIndices[-4] # add the space of 4 between each different value of fourth attribute

        distance = orange.ExamplesDistanceConstructor_Manhattan()
        distance.normalize = 0
        learner = orange.kNNLearner(rankWeight = 0, k = len(projData)/2, distanceConstructor = distance)
        if self.attributeOrderTestingMethod == AO_CROSSVALIDATION:
            results = orngTest.leaveOneOut([learner], (projData, self.weightID))
        else:
            results = orngTest.learnAndTestOnLearnData([learner], (projData, self.weightID))
        return orngStat.AP(results)[0]

    # for a given subset of attributes (max 4) find which permutation is most visual friendly. optimizeValueOrder
    def findOptimalAttributeOrder(self, attrs, optimizeValueOrder = 0):
        if not self.data or not self.data.domain.classVar or self.data.domain.classVar.varType != orange.VarTypes.Discrete:
            return None
        apriori = [max(1, self.aprioriDistribution[val]) for val in self.data.domain.classVar.values]
        conditions = {}
        newFeature, quality = FeatureByCartesianProduct(self.data, attrs)
        dist = orange.Distribution(newFeature, self.data)
        cont = orange.ContingencyAttrClass(newFeature, self.data)
        for key in cont.keys():
            if dist[key] == 0: conditions[key] = (0, 0)
            else:
                quotients = map(operator.div, cont[key], apriori)
                conditions[key] = (dist[key], self.data.domain.classVar.values[quotients.index(max(quotients))], cont[key])

        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.data.domain.classVar])
        self.weightID = orange.newmetaid()
        domain.addmeta(self.weightID, orange.FloatVariable("ExampleWeight"))

        # create permutations of attributes and attribute values
        attrPerms = orngVisFuncts.permutations(range(len(attrs)))
        valuePerms = {}     # for each attribute we generate permutations of its values (if optimizeValueOrder = 1)
        for attr in attrs:
            if optimizeValueOrder:  valuePerms[attr] = orngVisFuncts.permutations(getVariableValuesSorted(self.data.domain[attr]))
            else:                   valuePerms[attr] = [getVariableValuesSorted(self.data.domain[attr])]

        if self.__class__.__name__ != "orngMosaic":
            self.setStatusBarText("Generating possible attribute orders...")
            self.visualizationWidget.progressBarInit()

        possibleOrders = []
        triedIndices = [0]*(len(attrs))                 # list of indices that point to the next permutation of values that will be tried
        maxVals = [len(valuePerms[attr]) for attr in attrs]
        while triedIndices[-1] < maxVals[-1]:           # we have no more possible placements if we have an overflow
            valueOrder = [valuePerms[attrs[i]][triedIndices[i]] for i in range(len(attrs))]
            possibleOrders.append(valueOrder)           # list of orders that we will evaluate
            triedIndices[0] += 1
            if self.cancelOptimization: break
            for i in range(len(attrs)-1):
                if triedIndices[i] >= maxVals[i]:
                    triedIndices[i+1] += 1
                    triedIndices[i] = 0

        bestPlacements = []
        total = len(attrPerms) * len(possibleOrders)
        strCount = orngVisFuncts.createStringFromNumber(total)
        self.evaluatedProjectionsCount = 0
        for attrPerm in attrPerms:                      # for all attribute permutations
            currAttrs = [attrs[i] for i in attrPerm]
            if self.cancelOptimization: break
            tempPerms = []
            for order in possibleOrders:                # for all permutations of attribute values
                currValueOrder = [order[i] for i in attrPerm]
                val = self.evaluateAttributeOrder(currAttrs, currValueOrder, conditions, map(attrPerm.index, range(len(attrPerm))), domain)
                tempPerms.append((val*100, currAttrs, currValueOrder))
                self.evaluatedProjectionsCount += 1
                if self.evaluatedProjectionsCount % 10 == 0 and self.__class__.__name__ != "orngMosaic":
                    self.setStatusBarText("Evaluated %s/%s attribute orders..." % (orngVisFuncts.createStringFromNumber(self.evaluatedProjectionsCount), strCount))
                    self.visualizationWidget.progressBarSet(100*self.evaluatedProjectionsCount/float(total))
                    if self.isOptimizationCanceled(): break
            bestPlacements.append(max(tempPerms))

        if self.__class__.__name__ != "orngMosaic":
            self.setStatusBarText("")
            self.visualizationWidget.progressBarFinished()

        bestPlacements.sort()
        bestPlacements.reverse()
        return bestPlacements

    #
    # MISC FUNCTIONS
    #

    # use bisection to find correct index
    def findTargetIndex(self, score):
        top = 0; bottom = len(self.results)

        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if max(score, self.results[mid][SCORE]) == score: bottom = mid
            else: top = mid

        if len(self.results) == 0: return 0
        if max(score, self.results[top][SCORE]) == score:
            return top
        else:
            return bottom

    # insert new result - give parameters: score of projection, number of examples in projection and list of attributes.
    def insertItem(self, score, attrList, index, tryIndex, extraInfo = []):
        self.results.insert(index, (score, attrList, tryIndex, extraInfo))
        self.shownResults.insert(index, (score, attrList, tryIndex, extraInfo))

   
    # from a list of attributes build a nice string with attribute names
    def buildAttrString(self, attrList):
        if len(attrList) == 0: return ""

        strList = attrList[0]
        for attr in attrList[1:]:
            strList += ", " + attr
        return strList


    # select only those examples that have the specified attribute values
    # also remove examples that have missing values at those attributes
    def getDataSubset(self, attrList, attrValues):
        temp = self.data.selectref(dict([(attrList[i], attrValues[i]) for i in range(len(attrList))]))
        filter = orange.Filter_isDefined(domain=self.data.domain)
        for v in self.data.domain.variables:
            filter.check[v] = v.name in attrList
        return filter(temp)


    #
    # LOADING, SAVING, .....
    #
    # save evaluated projections into a file
    def save(self, filename):
        dict = {}
        for attr in self.saveSettingsList:
            dict[attr] = self.__dict__[attr]
        dict["dataCheckSum"] = self.data.checksum()

        file = open(filename, "wt")
        file.write("%s\n" % (str(dict)))
        for (score, attrList, tryIndex, extraInfo) in self.shownResults:
            file.write("(%.4f, %s, %d)\n" % (score, attrList, tryIndex))
        file.flush()
        file.close()

    def load(self, filename, ignoreCheckSum = 0):
        self.clearResults()
        file = open(filename, "rt")
        settings = eval(file.readline()[:-1])

        if not ignoreCheckSum and settings.has_key("dataCheckSum") and settings["dataCheckSum"] != self.data.checksum():
            if self.__class__.__name__ != "orngMosaic":
                from PyQt4.QtGui import QMessageBox
                if QMessageBox.information(self, 'VizRank', 'The current data set has a different checksum than the data set that was used to evaluate visualizations in this file.\nDo you want to continue loading anyway, or cancel?','Continue','Cancel', '', 0,1):
                    file.close()
                    return
            else:
                print "dataset checksum does not agree with the checksum in the projections file. loading anyway"

        #self.setSettings(settings)
        for key in settings.keys():
            setattr(self, key, settings[key])

        ind = 0
        for line in file.xreadlines():
            (score, attrList, tryIndex) = eval(line)
            self.insertItem(score, attrList, ind, tryIndex)
            ind+=1
        file.close()

        return ind


# #############################################################################
# definition of tree of mosaics
class MosaicTreeNode:
    def __init__(self, parent, attrs):
        self.children = {}
        self.parent = parent
        self.attrs = attrs
        self.branches = {}        # links to other MosaicTreeNode instances that represent branches. Keys are attribute values, e.g. ([0,1,3], [1,2,2])
        self.branchSelector = None
        self.selectionIndices = None


# #############################################################################
# learner that builds MosaicTreeLearner
class MosaicTreeLearner(orange.Learner):
    def __init__(self, mosaic = None, statusFunct = None):
        if not mosaic:
            mosaic = orngMosaic()
        self.mosaic = mosaic
        self.statusFunct = statusFunct
        #self.mosaic.qualityMeasure = MDL        # always use MDL for estimating quality of projections - this also stops building tree if no combination of attributes produces an improvement
        self.name = self.mosaic.learnerName

    def __call__(self, examples, weightID = 0):
        return MosaicTreeClassifier(self.mosaic, examples, self.statusFunct)


# #############################################################################
# class that builds a tree of mosaics that can be used as a classifier
class MosaicTreeClassifier(orange.Classifier):
    def __init__(self, mosaic, data, statusFunct = None):
        self.mosaic = mosaic

        # discretize domain if necessary
        mosaic.setData(data)
        data = mosaic.data

        stop = orange.TreeStopCriteria_common()
        stop.minExamples = 5
        self.mosaicTree = None

        treeLearner = orange.TreeLearner()
        treeLearner.split = SplitConstructor_MosaicMeasure(mosaic, statusFunct)
        treeLearner.stop = stop
        tree = treeLearner(data)
        if tree.tree and tree.tree.branchSelector:
            self.mosaicTree = self.createTreeNodes(tree.tree, data, None, [1]*len(data))
        if statusFunct:
            if self.mosaicTree:
                statusFunct("Mosaic tree was built successfully.")
            else:
                statusFunct("No tree was generated.")

    def createTreeNodes(self, node, data, parentTreeNode, selectionIndices):
        treeNode = MosaicTreeNode(parentTreeNode, node.branchSelector.attrList)
        treeNode.branchSelector = node.branchSelector
        treeNode.selectionIndices = selectionIndices

        if node.branches:    # if internal node
            for i in range(len(node.branches)):
                if not node.branches[i] or not node.branches[i].branchSelector:
                    continue        # if we are at a leaf

                selectedAttrValues = node.branchDescriptions[i]
                pp = orange.Preprocessor_take()
                pp.values[node.branchSelector.classVar] = selectedAttrValues
                selectedIndices = list(pp.selectionVector(data.select([node.branchSelector.classVar])))
                selectedData = data.selectref(selectedIndices)

                treeNode.branches[selectedAttrValues] = self.createTreeNodes(node.branches[i], selectedData, treeNode, selectedIndices)

        return treeNode


    # for a given example run argumentation and find out to which class it most often fall
    def __call__(self, example, returnType = orange.GetBoth):
        currNode = self.mosaicTree
        while currNode:
            val = currNode.branchSelector.classVar.getValueFrom(example).value
            if currNode.branches.has_key(val):
                currNode = currNode.branches[val]
            else:
                return currNode.branchSelector.classifyExample(example, returnType)        # we are in the leaf of the mosaic tree. classify to the prevailing class


# a measure that evaluates different projections and then says that the best "attribute" is the best projection
class SplitConstructor_MosaicMeasure(orange.TreeSplitConstructor):
    def __init__(self, mosaic, statusFunct = None):
        self.mosaic = mosaic
        self.statusFunct = statusFunct
        self.nodeCount = 0
        self.measure = MeasureAttribute_MDL()

    def updateStatus(self, evaluatingProjections):
        if self.statusFunct:
            s = "%sCurrent tree has %d nodes" % (evaluatingProjections and "Please wait, evaluating projections. " or "", self.nodeCount)
            self.statusFunct(s)

    def __call__(self, gen, weightID, contingencies, apriori, candidates, nodeClassifier):
        self.mosaic.setData(gen)
        self.updateStatus(1)
        if self.mosaic.cancelTreeBuilding:
            return None
        self.mosaic.evaluateProjections()
        if self.mosaic.cancelTreeBuilding or len(self.mosaic.results) == 0:       # or self.mosaic.results[0][0] <= 0:     # if no results or score <=0 then stop building
            self.updateStatus(0)
            return None

        score, attrList, tryIndex, extraInfo = self.mosaic.results[0]

        newFeature = mergeAttrValues(gen, attrList, self.measure, removeUnusedValues = 0)
        dist = orange.Distribution(newFeature, gen).values()
        if max(dist) == sum(dist):    # if all examples belong to one attribute value then this is obviously a useless attribute and we should stop building
            self.updateStatus(0)
            return None

        self.nodeCount += 1
        self.updateStatus(0)
        return (CartesianClassifier(newFeature, attrList, gen), newFeature.values, None, score)


class CartesianClassifier(orange.Classifier):
    def __init__(self, var, attrList, data):
        self.classVar = var
        self.attrList = attrList
        self.data = data
        self.valueMapping = dict(self.createValueDict(attrList, []))

        self.values = {}    # dict of "3-1+8-9+2-2" -> [(3,1), (8,9), (2,2)]
        #for classVal in self.classVar.values:       # we cannot use this because when combining discretized attributes, classVals are c1, c2, ... and not 3-1+...
        #    self.values[classVal] = filter(None, [self.valueMapping.get(val, None) for val in classVal.split("+")])
        for val in self.classVar.values:
            self.values[val] = []
        for ind, combination in enumerate(LimitedCounter([len(data.domain[attr].values) for attr in self.attrList])):
            self.values[self.classVar.getValueFrom.lookupTable[ind].value].append(tuple([data.domain[self.attrList[attrInd]].values[attrValInd] for attrInd, attrValInd in enumerate(combination)]))


    # create a mapping from "0-1-1-4" back to [0, 1, 1, 4]
    def createValueDict(self, attrList, valueList):
        if attrList == []: return valueList

        attrValues = self.data.domain[attrList[0]].values
        if valueList == []:
            return self.createValueDict(attrList[1:], [(val, (val,)) for val in attrValues])
        else:
            newValueList = []
            for val in attrValues:
                newValueList += [(pre+"-"+val, vals+(val,)) for (pre, vals) in valueList]
            return self.createValueDict(attrList[1:], newValueList)


    # determine to which class would the example be classifed based on this combination of attributes
    # i.e. get the majority class for this cartesian product of attributes
    def classifyExample(self, ex, what = orange.Classifier.GetValue):
        val = self.classVar.getValueFrom(ex).value
        classDist = orange.ContingencyAttrClass(self.classVar, self.data)[val]
        classValue = classDist.keys()[classDist.values().index(max(classDist.values()))]
        if what == orange.Classifier.GetValue:
            return orange.Value(self.data.domain.classVar, classValue)
        elif what == orange.Classifier.GetProbabilities:
            return classDist
        else:
            return (orange.Value(self.data.domain.classVar, classValue), classDist)


    # clasify the example ex based on the self.data
    def __call__(self, ex, what = orange.Classifier.GetValue):
        if 1 in [ex[attr].isSpecial() for attr in self.attrList]:
            return orange.Value("?")
        val = self.classVar.getValueFrom(ex)
        if what == orange.Classifier.GetValue:
            return val
        probs = orange.DiscDistribution(self.classVar)
        probs[val] = 1.0
        if what == orange.Classifier.GetProbabilities:
            return probs
        else:
            return (val, probs)


#test widget appearance
if __name__=="__main__":
    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\zoo.tab")
    a = mergeAttrValues(data, ["milk", "legs"], MeasureAttribute_MDL())

    mosaic = orngMosaic()
    mosaic.setData(data)
    mosaic.qualityMeasure = DISTANCE_MEASURE
    mosaic.evaluateProjections()

    learner = MosaicTreeLearner(mosaic)
    classifier = learner(data)
