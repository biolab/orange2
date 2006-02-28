from orngCI import FeatureByCartesianProduct
import orange, OWVisAttrSelection, OWVisFuncts
import time
import Numeric, orngContingency
from math import sqrt, log, e
import orngTest

# quality measures
CHI_SQUARE = 0
CRAMERS_PHI = 1
INFORMATION_GAIN = 2
GAIN_RATIO = 3
INTERACTION_GAIN = 4
AVERAGE_PROBABILITY_OF_CORRECT_CLASSIFICATION = 5

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

# items in the results list
SCORE = 0
ATTR_LIST = 1
TRY_INDEX = 2

CROSSVALIDATION = 0
PROPORTION_TEST = 1

discMeasures = [("None", None), ("ReliefF", orange.MeasureAttribute_relief(k=10, m=50)), ("Gain ratio", orange.MeasureAttribute_gainRatio()), ("Gini index", orange.MeasureAttribute_gini())]

class orngMosaic:
    def __init__(self):
        self.attributeCount = 2
        self.optimizationType = MAXIMUM_NUMBER_OF_ATTRS
        self.qualityMeasure = INFORMATION_GAIN
        self.attrDisc = MEAS_RELIEFF
        self.percentDataUsed = 100

        self.ignoreTooSmallCells = 1    # when computing chi-square and kramer's phi, ignore cells with less than 5 elements

        self.testingMethod = 0
        self.evaluationTime = 2
        self.argumentCount = 5
        self.mValue = 2.0
        self.probabilityEstimation = M_ESTIMATE
        self.learnerName = "Mosaic Classifier"
        self.resultListIndices = []
        self.useOnlyRelevantInteractionsInArgumentation = 1

        self.data = None
        self.evaluatedAttributes = None   # save last evaluated attributes

        self.results = []

    def clearResults(self):
        self.results = []

    def setData(self, data):
        self.data = None
        self.evaluatedAttributes = None
        self.aprioriDistribution = None
        self.logits = {}
        self.arguments = {}
        self.classVals = []

        self.clearResults()

        if not data: return

        # take only discrete attributes
        discAttrs = []
        for attr in data.domain:
            if attr.varType == orange.VarTypes.Discrete: discAttrs.append(attr.name)

        self.data = data.select(discAttrs)
        self.attributeNameIndex = dict([(self.data.domain[i].name, i) for i in range(len(self.data.domain))])

        if self.data.domain.classVar:
            self.classVals = [val for val in self.data.domain.classVar.values]
            self.aprioriDistribution = orange.Distribution(self.data.domain.classVar.name, self.data)
            s = sum(self.aprioriDistribution)
            for i in range(len(self.aprioriDistribution)):
                if 1-(self.aprioriDistribution[i]/s):
                    p = (self.aprioriDistribution[i]/s) / (1-(self.aprioriDistribution[i]/s))
                else: p = 99999.99
                self.logits[self.classVals[i]] = log(p)

    
    # given a dataset return a list of attributes where attribute are sorted by their decreasing importance for class discrimination
    def getEvaluatedAttributes(self, data):
        if not data.domain.classVar or data.domain.classVar.varType != orange.VarTypes.Discrete: return []
        if self.evaluatedAttributes: return self.evaluatedAttributes
        
        try:
            # evaluate attributes using the selected attribute measure
            self.evaluatedAttributes = OWVisAttrSelection.evaluateAttributes(data, None, discMeasures[self.attrDisc][1])
        except:
            import sys
            type, val, traceback = sys.exc_info()
            sys.excepthook(type, val, traceback)  # print the exception
            
        if not self.evaluatedAttributes: return []
        else:                            return self.evaluatedAttributes


    # create a dictionary with all possible pairs of "combination-of-attr-values" : count
    def getConditionalDistributions(self, data, attrs):
        dict = {}
        for i in range(1, len(attrs)+1):
            newFeature, quality = FeatureByCartesianProduct(data, attrs[:i])
            dist = orange.Distribution(newFeature, data)
            for val in newFeature.values: dict[val] = dist[val]

        if data.domain.classVar:
            cont = orange.ContingencyAttrClass(newFeature, data)
            clsValues = data.domain.classVar.values
            for val in newFeature.values:
                for classV in clsValues:
                    dict[val+"-"+classV] = cont[val][classV]
        return dict

    def isOptimizationCanceled(self):
        return (time.time() - self.startTime) / 60 >= self.evaluationTime

    def evaluateProjections(self):
        if not self.data or not self.classVals: return

        self.clearResults()
        self.resultListIndices = []
        
        if self.__class__.__name__ == "OWMosaicOptimization":
            self.disableControls()
            self.parentWidget.progressBarInit()
            from qt import qApp
            
        self.startTime = time.time()

        if self.optimizationType == 0: maxLength = self.attributeCount; minLength = self.attributeCount
        else:                          maxLength = self.attributeCount; minLength = 1

        evaluatedAttrs = self.getEvaluatedAttributes(self.data)
        if evaluatedAttrs == []:
            self.finishEvaluation(0)
            return

        triedPossibilities = 0; totalPossibilities = 0
        if self.optimizationType == 0: totalPossibilities = OWVisFuncts.combinationsCount(self.attributeCount, len(evaluatedAttrs))
        else:
            for i in range(1, self.attributeCount+1): totalPossibilities += OWVisFuncts.combinationsCount(i, len(evaluatedAttrs))

        for z in range(len(evaluatedAttrs)):
            for u in range(minLength-1, maxLength):
                combinations = OWVisFuncts.combinations(evaluatedAttrs[:z], u)
                
                for attrList in combinations:
                    attrs = [evaluatedAttrs[z]] + attrList

                    val = self._Evaluate(attrs)

                    if self.isOptimizationCanceled():
                        self.finishEvaluation(triedPossibilities)
                        return

                    triedPossibilities += 1
                    self.insertItem(val, attrs, self.findTargetIndex(val, max), triedPossibilities)
                    
                    if self.__class__.__name__ == "OWMosaicOptimization":
                        self.parentWidget.progressBarSet(100.0*triedPossibilities/float(totalPossibilities))
                        self.setStatusBarText("Evaluated %s visualizations..." % (OWVisFuncts.createStringFromNumber(triedPossibilities)))
                        qApp.processEvents()        # allow processing of other events

        self.finishEvaluation(triedPossibilities)


    def finishEvaluation(self, evaluatedProjections):
        if self.__class__.__name__ == "OWMosaicOptimization":
            secs = time.time() - self.startTime
            self.setStatusBarText("Evaluation stopped (evaluated %s projections in %d min, %d sec)" % (OWVisFuncts.createStringFromNumber(evaluatedProjections), secs/60, secs%60))
            self.parentWidget.progressBarFinished()
            self.enableControls()
            self.finishedAddingResults()


    def _Evaluate(self, attrs):
        newFeature, quality = FeatureByCartesianProduct(self.data, attrs)
        
        classEntropy = orngContingency.Entropy(Numeric.array([val for val in self.aprioriDistribution]))

        retVal = -1
        if self.qualityMeasure in [CHI_SQUARE, CRAMERS_PHI]:
            aprioriSum = sum(self.aprioriDistribution)
            retVal = 0.0

            for dist in orange.ContingencyAttrClass(newFeature, self.data):
                for i in range(len(self.aprioriDistribution)):
                    expected = sum(dist) * (self.aprioriDistribution[i] / float(aprioriSum))
                    if self.ignoreTooSmallCells and expected < 5: continue       # statisticians advise ignoring terms that have less than 5 expected examples, since they can significantly disturb chi-square
                    if not expected: continue                                    # prevent division by zero
                    retVal += (dist[i] - expected)**2 / expected

            if self.qualityMeasure == CRAMERS_PHI:
                vals = min(len(newFeature.values), len(self.data.domain.classVar.values))-1
                if vals:
                    retVal = sqrt(retVal / (len(self.data) * vals))
                    
        elif self.qualityMeasure == GAIN_RATIO:
            retVal = orange.MeasureAttribute_gainRatio(newFeature, self.data)
        elif self.qualityMeasure == INFORMATION_GAIN:
            retVal = orange.MeasureAttribute_info(newFeature, self.data)
            if classEntropy: retVal = retVal * 100.0 / classEntropy
        elif self.qualityMeasure == INTERACTION_GAIN:
            new = orange.MeasureAttribute_info(newFeature, self.data)
            gains = [orange.MeasureAttribute_info(attr, self.data) for attr in attrs]
            retVal = new - sum(gains)
            if classEntropy: retVal = retVal * 100.0 / classEntropy
        elif self.qualityMeasure == AVERAGE_PROBABILITY_OF_CORRECT_CLASSIFICATION:
            
            d = self.data.select([newFeature, self.data.domain.classVar])     # create a dataset that has only this new feature and class info
            
            if self.testingMethod == PROPORTION_TEST:
                pick = orange.MakeRandomIndices2(stratified = orange.MakeRandomIndices.StratifiedIfPossible, p0 = 0.7, randomGenerator = 0)
                indices = [pick(d) for i in range(10)]
            elif self.testingMethod == CROSSVALIDATION:
                ind = orange.MakeRandomIndicesCV(d, 10, randseed="*", stratified = orange.MakeRandomIndices.StratifiedIfPossible)
                indices = [[val == i for val in ind] for i in range(10)]
            
            acc = 0.0; count = 0
            for ind in indices:
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
            
            """
            retVal = 0.0
            for dist in orange.ContingencyAttrClass(newFeature, self.data):
                s = sum(dist)
                if s: retVal += (100.0 * s * sum([(v/float(s))**2 for v in dist])) /float(len(self.data))
            """

        return retVal


    # use bisection to find correct index
    def findTargetIndex(self, score, funct):
        top = 0; bottom = len(self.results)

        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if funct(score, self.results[mid][SCORE]) == score: bottom = mid
            else: top = mid

        if len(self.results) == 0: return 0
        if funct(score, self.results[top][SCORE]) == score:
            return top
        else: 
            return bottom

    # insert new result - give parameters: score of projection, number of examples in projection and list of attributes.
    def insertItem(self, score, attrList, index, tryIndex):
        self.results.insert(index, (score, attrList, tryIndex))
        
        if self.__class__.__name__ == "OWMosaicOptimization":
            string = ""
            if self.showRank: string += str(index+1) + ". "
            if self.showScore: string += "%.2f : " % (score)
            string += self.buildAttrString(attrList)
            self.resultList.insertItem(string, index)
            self.resultListIndices.insert(index, index)

    # from a list of attributes build a nice string with attribute names
    def buildAttrString(self, attrList):
        if len(attrList) == 0: return ""
        
        strList = attrList[0]
        for attr in attrList[1:]:
            strList += ", " + attr
        return strList

    # ######################################################
    # Argumentation functions
    # ######################################################
    def findArguments(self, example = None):
        self.arguments = dict([(val, []) for val in self.classVals])

        if not example or len(self.results) == 0: return None, None
        if not (self.data and self.data.domain.classVar and self.logits and self.classVals): return None, None

        if self.__class__.__name__ == "OWMosaicOptimization":
            from qt import qApp

        for index in range(len(self.results)):
            (score, attrList, tryIndex) = self.results[index]
            args, errors = self.evaluateArgument(example, attrList, score)
            if not args: continue
            
            for val in self.classVals:
                pos = self.getArgumentIndex(args[val], val)
                self.arguments[val].insert(pos, (args[val], score, attrList, index, errors[val]))
                if self.__class__.__name__ == "OWMosaicOptimization" and val == str(self.classValueList.currentText()):
                    self.insertArgument(args[val], errors[val], attrList, pos)
                    qApp.processEvents()

        #predictions = [self.aprioriDistribution[i] for i in range(len(self.aprioriDistribution))]
        #for j in range(nrOfClases):
        #   for i in range(min(self.argumentCount, nrOfArguments))):
        #        predictions[j] *= self.arguments[j][i][0]

        nrOfArguments = len(self.arguments[self.classVals[0]])
        #nrOfClases = len(self.aprioriDistribution)
        #predictions = [0 for i in range(nrOfClases)]
        predictions = []
        
        #for j in range(nrOfClases):
        for val in self.classVals:
            #predictions[j] = sum([self.arguments[j][i][1] * self.arguments[j][i][0] for i in range(min(nrOfArguments, self.argumentCount))])   # projection score * argument value
            #predictions[j] = sum([self.arguments[j][i][0] for i in range(min(nrOfArguments, self.argumentCount))])                              # argument value
            
            # take best self.argumentCount arguments and the worst self.argumentCount arguments. sum them together and this is it. 
            #if nrOfArguments < self.argumentCount * 2: predictions[j] = sum([self.arguments[j][i][0] for i in range(nrOfArguments)])
            #else: predictions[j] = sum([self.arguments[j][i][0] for i in range(self.argumentCount)] + [self.arguments[j][i][0] for i in range(nrOfArguments-self.argumentCount, nrOfArguments)])
            predictions.append(self.logits[val] + sum([v[0] for v in self.arguments[val]]))
            """
            predictions[j] = 1.0
            for arg in self.arguments[j]:
                predictions[j] *= arg[0]
            """
            
##        Min = min(predictions)
##        if Min < 0:
##            predictions = [val - Min for val in predictions]

        # use predictions from all arguments to classify an example
        probabilities = []
        for i, val in enumerate(predictions):
            if val < -100: p = 0
            else:         p = 1 / (1 + e**-val)
            probabilities.append(p)
        #self.printVerbose(str(probabilities) + " " + str(sum(probabilities)))
        classValue = self.data.domain.classVar[probabilities.index(max(probabilities))]
        dist = orange.DiscDistribution([val/float(sum(probabilities)) for val in probabilities])
        dist.variable = self.data.domain.classVar
        return classValue, dist



    def estimateClassProbabilities(self, data, example, attrList, subData = None, subDataDistribution = None, aprioriDistribution = None, probabilityEstimation = -1, mValue = -1):
        if probabilityEstimation == -1: probabilityEstimation = self.probabilityEstimation
        if aprioriDistribution == None: aprioriDistribution = self.aprioriDistribution
        if mValue == -1: mValue = self.mValue
        
        if not subData:
            attrVals = [example[attr] for attr in attrList]
            if "?" in attrVals: return None      # the testExample has a missing value at one of the visualized attributes
            subData = orange.Preprocessor_take(data, values = dict([(data.domain[attr], example[attr]) for attr in attrList]))
        if not subDataDistribution:
            subDataDistribution = orange.Distribution(data.domain.classVar.name, subData)

        lenData = len(data)
        lenSubData = len(subData)
        
        aprioriProbabilities = [nrOfCases / float(lenData) for nrOfCases in aprioriDistribution]
        actualProbabilities = []

        # estimate probabilities
        if probabilityEstimation == RELATIVE:
            if not len(subData):
                self.printVerbose("OWMosaicOptimization: Empty data subset. Unable to compute relative frequency.")
                return [0.0 for i in range(len(aprioriDistribution))]      # prevent division by zero
            actualProbabilities = [ subDataDistribution[index] / float(lenSubData) for index in range(len(aprioriDistribution))]      # P(c_i | a_k) / P(c_i)

        elif probabilityEstimation == LAPLACE:
            for index in range(len(aprioriDistribution)):
                actualProbabilities.append((subDataDistribution[index]+1) / float(lenSubData+len(subDataDistribution)))      # (r+1 / n+c) / P(c_i)

        elif probabilityEstimation == M_ESTIMATE:
            for index in range(len(aprioriDistribution)):
                n = subDataDistribution[index]
                pa = aprioriDistribution[index]/float(sum(aprioriDistribution))
                actualProbabilities.append((pa * mValue + n) / float(lenSubData + mValue))       # p = (pa*m+n)/(N+m)

        return actualProbabilities

    # compute probability that the combination of attribute values in attrValues is significantly different from being independent
    # see Kononenko: Semi-naive Bayes
    def getInteractionImportanceProbability(self, data, attrList, attrValues, subData = None, contingencies = None, aprioriDistribution = None, subDataDistribution = None):
        assert len(attrList) == len(attrValues)

        if not subData:
            subData = orange.Preprocessor_take(data, values = dict([(data.domain[attrList[i]], attrValues[i]) for i in range(len(attrList))]))
        if not aprioriDistribution: aprioriDistribution = self.aprioriDistribution
        if not subDataDistribution:
            subDataDistribution = orange.Distribution(data.domain.classVar.name, subData)
        if not contingencies:
            contingencies = [orange.ContingencyAttrClass(attr, data) for attr in attrList]

        lenSubData = len(subData)
        lenData = len(data)
        eps = 0.0
        for i in range(len(aprioriDistribution)):
            product = 1.0
            for x in [contingencies[j][attrValues[j]] for j in range(len(attrList))]:
                product *= (x[i] / sum(x))      # P(cj|ai)
            eps += (aprioriDistribution[i]/lenData) * abs((subDataDistribution[i]/lenSubData) - (product*lenData)/aprioriDistribution[i])

        if eps*lenSubData == 0: return 9999999
        return 1/(4*eps*eps*lenSubData)         


    # for a given example and a projection evaluate arguments for each class value
    def evaluateArgument(self, example, attrList, score):
        attrVals = [example[attr] for attr in attrList]
        if "?" in attrVals: return None, None      # the testExample has a missing value at one of the visualized attributes

        subData = orange.Preprocessor_take(self.data, values = dict([(self.data.domain[attr], example[attr]) for attr in attrList]))
        if not subData or len(subData) == 0: return None, None

        lenData = len(self.data)
        lenSubData = len(subData)

        if self.useOnlyRelevantInteractionsInArgumentation:
            P = self.getInteractionImportanceProbability(self.data, attrList, [example[attr] for attr in attrList], subData)
            if P > 0.5: return None, None

        arguments = {}
        errors = {}

        aprioriProbabilities = [nrOfCases / float(lenData) for nrOfCases in self.aprioriDistribution]
        actualProbabilities = self.estimateClassProbabilities(self.data, example, attrList, subData)    # estimate probabilities for the example and given attribute list
        #print "apriori",attrList[0],  example[attrList[0]], aprioriProbabilities
        #print "actual", attrList[0], example[attrList[0]], actualProbabilities
                
        for i in range(len(self.aprioriDistribution)):
            # compute log odds of P(c|ai)/P(c)
            val = 0
            #val = (actualProbabilities[i] ) / (aprioriProbabilities[i])
            if (1-actualProbabilities[i]) * aprioriProbabilities[i] > 0.0:
                val = (actualProbabilities[i] * (1-aprioriProbabilities[i])) / ((1-actualProbabilities[i]) * aprioriProbabilities[i])
                if val > 0: val = log(val)
                else: val = -999.99
            else: val = 999.99

            # compute confidence interval
            approxZero = 0.0001
            aprioriPc0 = max(approxZero, aprioriProbabilities[i]); aprioriPc1 = max(approxZero, 1-aprioriProbabilities[i])
            actualPc0  = max(approxZero, actualProbabilities[i]);  actualPc1  = max(approxZero, 1-actualProbabilities[i])
            part1 = 1 / (len(self.data) * aprioriPc0 * aprioriPc1)
            part2 = 1 / max(approxZero, lenSubData  * actualPc0  * actualPc1)
            error = sqrt(max(0, part2 - part1)) * 1.64

            arguments[self.classVals[i]] = val
            errors[self.classVals[i]] = error

        """
        # prepare array of arguments for each class value
        for i in range(len(self.aprioriDistribution)):  # compute the value of the argument as peasons residual * sqrt(lenSubData)
            #pearson = lenSubData*(actualProbabilities[i] - aprioriProbabilities[i]) / sqrt(lenSubData*aprioriProbabilities[i])
            #arguments.append(pearson * sqrt(lenSubData) * score)
            #arguments.append(actualProbabilities[i] / float(aprioriProbabilities[i]))

            # compute log odds of P(c|ai)/P(c)
            val = 0
            if (1-actualProbabilities[i]) * aprioriProbabilities[i] > 0.0:
                val = (actualProbabilities[i] * (1-aprioriProbabilities[i])) / ((1-actualProbabilities[i]) * aprioriProbabilities[i])
                if val > 0: val = log(val)
                else: val = -999.99
            else: val = 999.99

            # compute confidence interval
            approxZero = 0.0001
            aprioriPc0 = max(approxZero, aprioriProbabilities[i]); aprioriPc1 = max(approxZero, 1-aprioriProbabilities[i])
            actualPc0  = max(approxZero, actualProbabilities[i]);  actualPc1  = max(approxZero, 1-actualProbabilities[i])
            part1 = 1 / (len(self.data) * aprioriPc0 * aprioriPc1)
            part2 = 1 / max(approxZero, lenSubData  * actualPc0  * actualPc1)
            error = sqrt(max(0, part2 - part1)) * 1.64

            arguments.append(val)
            errors.append(error)
        """
        
        return arguments, errors


    def getArgumentIndex(self, value, classValue):
        if len(self.arguments[classValue]) == 0: return 0
        
        top = 0; bottom = len(self.arguments[classValue])
        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if max(value, self.arguments[classValue][mid][0]) == value: bottom = mid
            else: top = mid

        if max(value, self.arguments[classValue][top][0]) == value:  return top
        else:                                                        return bottom
        

# #############################################################################
# class that represents kNN classifier that classifies examples based on top evaluated projections
class MosaicVizRankClassifier(orange.Classifier):
    def __init__(self, mosaic, data):
        self.Mosaic = mosaic

        if self.Mosaic.__class__.__name__ == "OWMosaicOptimization":
            self.Mosaic.parentWidget.subsetdataHander(None)
            self.Mosaic.parentWidget.cdata(data)
        else:
            self.Mosaic.setData(data)

        if self.Mosaic.__class__.__name__ == "OWMosaicOptimization": self.Mosaic.useTimeLimit = 1   
        self.Mosaic.evaluateProjections()
        if self.Mosaic.__class__.__name__ == "OWMosaicOptimization": del self.Mosaic.useTimeLimit


    # for a given example run argumentation and find out to which class it most often fall        
    def __call__(self, example, returnType = orange.GetBoth):
        if self.Mosaic.__class__.__name__ == "OWMosaicOptimization":
            table = orange.ExampleTable(example.domain)
            table.append(example)
            self.Mosaic.parentWidget.subsetdataHander(table)
            classVal, prob = self.Mosaic.findArguments(example, 0, 0)
        else:
            classVal, prob = self.Mosaic.findArguments(example)

        if returnType == orange.GetBoth: return classVal, prob
        else:                            return classVal
        

# #############################################################################
# learner that builds MosaicVizRankLearner
class MosaicVizRankLearner(orange.Learner):
    def __init__(self, mosaic = None):
        if not mosaic: mosaic = orngMosaic()
        self.Mosaic = mosaic
        self.name = self.Mosaic.learnerName
        
    def __call__(self, examples, weightID = 0):
        return MosaicVizRankClassifier(self.Mosaic, examples)
