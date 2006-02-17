from orngCI import FeatureByCartesianProduct
import orange, OWVisAttrSelection, OWVisFuncts
import time
from math import sqrt, log

# quality measures
CHI_SQUARE = 0
GAIN_RATIO = 1
INFORMATION_GAIN = 2
INTERACTION_GAIN = 3
AVERAGE_PROBABILITY_OF_CORRECT_CLASSIFICATION = 4

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

discMeasures = [("None", None), ("ReliefF", orange.MeasureAttribute_relief(k=10, m=50)), ("Gain ratio", orange.MeasureAttribute_gainRatio()), ("Gini index", orange.MeasureAttribute_gini())]

class orngMosaic:
    def __init__(self):
        self.attributeCount = 2
        self.optimizationType = MAXIMUM_NUMBER_OF_ATTRS
        self.qualityMeasure = INFORMATION_GAIN
        self.attrDisc = MEAS_RELIEFF
        self.percentDataUsed = 100

        self.evaluationTime = 2
        self.argumentCount = 5
        self.mValue = 2.0
        self.probabilityEstimation = M_ESTIMATE
        self.learnerName = "Mosaic Classifier"

        self.data = None
        self.evaluatedAttributes = None   # save last evaluated attributes

        self.results = []

    def clearResults(self):
        self.results = []

    def setData(self, data):
        self.data = None
        self.evaluatedAttributes = None
        self.aprioriDistribution = None
        self.clearResults()

        if not data: return

        # take only discrete attributes
        discAttrs = []
        for attr in data.domain:
            if attr.varType == orange.VarTypes.Discrete: discAttrs.append(attr.name)

        self.data = data.select(discAttrs)
        self.attributeNameIndex = dict([(self.data.domain[i].name, i) for i in range(len(self.data.domain))])

    
    # get only the data examples that belong to one of the selected class values
    def getData(self):
        return self.data


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
        if not self.data: return

        self.clearResults()
        if self.__class__.__name__ == "OWMosaicOptimization":
            self.disableControls()
            self.parentWidget.progressBarInit()
            from qt import qApp
            
        self.startTime = time.time()

        hasMissingData = (len(self.data) != len(orange.Preprocessor_dropMissing(self.data)))
        if self.optimizationType == 0: maxLength = self.attributeCount; minLength = self.attributeCount
        else:                          maxLength = self.attributeCount; minLength = 1

        data = self.getData()   # get only the examples that have one of the class values that is selected in the class value list
        if not data:
            if self.__class__.__name__ == "OWMosaicOptimization": QMessageBox.critical(None,'No data','There is no data or no class value is selected in the Manage tab.',QMessageBox.Ok)
            return
            
        if self.percentDataUsed != 100:
            indices = orange.MakeRandomIndices2(data, 1.0-float(self.percentDataUsed)/100.0)
            data = data.select(indices)

        evaluatedAttrs = self.getEvaluatedAttributes(data)
        if evaluatedAttrs == []:
            self.finishEvaluation(0)
            return

        self.aprioriDistribution = orange.Distribution(data.domain.classVar.name, data)
        #attributes = [self.attributeNameIndex[name] for name in evaluatedAttrs]
        classIndex = self.attributeNameIndex[data.domain.classVar.name]

        triedPossibilities = 0; totalPossibilities = 0
        if self.optimizationType == 0: totalPossibilities = OWVisFuncts.combinationsCount(self.attributeCount, len(evaluatedAttrs))
        else:
            for i in range(1, self.attributeCount+1): totalPossibilities += OWVisFuncts.combinationsCount(i, len(evaluatedAttrs))

        for z in range(len(evaluatedAttrs)):
            for u in range(minLength-1, maxLength):
                combinations = OWVisFuncts.combinations(evaluatedAttrs[:z], u)
                
                for attrList in combinations:
                    attrs = [evaluatedAttrs[z]] + attrList

                    val = self._Evaluate(data, attrs)

                    if self.isOptimizationCanceled():
                        self.finishEvaluation(triedPossibilities)
                        return

                    triedPossibilities += 1
                    self.insertItem(val, attrs, self.findTargetIndex(val, max), triedPossibilities)
                    
                    if self.__class__.__name__ == "OWMosaicOptimization":
                        #qApp.processEvents()
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


    def _Evaluate(self, data, attrs):
        newFeature, quality = FeatureByCartesianProduct(data, attrs)
        #print "started eval with", data, attrs
        
        if self.qualityMeasure == GAIN_RATIO:
            retVal = orange.MeasureAttribute_gainRatio(newFeature, data)
        elif self.qualityMeasure == INFORMATION_GAIN:
            retVal = orange.MeasureAttribute_info(newFeature, data)
        elif self.qualityMeasure == INTERACTION_GAIN:
            new = orange.MeasureAttribute_info(newFeature, data)
            gains = [orange.MeasureAttribute_info(attr, data) for attr in attrs]
            retVal = new - sum(gains)
        elif self.qualityMeasure == AVERAGE_PROBABILITY_OF_CORRECT_CLASSIFICATION:
            retVal = 0.0
            for dist in orange.ContingencyAttrClass(newFeature, data):
                s = sum(dist)
                if s: retVal += (100.0 * s * sum([(v/float(s))**2 for v in dist])) /float(len(data))
        else:
            aprioriSum = sum(self.aprioriDistribution)
            retVal = 0.0

            for dist in orange.ContingencyAttrClass(newFeature, data):
                for i in range(len(self.aprioriDistribution)):
                    expected = float(len(data) * self.aprioriDistribution.values()[i]) / float(aprioriSum)
                    if expected < 5: continue       # statisticians advise ignoring terms that have less than 5 expected examples, since they can significantly disturb chi-square
                    
                    if self.qualityMeasure == CHI_SQUARE and expected:
                        retVal += (dist[i] - expected)**2 / expected
        #del newFeature
        #print "returning"
        return retVal


    def getProjectionQuality(self, data, attrList):
        if not self.aprioriDistribution: self.aprioriDistribution = orange.Distribution(data.domain.classVar.name, data)
        return self._Evaluate(data, attrList)


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
        self.arguments = [[] for i in range(len(self.data.domain.classVar.values))]

        if not example or len(self.results) == 0: return None, None
        
        data = self.getData()   # get only the examples that have one of the class values that is selected in the class value list
        if not data:  return None, None

        if self.percentDataUsed != 100:
            indices = orange.MakeRandomIndices2(data, 1.0-float(self.percentDataUsed)/100.0)
            data = data.select(indices)

        self.aprioriDistribution = orange.Distribution(data.domain.classVar.name, data)
        currentClassValue = 0
        if self.__class__.__name__ == "OWMosaicOptimization":
            currentClassValue = self.classValueList.currentItem()
            from qt import qApp
        
        for index in range(len(self.results)):
            #if argumentsUsed >= self.argumentCount: break     # we have enough arguments. we stop.
            
            (score, attrList, tryIndex) = self.results[index]
            args, errors = self.evaluateArgument(data, example, attrList, score)
            if not args: continue
            
            for i in range(len(args)):
                pos = self.getArgumentIndex(args[i], i)
                self.arguments[i].insert(pos, (args[i], score, attrList, index, errors[i]))
                if self.__class__.__name__ == "OWMosaicOptimization" and i == currentClassValue:
                    self.insertArgument(args[i], errors[i], attrList, pos)
                    qApp.processEvents()
                    
            #argumentsUsed += 1

        #predictions = [self.aprioriDistribution[i] for i in range(len(self.aprioriDistribution))]
        #for j in range(nrOfClases):
        #   for i in range(min(self.argumentCount, nrOfArguments))):
        #        predictions[j] *= self.arguments[j][i][0]

        nrOfArguments = len(self.arguments[0])
        nrOfClases = len(self.aprioriDistribution)
        predictions = [0 for i in range(nrOfClases)]
        
        for j in range(nrOfClases):
            #predictions[j] = sum([self.arguments[j][i][1] * self.arguments[j][i][0] for i in range(min(nrOfArguments, self.argumentCount))])   # projection score * argument value
            #predictions[j] = sum([self.arguments[j][i][0] for i in range(min(nrOfArguments, self.argumentCount))])                              # argument value
            
            # take best self.argumentCount arguments and the worst self.argumentCount arguments. sum them together and this is it. 
            if nrOfArguments < self.argumentCount * 2: predictions[j] = sum([self.arguments[j][i][0] for i in range(nrOfArguments)])
            else: predictions[j] = sum([self.arguments[j][i][0] for i in range(self.argumentCount)] + [self.arguments[j][i][0] for i in range(nrOfArguments-self.argumentCount, nrOfArguments)])
            
        Min = min(predictions)
        if Min < 0:
            predictions = [val - Min for val in predictions]
            
        return self.classifyExampleFromPredictions(predictions)


    # for a given example and a projection evaluate arguments for each class value
    def evaluateArgument(self, data, example, attrList, score):
        attrVals = [example[attr] for attr in attrList]
        if "?" in attrVals: return None, None      # the testExample has a missing value at one of the visualized attributes

        subData = orange.Preprocessor_take(data, values = dict([(data.domain[attr], example[attr]) for attr in attrList]))
        if not subData or len(subData) == 0: return None, None

        aprioriSum = sum(self.aprioriDistribution)
        actualDistribution = orange.Distribution(data.domain.classVar.name, subData)
        actualSum = len(subData)
        aprioriProbabilities = [nrOfCases / float(aprioriSum) for nrOfCases in self.aprioriDistribution]
                
        actualProbabilities = []
        
        # estimate probabilities
        if self.probabilityEstimation == RELATIVE:
            if not len(subData):
                self.printVerbose("OWMosaicOptimization: Empty data subset. Unable to compute relative frequency.")
                return [0.0 for i in range(len(self.aprioriDistribution))]      # prevent division by zero
            actualProbabilities = [ actualDistribution[index] / float(len(subData)) for index in range(len(self.aprioriDistribution))]      # P(c_i | a_k) / P(c_i)

        elif self.probabilityEstimation == LAPLACE:
            for index in range(len(self.aprioriDistribution)):
                actualProbabilities.append((actualDistribution[index]+1) / float(len(subData)+len(actualDistribution)))      # (r+1 / n+c) / P(c_i)

        elif self.probabilityEstimation == M_ESTIMATE:
            for index in range(len(self.aprioriDistribution)):
                n = actualDistribution[index]
                pa = self.aprioriDistribution[index]/float(sum(self.aprioriDistribution))
                actualProbabilities.append((pa * self.mValue + n) / float(sum(actualDistribution) + self.mValue))       # p = (pa*m+n)/(N+m)

        arguments = []
        errors = []

        # prepare array of arguments for each class value
        for i in range(len(self.aprioriDistribution)):  # compute the value of the argument as peasons residual * sqrt(actualSum)
            #pearson = actualSum*(actualProbabilities[i] - aprioriProbabilities[i]) / sqrt(actualSum*aprioriProbabilities[i])
            #arguments.append(pearson * sqrt(actualSum) * score)
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
            part1 = 1 / (len(data) * aprioriPc0 * aprioriPc1)
            part2 = 1 / max(approxZero, actualSum  * actualPc0  * actualPc1)
            error = sqrt(max(0, part2 - part1)) * 1.64

            arguments.append(val)
            errors.append(error)
        
        if sum(arguments) == 0: return None, None   # if there is a zero probability for all class values, then skip this argument
        return arguments, errors


    # use predictions from all arguments to classify an example
    def classifyExampleFromPredictions(self, predictions):
        if sum(predictions) == 0:
            dist = orange.Distribution(self.data.domain.classVar.name, self.data)
            predictions = [dist[i] for i in range(len(dist))]

        # find the most probable class value and return it with its probability
        classValue = self.data.domain.classVar[predictions.index(max(predictions))]
        dist = orange.DiscDistribution([val/float(sum(predictions)) for val in predictions])
        dist.variable = self.data.domain.classVar
        return classValue, dist


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
# learner that builds VizRankClassifier
class MosaicVizRankLearner(orange.Learner):
    def __init__(self, mosaic = None):
        if not mosaic: mosaic = orngMosaic()
        self.Mosaic = mosaic
        self.name = self.Mosaic.learnerName
        
    def __call__(self, examples, weightID = 0):
        return MosaicVizRankClassifier(self.Mosaic, examples)

