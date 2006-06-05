from orngCI import FeatureByCartesianProduct
import orange, orngVisFuncts
import time, operator
import Numeric, orngContingency
from math import sqrt, log, e
import orngTest, orngStat
from orngScaleData import getVariableValuesSorted, getVariableValueIndices

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
        self.mValue = 2.0
        self.probabilityEstimation = M_ESTIMATE
        self.learnerName = "Mosaic Classifier"
        self.resultListIndices = []
        #self.useOnlyRelevantInteractionsInArgumentation = 0
        self.automaticallyRemoveWeakerArguments = 1

        self.data = None
        self.evaluatedAttributes = None   # save last evaluated attributes
        self.stopOptimization = 0           # used to stop attribute and value order

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

        # discretize attributes
        entroDisc = orange.EntropyDiscretization()
        discAttrs = []
        for attr in data.domain:
            if attr.varType == orange.VarTypes.Discrete:
                discAttrs.append(attr)
            elif data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete:
                newAttr = entroDisc(attr, data)
                newAttr.name = attr.name
                if len(newAttr.values) > 1:
                    discAttrs.append(newAttr)
        self.data = data.select(discAttrs)
        self.attributeNameIndex = dict([(self.data.domain[i].name, i) for i in range(len(self.data.domain))])

        if self.data.domain.classVar:
            self.classVals = [val for val in self.data.domain.classVar.values]
            self.aprioriDistribution = orange.Distribution(self.data.domain.classVar.name, self.data)
            s = sum(self.aprioriDistribution)
            for i in range(len(self.aprioriDistribution)):
                if (self.aprioriDistribution[i]/s) > 0 and (self.aprioriDistribution[i]/s) < 1:
                    p = (self.aprioriDistribution[i]/s) / (1-(self.aprioriDistribution[i]/s))
                elif (self.aprioriDistribution[i]/s) == 0:
                    p = 0.0001
                elif (self.aprioriDistribution[i]/s) == 1:
                    p = 99999.99
                self.logits[self.classVals[i]] = log(p)

    
    # given a dataset return a list of attributes where attribute are sorted by their decreasing importance for class discrimination
    def getEvaluatedAttributes(self, data):
        if not data.domain.classVar or data.domain.classVar.varType != orange.VarTypes.Discrete: return []
        if self.evaluatedAttributes: return self.evaluatedAttributes
        
        try:
            # evaluate attributes using the selected attribute measure
            self.evaluatedAttributes = orngVisFuncts.evaluateAttributes(data, None, discMeasures[self.attrDisc][1])
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
        if self.optimizationType == 0: totalPossibilities = orngVisFuncts.combinationsCount(self.attributeCount, len(evaluatedAttrs))
        else:
            for i in range(1, self.attributeCount+1): totalPossibilities += orngVisFuncts.combinationsCount(i, len(evaluatedAttrs))

        for z in range(len(evaluatedAttrs)):
            for u in range(minLength-1, maxLength):
                combinations = orngVisFuncts.combinations(evaluatedAttrs[:z], u)
                
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
                        self.setStatusBarText("Evaluated %s visualizations..." % (orngVisFuncts.createStringFromNumber(triedPossibilities)))
                        qApp.processEvents()        # allow processing of other events

        self.finishEvaluation(triedPossibilities)


    def finishEvaluation(self, evaluatedProjections):
        if self.__class__.__name__ == "OWMosaicOptimization":
            secs = time.time() - self.startTime
            self.setStatusBarText("Evaluation stopped (evaluated %s projections in %d min, %d sec)" % (orngVisFuncts.createStringFromNumber(evaluatedProjections), secs/60, secs%60))
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
            
            self.resultList.insertItem("%.2f : %s" % (score, self.buildAttrString(attrList)), index)
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

##        if self.useOnlyRelevantInteractionsInArgumentation:
##            self.classDistributionsForExample = {}
##            for attr in self.data.domain.attributes:
##                temp = self.data.filter({attr:example[attr]})
##                self.classDistributionsForExample[attr.name] = orange.Distribution(self.data.domain.classVar, temp)

        if self.__class__.__name__ == "OWMosaicOptimization":
            from qt import qApp

        for index in range(len(self.results)):
            (score, attrList, tryIndex) = self.results[index]
            args = self.evaluateArgument(example, attrList, score)      # call evaluation of a specific projection
            if not args: continue
            
            for val in self.classVals:
                pos = self.getArgumentIndex(args[val][0], val)
                self.arguments[val].insert(pos, (args[val][0], score, attrList, index, args[val][1]))
                if self.__class__.__name__ == "OWMosaicOptimization" and val == str(self.classValueList.currentText()):
                    self.insertArgument(args[val][0], args[val][1], attrList, pos)
                    qApp.processEvents()

        if self.automaticallyRemoveWeakerArguments:
            self.removeWeakerArguments()

        nrOfArguments = len(self.arguments.values()[0])
        predictions = []
        
        for val in self.classVals:
            predictions.append(self.logits[val] + sum([v[0] for v in self.arguments[val]]))
            
        # use predictions from all arguments to classify an example
        probabilities = []
        for i, val in enumerate(predictions):
            if val < -100: p = 0
            else:         p = 1 / (1 + e**-val)
            probabilities.append(p)

        classValue = self.data.domain.classVar[probabilities.index(max(probabilities))]
        dist = orange.DiscDistribution([val/float(sum(probabilities)) for val in probabilities])
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

##        if self.useOnlyRelevantInteractionsInArgumentation and len(attrList) > 1:
##            P = self.getInteractionImportanceProbability(attrList, [example[attr] for attr in attrList], subData)
##            if P > 0.5: return None

        arguments = {}

        aprioriProbabilities = [nrOfCases / float(lenData) for nrOfCases in self.aprioriDistribution]
        actualProbabilities = self.estimateClassProbabilities(self.data, example, attrList, subData)    # estimate probabilities for the example and given attribute list
                
        for i in range(len(self.aprioriDistribution)):
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

            arguments[self.classVals[i]] = (val, error)

        return arguments

        
    # probability estimation function
    def estimateClassProbabilities(self, data, example, attrList, subData = None, subDataDistribution = None, aprioriDistribution = None, probabilityEstimation = -1, mValue = -1):
        if probabilityEstimation == -1: probabilityEstimation = self.probabilityEstimation
        if aprioriDistribution == None: aprioriDistribution = self.aprioriDistribution
        if mValue == -1: mValue = self.mValue
        
        if not subData:
            attrVals = [example[attr] for attr in attrList]
            if "?" in attrVals: return None      # the testExample has a missing value at one of the visualized attributes
            subData = self.getDataSubset(attrList, attrVals)
            #subData = orange.Preprocessor_take(data, values = dict([(data.domain[attr], example[attr]) for attr in attrList]))
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

        s = sum(actualProbabilities)
        if "%.3f" % s != "1.000":
            print "probabilities don't sum to 1, but to", s, actualProbabilities

        return actualProbabilities


    def removeWeakerArguments(self):
        if not self.arguments or not self.arguments.values(): return

        for classValue in self.arguments.keys():
            # create a dict for arguments of different lengths. each dict has arguments of this length and corresponding scores
            argList = {1: {}, 2: {}, 3:{}, 4:{}}
            arguments = self.arguments[classValue]
            for i in range(len(arguments)):
                attrs = arguments[i][2]
                attrs.sort()
                argList[len(attrs)][tuple(attrs)] = (arguments[i][0], i)

            if len(argList[1]) == 0: return     # in case that we only evaluated projections with EXACTLY X attributes we cannot remove weak arguments

            for count in [4,3,2]:
                args = argList[count]

                candidates = []     # candidate projections for deleting
                for key in args.keys():
                    splits = orngVisFuncts.getPossibleSplits(list(key))
                    for split in splits:
                        vals = [argList[len(v)].get(tuple(v), [None, None])[0] for v in split]
                        if None in vals or abs(sum(vals)) >= abs(args[key][0]) or not sum([abs(val) == val for val in vals]) in [0, len(vals)]:   # do all values in vals have the same sign
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
            indicesToKeep = []
            for val, index in argList[1].values() + argList[2].values() + argList[3].values() + argList[4].values():
                indicesToKeep.append(index)

            # we remove all arguments that are not in indicesToKeep
            indicesToKeep.sort()
            arguments = self.arguments[classValue]
            for j in range(len(arguments))[::-1]:
                if len(indicesToKeep) == 0 or j != indicesToKeep[-1]:
                    arguments.pop(j)
                else: indicesToKeep.pop()
            
    # compute probability that the combination of attribute values in attrValues is significantly different from being independent
    # see Kononenko: Semi-naive Bayes
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

    # select only those examples that have the specified attribute values
    # also remove examples that have missing values at those attributes
    def getDataSubset(self, attrList, attrValues):
        temp = self.data.selectref(dict([(attrList[i], attrValues[i]) for i in range(len(attrList))]))
        filter = orange.Filter_isDefined(domain=self.data.domain)
        for v in self.data.domain.variables:
            filter.check[v] = v.name in attrList
        return filter(temp)
    

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

        triedIndices = [0]*(len(attrs))
        maxVals = [len(val) for val in valueOrder]
        xpos = 0; ypos = 0
        while triedIndices[0] < maxVals[0]:
            vals = [valueOrder[i][triedIndices[i]] for i in range(len(attrs))]
            combVal = reduce(operator.concat, map(operator.concat, [vals[i] for i in revert], ["-"]*len(vals)))[:-1]
            if conditions[combVal][0] > 0:
                #projData.append([xpos, ypos, self.data.domain.classVar.values[quotients.index(max(quotients))]])
                projData.append([xpos, ypos, conditions[combVal][1]])
                projData[-1].setmeta("ExampleWeight", conditions[combVal][0]/max(1,len(self.data))) # set weight of the rectangle

            triedIndices[-1] += 1
            for i in range(1, len(attrs))[::-1]:
                if triedIndices[i] >= maxVals[i]:
                    triedIndices[i-1] += 1
                    triedIndices[i] = 0

            xpos = triedIndices[-1]
            ypos = len(attrs) > 1 and 2 * triedIndices[-2]
            if len(attrs) > 2: xpos += (len(attrs) + maxVals[-1]) * triedIndices[-3] # len(attrs) will add the factor 3 or 4
            if len(attrs) > 3: ypos += (4 + maxVals[-2]) * triedIndices[-4]
            
        learner = orange.kNNLearner(rankWeight = 0, k = len(projData)/2)
        results = orngTest.learnAndTestOnLearnData([learner], (projData, self.weightID))
        return orngStat.AP(results)[0]

    # for a given subset of attributes (max 4) find which permutation is most visual friendly. optimizeValueOrder
    def findOptimalAttributeOrder(self, attrs, optimizeValueOrder = 0):
        apriori = [max(1, self.aprioriDistribution[val]) for val in self.data.domain.classVar.values]
        conditions = {}
        newFeature, quality = FeatureByCartesianProduct(self.data, attrs)
        dist = orange.Distribution(newFeature, self.data)
        cont = orange.ContingencyAttrClass(newFeature, self.data)
        for key in cont.keys():
            if dist[key] == 0: conditions[key] = (0, 0)
            else:
                quotients = map(operator.div, cont[key], apriori)
                conditions[key] = (dist[key], self.data.domain.classVar.values[quotients.index(max(quotients))])
        
        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.data.domain.classVar])
        self.weightID = orange.newmetaid()
        domain.addmeta(self.weightID, orange.FloatVariable("ExampleWeight"))
        
        # create permutations of attributes and attribute values
        attrPerms = orngVisFuncts.permutations(range(len(attrs)))
        valuePerms = {}
        for attr in attrs:
            if optimizeValueOrder:  valuePerms[attr] = orngVisFuncts.permutations(getVariableValuesSorted(self.data, attr))
            else:                   valuePerms[attr] = [getVariableValuesSorted(self.data, attr)]

        if self.__class__.__name__ == "OWMosaicOptimization":
            self.setStatusBarText("Generating possible attribute orders...")
            self.parentWidget.progressBarInit()

        possibleOrders = []
        triedIndices = [0]*(len(attrs))
        maxVals = [len(valuePerms[attr]) for attr in attrs]
        while triedIndices[-1] < maxVals[-1]:    # we have no more possible placements if we have an overflow
            valueOrder = [valuePerms[attrs[i]][triedIndices[i]] for i in range(len(attrs))]
            possibleOrders.append(valueOrder)
            triedIndices[0] += 1
            if self.stopOptimization: break
            for i in range(len(attrs)-1):
                if triedIndices[i] >= maxVals[i]:
                    triedIndices[i+1] += 1
                    triedIndices[i] = 0

        bestPlacements = []
        current = 0
        total = len(attrPerms) * len(possibleOrders)
        strCount = orngVisFuncts.createStringFromNumber(total)
        for attrPerm in attrPerms:
            currAttrs = [attrs[i] for i in attrPerm]
            if self.stopOptimization: break
            tempPerms = []
            for order in possibleOrders:
                currValueOrder = [order[i] for i in attrPerm]
                val = self.evaluateAttributeOrder(currAttrs, currValueOrder, conditions, map(attrPerm.index, range(len(attrPerm))), domain)
                tempPerms.append((val, currAttrs, currValueOrder))
                current += 1
                if current % 10 == 0 and self.__class__.__name__ == "OWMosaicOptimization":
                    self.setStatusBarText("Evaluated %s/%s attribute orders..." % (orngVisFuncts.createStringFromNumber(current), strCount))
                    self.parentWidget.progressBarSet(100*current/float(total))
                    if self.stopOptimization: break
            bestPlacements.append(max(tempPerms))

        if self.__class__.__name__ == "OWMosaicOptimization":
            self.setStatusBarText("")
            self.parentWidget.progressBarFinished()
            
        if len(bestPlacements) == 0:
            return None, None, None

        bestPlacements.sort()
        bestPlacements.reverse()
        return bestPlacements
        
        

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



#test widget appearance
if __name__=="__main__":
    """
    data = orange.ExampleTable(r"C:\Development\Python23\Lib\site-packages\Orange\Datasets\breastcancer2004.tab")
    example = orange.ExampleTable(r"C:\Development\Python23\Lib\site-packages\Orange\Datasets\breastcancer2004-1 example 2.tab")
    data = orange.ExampleTable(r"C:\Development\Python23\Lib\site-packages\Orange\Datasets\UCI\zoo.tab")
    example = data
    mosaic = orngMosaic()
    mosaic.attributeCount= 3
    mosaic.setData(data)
    mosaic.probabilityEstimation = RELATIVE
    #mosaic.useOnlyRelevantInteractionsInArgumentation = 0
    mosaic.automaticallyRemoveWeakerArguments = 1
    mosaic.evaluateProjections()
    mosaic.findArguments(example[0])
    """
    data = orange.ExampleTable(r"E:\Development\Python23\Lib\site-packages\Orange\Datasets\UCI\wine.tab")
    mosaic = orngMosaic()
    mosaic.setData(data)
    ret = mosaic.findOptimalAttributeOrder(["A11", "A13", "A7"], 1) #optimizeValueOrder = 1
    print ret
