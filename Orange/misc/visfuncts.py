import orange, statc, random
from Orange.feature import scoring
from Orange.misc import progress_bar_milestones
import copy
from math import ceil
##import numpy
##from LinearAlgebra import *

_differentClassPermutationsDict = {}
_projectionListDict = {}


# ##########################################################################################
# MOSAIC OPTIMIZATION FUNCTION
# ##########################################################################################

_possibleSplits = {}
_possibleSplits[1] = [[[0]]]
_possibleSplits[2] = [[[0], [1]]]
_possibleSplits[3] = [[[0], [1,2]], [[0,1],[2]], [[0,2], [1]], [[0], [1], [2]]]
_possibleSplits[4] = [[[0], [1,2,3]], [[1], [0,2,3]], [[2], [0,1,3]], [[3], [0,1,2]],
    [[0,1], [2,3]], [[0,2], [1,3]], [[0,3], [1,2]], [[0,1,2], [3]], [[0,1,3], [2]], [[0,2,3], [1]], [[1,2,3], [0]],
    [[0], [1], [2,3]], [[0], [2], [1,3]], [[0], [3], [1,2]], [[1], [2], [0,3]], [[1], [3], [0,2]], [[2], [3], [0,1]], [[0], [1], [2], [3]]]

# get possible splits of attributes in attrs into subgroups , e.g.: ["a","b","c"] -> [[['a'], ['b', 'c']], [['a', 'b'], ['c']], [['a', 'c'], ['b']], [['a'], ['b'], ['c']]]
def getPossibleSplits(attrs):
    if len(attrs) > 4: return []
    retSplit = []
    for split in _possibleSplits[len(attrs)]:
        parts = []
        for arr in split:
            a = [attrs[i] for i in arr]
            parts.append(a)
        retSplit.append(parts)
    return retSplit

# ##########################################################################################
# MISC FUNCTIONS
# ##########################################################################################

# take a number and return a formated string, eg: 2341232 -> "2,341,232"
def createStringFromNumber(num):
    s = str(num)
    arr = range(len(s)-2)[:0:-3]
    for i in arr:
        s = s[:i] + "," + s[i:]
    return s


# factoriela
def fact(i):
    ret = 1
    for j in range(2, i+1): ret*= j
    return ret

# get a sublist of permutations of list. Return only those permutations that cannot be attained by rotating one other permutation.
# used in radviz where we have to try different placements of attributes
def generateDifferentPermutations(list):
    if len(list) == 0: return []
    temp = permutations(list[:-1])
    newPerms = {}
    for p in temp:
        if not newPerms.has_key(tuple(p[::-1])):
            newPerms[tuple(p)] = p
    lastVal = list[-1]
    return [p + [lastVal] for p in newPerms.values()]


# compute permutations of elements in list
def permutations(list):
    if list==[]:
        return [[]]
    else:
        return [ [list[i]] + p for i in range(len(list)) for p in permutations(list[:i] + list[i+1:])]

# return number of combinations where we select "select" from "total"
def combinationsCount(select, total):
    return fact(total)/ (fact(total-select)*fact(select))

# get the actual combinations of items
def combinations(items, count):
    if count > len(items): return []
    answer = []
    indices = range(count)
    if indices == []: return [[]]   # if count = 0 return empty list

    indices[-1] = indices[-1] - 1
    while 1:
        limit = len(items) - 1
        i = count - 1
        while i >= 0 and indices[i] == limit:
            i = i - 1
            limit = limit - 1
        if i < 0: break

        val = indices[i]
        for i in xrange( i, count ):
            val = val + 1
            indices[i] = val
        temp = []
        for i in indices:
            temp.append( items[i] )
        answer.append( temp )
    return answer

# ##########################################################################################
# VIZRANK FUNCTIONS
# ##########################################################################################

# randomly switch two elements in a list. repeat nrOfTimes.
def switchTwoElements(list, nrOfTimes = 1):
    if len(list) < 2: return list
    for i in range(nrOfTimes):
        index1 = random.randint(0, len(list)-1)
        index2 = random.randint(0, len(list)-1)
        if index1 == index2:
            index2 = (index1 + 1) % len(list)
        exVal = list[index1]
        list[index1] = list[index2]
        list[index2] = exVal
    return list

# give a list of elements (e.g. [1,2,3,4,5,6,7]) and number of groups/classes (e.g. 2) and the number of switches you want to perform
# we assume that the the first len(list)/groupCount elements belong to one group, the second to second group and so on. We only want
# to switch elements inside a group, not elements between groups
def switchTwoElementsInGroups(list, groupCount, nrOfTimes = 1):
    # find group boundaries
    count = len(list)
    boundaries = []
    index = 0
    while groupCount > 0:
        boundaries.append(int(ceil(count/float(groupCount))))
        groupCount -= 1
        count -= boundaries[-1]
    currSum = 0
    for i in range(len(boundaries)):
        boundaries[i] = boundaries[i] + currSum
        currSum = boundaries[i]

    parts = []
    boundaries = [0] + boundaries
    for i in range(len(boundaries)-1):
        parts.append(list[boundaries[i] : boundaries[i+1]])

    # take parts, and for each part switch a few elements
    retList = []
    while parts != []:
        count = int(ceil(nrOfTimes/float(len(parts))))
        nrOfTimes -= count
        part = parts.pop(random.randint(0, len(parts)-1))
        retList += switchTwoElements(part, count)

    return retList



# ##########################################################################################
# MEASURES FOR EVALUATING INTERESTINGNESS OF ATTRIBUTES
# ##########################################################################################

# class that measures correlation between continuous class value and continuous attribute
class MeasureCorrelation:
    def __init__(self):
        pass

    def __call__(self, attr, data):
        return self.MeasureAttribute_info(attr, data)

    def MeasureAttribute_info(self, attr, data):
        table = data.select([attr, data.domain.classVar])
        table = orange.Preprocessor_dropMissing(table)
        a1 = [table[k][0].value for k in range(len(table))]
        a2 = [table[k][1].value for k in range(len(table))]

        val, prob = statc.pearsonr(a1, a2)
        return val


# fisher discriminant implemented to be used as orange.MeasureAttribute
class MeasureFisherDiscriminant:
    def __init__(self):
        self.dataset = None
        self.attrInfo = {}
        self.stats = []

    def __call__(self, attr, data):
        return self.MeasureAttribute_info(attr, data)

    def MeasureAttribute_info(self, attr, data):
        # if basic statistics is not computed for this dataset -> compute it
        if not (self.stats and self.dataset == data):
            self.stats = {}
            self.dataset = data

            arr = [0] * len(data.domain.attributes)
            for val in data.domain.classVar.values:
                data2 = data.select({data.domain.classVar: val})
                bas = orange.DomainBasicAttrStat(data2)
                self.stats[val] = bas

            for i in range(len(self.stats.keys())):
                statI = self.stats[self.stats.keys()[i]]
                if len(statI) == 0: continue
                for j in range(i+1, len(self.stats.keys())):
                    statJ = self.stats[self.stats.keys()[j]]
                    if len(statJ) == 0: continue
                    for attribute in range(len(data.domain.attributes)):
                        if data.domain.attributes[attribute].varType != orange.VarTypes.Continuous: continue
                        bottom = (statI[attribute].n * statI[attribute].dev + statJ[attribute].n * statJ[attribute].dev)
                        if bottom == 0.0: bottom = 0.001
                        val = abs(statI[attribute].avg - statJ[attribute].avg) * (statI[attribute].n + statJ[attribute].n)/bottom
                        arr[attribute] += val

            # normalize values in arr so that the largest value will be 1 and others will be proportionally smaller
            largest = max(arr)
            if largest != 0:
                arr = [val/largest for val in arr]

            for i in range(len(data.domain.attributes)):
                self.attrInfo[data.domain.attributes[i].name] = arr[i]

        return self.attrInfo[data.domain[attr].name]


# signal to noise measure implemented as orange.MeasureAttribute
class S2NMeasure:
    def __init__(self):
        self.attrInfo = {}
        self.data = None

    def __call__(self, attr, data):
        # if the data changed clear the attribute values
        if data != self.data:
            self.attrInfo = {}
            self.data = data

        if self.attrInfo == {}:
            classVar = data.domain.classVar
            datas = [data.select({data.domain.classVar.name: [val]}) for val in data.domain.classVar.values]
            stats = [orange.DomainBasicAttrStat(d) for d in datas]
            cls = range(len(stats))
            clsCount = len(stats)
            for i in range(len(stats[0])):
                if stats[0][i] == None: continue
                temp = 0.0
                for j in cls:
                    for k in range(j+1, clsCount):
                        if (stats[j][i].dev + stats[k][i].dev) > 0:
                            temp += abs((stats[j][i].avg - stats[k][i].avg) / (stats[j][i].dev + stats[k][i].dev))
                self.attrInfo[data.domain.attributes[i].name] = temp

        if self.attrInfo.has_key(data.domain[attr].name):
            return self.attrInfo[data.domain[attr].name]
        else:
            return -1


# same class as above, just that we can use it to evaluate attribute for each class value
class S2NMeasureMix(S2NMeasure):
    def __init__(self):
        S2NMeasure.__init__(self)
        self.attrInfoMix = {}
        self.dataMix = None
        self.sortedAttrList = []

    def __call__(self, attribute, data):

        # if the data changed clear the attribute values
        if data != self.dataMix:
            self.attrInfoMix = {}
            self.attrInfo = {}
            self.dataMix = data

        if self.attrInfoMix == {}:
            attrs = range(len(data.domain.attributes))
            classVar = data.domain.classVar
            #shortData = data.select(attrs + [classVar])
            datas = [data.select({classVar.name: [val]}) for val in classVar.values]
            statistics = [orange.DomainBasicAttrStat(d) for d in datas]

            cls = []
            for classVarIndex, c in enumerate(classVar.values):   # for each class value compute how good is each attribute for discriminating this class value against all other
                attrValsList = []
                newData = mergeClassValues(data, c)
                for attrIndex in range(len(attrs)):
                    if data.domain[attrIndex].varType == orange.VarTypes.Discrete:      # ignore discrete attributes
                        continue
                    val = S2NMeasure.__call__(self, attrs[attrIndex], newData)
                    if statistics[0][attrIndex] == None:
                        attrValsList.append((0,attrs[attrIndex]))
                    else:
                        aves = [stat[attrIndex].avg for stat in statistics]
                        if max(aves) != aves[classVarIndex] :
                            val = -val
                        attrValsList.append((val, attrs[attrIndex]))
                attrValsList.sort()
                attrValsList = [element[1] for element in attrValsList]     # remove the value
                attrValsList.reverse()
                cls.append(attrValsList)

            attrPositionsDict = dict([(attr, []) for attr in cls[0]])
            for arr in cls:
                for i in range(len(arr)):
                    attrPositionsDict[arr[i]].append(i)

            numClasses = len(classVar.values)
            currPos = [0 for i in range(numClasses)]
            self.sortedAttrList = []
            ableToAdd = 1
            while ableToAdd:    # sometimes some attributes are duplicated. in such cases we will add only one instance of such attribute to the list
                ableToAdd = 0
                for i in range(numClasses):
                    pos = currPos[i]
                    while pos < len(cls[i]) and cls[i][pos] == None:
                        pos += 1
                    currPos[i] = pos + 1
                    if pos >= len(cls[i]):
                        continue
                    ableToAdd = 1

                    attr = cls[i][pos]
                    self.sortedAttrList.append(attr)
                    attrPositions = attrPositionsDict[attr]     # get indices in cls where attribute attr is placed
                    for j in range(numClasses):
                        cls[j][attrPositions[j]] = None

            count = len(self.sortedAttrList)
            for (i, attr) in enumerate(self.sortedAttrList):
                self.attrInfoMix[data.domain[attr].name] = count-i

        if self.attrInfoMix.has_key(data.domain[attribute].name):
            return self.attrInfoMix[data.domain[attribute].name]
        else:
            return -1

# ##########################################################################################
# ##########################################################################################

def mergeClassValues(data, value):
    selection = orange.EnumVariable("Selection", values = ["0", "1"])

    selectedClassesStr = [value]
    nonSelectedClassesStr = []
    for val in data.domain.classVar.values:
        if val not in selectedClassesStr: nonSelectedClassesStr.append(val)

    shortData1 = data.select({data.domain.classVar.name: selectedClassesStr})
    shortData2 = data.select({data.domain.classVar.name: nonSelectedClassesStr})
    d1 = orange.Domain(shortData1.domain.attributes + [selection])
    selection.getValueFrom = lambda ex, what: orange.Value(selection, "0")
    data1 = orange.ExampleTable(d1, shortData1)

    selection.getValueFrom = lambda ex, what: orange.Value(selection, "1")
    data2 = orange.ExampleTable(d1, shortData2)
    data1.extend(data2)
    return data1


# almost equal to evaluateAttributesByEachClassValue function with one exception. if the class value that we want to discriminate
# has a lower average value than the other average class values then we multiply quality of this attribute with -1
# this way we get the attributes that have the highes expression and are also good at discrimination
def findAttributeGroupsForRadviz(data, measure):
    if isinstance(measure, S2NMeasureMix):
        measure(data.domain.attributes[0].name, data)                       # just call measure to compute quality of all attributes
        attrNames = [data.domain[ind].name for ind in measure.sortedAttrList]
        numClasses = len(data.domain.classVar.values)
        cls = [attrNames[i::numClasses] for i in range(numClasses)]
    else:
        attrVals = [(measure(attr.name, data), attr.name) for attr in data.domain.attributes]
        attrVals.sort()
        attrVals.reverse()
        attrNames = [attrVals[i][1] for i in range(len(attrVals))]  # remove quality values of the attributes
        cls = None

    return attrNames, cls


# for each class value test how good does the measure discriminate between this class value and all the other values merged
# return a list of lists. Each list contains tuples (val, attr), where val is the attribute quality for attr at separating one class value from the others
def evaluateAttributesByEachClassValue(data, measure, attrs):
    cls = []
    for c in data.domain.classVar.values:   # for each class value compute how good is each attribute for discriminating this class value against all other
        v = []
        data2 = mergeClassValues(data, c)
        for attr in attrs:
            v.append((measure(attr, data2), attr))
        v.sort()
        cls.append(v)
    return cls

# used by VizRank to evaluate attributes
def evaluateAttributesDiscClass(data, contMeasure, discMeasure):
    attrs = []
    #corr = MeasureCorrelation()
    for attr in data.domain.attributes:
        #if data.domain.classVar.varType == orange.VarTypes.Continuous and attr.varType == orange.VarTypes.Continuous: attrs.append((corr(attr.name, data), attr.name))
        if data.domain.classVar.varType == orange.VarTypes.Continuous and attr.varType == orange.VarTypes.Continuous: attrs.append((1, attr.name))
        elif data.domain.classVar.varType == orange.VarTypes.Continuous:            attrs.append((0.0, attr.name))
        elif discMeasure == None and attr.varType == orange.VarTypes.Discrete:      attrs.append((0.0, attr.name))
        elif contMeasure == None and attr.varType == orange.VarTypes.Continuous:    attrs.append((0.0, attr.name))
        elif attr.varType == orange.VarTypes.Continuous:                            attrs.append((contMeasure(attr.name, data), attr.name))
        else:                                                                       attrs.append((discMeasure(attr.name, data), attr.name))

    if discMeasure or contMeasure:
        attrs.sort()
        attrs.reverse()

    return [attr for (val, attr) in attrs]  # return only the ordered list of attributes and not also their values

def evaluateAttributesContClass(data, contMeasure, discMeasure):
    attrs = []
    for attr in data.domain.attributes:
        attrs.append((1, attr.name))
    #        elif discMeasure == None and attr.varType == orange.VarTypes.Discrete:      attrs.append((0.0, attr.name))
    #        elif contMeasure == None and attr.varType == orange.VarTypes.Continuous:    attrs.append((0.0, attr.name))
    #        elif attr.varType == orange.VarTypes.Continuous:                            attrs.append((contMeasure(attr.name, data), attr.name))
    #        else:                                                                       attrs.append((discMeasure(attr.name, data), attr.name))

    if discMeasure or contMeasure:
        attrs.sort()
        attrs.reverse()

    return [attr for (val, attr) in attrs]  # return only the ordered list of attributes and not also their values


def evaluateAttributesNoClass(data, contMeasure, discMeasure):
    attrs = []
    for attr in data.domain.attributes:
        attrs.append((1, attr.name))
    #        elif discMeasure == None and attr.varType == orange.VarTypes.Discrete:      attrs.append((0.0, attr.name))
    #        elif contMeasure == None and attr.varType == orange.VarTypes.Continuous:    attrs.append((0.0, attr.name))
    #        elif attr.varType == orange.VarTypes.Continuous:                            attrs.append((contMeasure(attr.name, data), attr.name))
    #        else:                                                                       attrs.append((discMeasure(attr.name, data), attr.name))

    if discMeasure or contMeasure:
        attrs.sort()
        attrs.reverse()

    return [attr for (val, attr) in attrs]  # return only the ordered list of attributes and not also their values




# ##############################################################################################
# ##############################################################################################


# SELECT ATTRIBUTES ##########################
def selectAttributes(data, attrContOrder, attrDiscOrder, projections = None):
    if data.domain.classVar == None or data.domain.classVar.varType != orange.VarTypes.Discrete:
        return ([attr.name for attr in data.domain.attributes], [], 0)

    shown = [data.domain.classVar.name]; hidden = []; maxIndex = 0    # initialize outputs

    # # both are RELIEF
    if attrContOrder == "ReliefF" and attrDiscOrder == "ReliefF":
        attrVals = scoring.score_all(data, orange.MeasureAttribute_relief())
        s,h = getTopAttrs(attrVals, 0.95)
        return (shown + s, hidden + h, 0)

    # # both are NONE
    elif attrContOrder == "None" and attrDiscOrder == "None":
        for item in data.domain.attributes:    shown.append(item.name)
        return (shown, hidden, 0)


    # disc and cont attribute list
    discAttrs = []; contAttrs = []
    for attr in data.domain.attributes:
        if attr.varType == orange.VarTypes.Continuous: contAttrs.append(attr.name)
        elif attr.varType == orange.VarTypes.Discrete: discAttrs.append(attr.name)


    ###############################
    # sort continuous attributes
    if attrContOrder == "None":
        shown += contAttrs
    elif attrContOrder in ["ReliefF", "Fisher discriminant", "Signal to Noise", "Signal to Noise For Each Class"]:
        if attrContOrder == "ReliefF":               measure = orange.MeasureAttribute_relief(k=10, m=50)
        elif attrContOrder == "Fisher discriminant": measure = MeasureFisherDiscriminant()
        elif attrContOrder == "Signal to Noise":     measure = S2NMeasure()
        else:                                        measure = S2NMeasureMix()

        dataNew = data.select(contAttrs + [data.domain.classVar])
        attrVals = scoring.score_all(dataNew, measure)
        s,h = getTopAttrs(attrVals, 0.95)
        shown += s
        hidden += h
    else:
        print "Unknown value for attribute order: ", attrContOrder

    # ###############################
    # sort discrete attributes
    if attrDiscOrder == "None":
        shown += discAttrs
    elif attrDiscOrder == "GainRatio" or attrDiscOrder == "Gini" or attrDiscOrder == "ReliefF":
        if attrDiscOrder == "GainRatio":   measure = orange.MeasureAttribute_gainRatio()
        elif attrDiscOrder == "Gini":       measure = orange.MeasureAttribute_gini()
        else:                               measure = orange.MeasureAttribute_relief()

        dataNew = data.select(discAttrs + [data.domain.classVar])
        attrVals = scoring.score_all(dataNew, measure)
        s,h = getTopAttrs(attrVals, 0.95)
        shown += s; hidden += h

    elif attrDiscOrder == "Oblivious decision graphs":
        #shown.append(data.domain.classVar.name)
        attrs = getFunctionalList(data)
        for item in attrs:
            shown.append(item)
        for attr in data.domain.attributes:
            if attr.name not in shown and attr.varType == orange.VarTypes.Discrete:
                hidden.append(attr.name)
    else:
        print "Unknown value for attribute order: ", attrDiscOrder

    return (shown, hidden, maxIndex)


def getTopAttrs(results, maxSum = 0.95, onlyPositive = 1):
    s = []; h = []
    sum = 0
    for (attr, val) in results:
        if not onlyPositive or val > 0: sum += val
    tempSum = 0
    for (attr, val) in results:
        if tempSum < maxSum*sum: s.append(attr)
        else: h.append(attr)
        tempSum += val
    return (s, h)


# ##########################################################################################
# PARALLEL COORDINATES FUNCTIONS
# ##########################################################################################

# find interesting attribute order for parallel coordinates
# attrInfo = [(val1, attr1, attr2), .... ]
def optimizeAttributeOrder(attrInfo, numberOfAttributes, optimizationDlg, app):
    while (attrInfo != []):
        proj = []
        projVal = []
        canAddAttribute = 1
        while canAddAttribute:
            if not optimizationDlg.canContinueOptimization(): return
            app.processEvents()        # allow processing of other events

            if len(proj) == 0:
                proj = [attrInfo[0][1], attrInfo[0][2]]
                projVal = [attrInfo[0][0]]
            elif len(proj) == numberOfAttributes:
                canAddAttribute = 0     # time to add the projection to the list
            else:
                proj, projVal, success = addBestToCurrentProj(proj, projVal, attrInfo)
                if not success: canAddAttribute = 0 # there are no more attributes that can be added to this projection

        if len(proj) == numberOfAttributes:
            # I (Ales) commented this out because fixItersectingPairs can enter an endless cycle 
            #proj, projVal = fixIntersectingPairs(proj, projVal, attrInfo)
            for i in range(len(proj)-1):
                removeAttributePair(proj[i], proj[i+1], attrInfo)
            optimizationDlg.addProjection(sum(projVal)/len(projVal), proj)
        else:
            for i in range(len(proj)-1):
                removeAttributePair(proj[i], proj[i+1], attrInfo)


# try rotating subsequences of proj to increase value of attribute order
def fixIntersectingPairs(proj, projVal, attrInfo):
    changed = 1
    allprojections = set()
    while changed:
        changed = 0
        for i in range(len(projVal)-1):
            if changed: continue
            for j in range(i+2, len(projVal)-1):
                if changed: continue
                val1, exists1 = getAttributePairValue(proj[i], proj[j], attrInfo)
                val2, exists2 = getAttributePairValue(proj[i+1], proj[j+1], attrInfo)
                if exists1 and exists2 and (val1 + val2 > projVal[i] + projVal[j]):
                    projVal[i] = val1
                    projVal[j] = val2
                    rev = proj[i+1:j+1]
                    rev.reverse()
                    tempProj = proj[:i+1] + rev + proj[j+1:]
                    proj = tempProj
                    changed = 1     # we rotated the projection. start checking from the begining

        tproj = tuple(proj)
        assert(tproj not in allprojections)
        allprojections.add(tproj)

    return proj, projVal

# return value for attribute pair (val, attr1, attr2) if exists. if not, return 0
def getAttributePairValue(attr1, attr2, attrInfo):
    for (val, a1, a2) in attrInfo:
        if (attr1 == a1 and attr2 == a2) or (attr1 == a2 and attr2 == a1): return (val, 1)
    return (0, 0)

# remove attribute pair (val, attr1, attr2) from attrInfo
def removeAttributePair(attr1, attr2, attrInfo):
    for (val, a1, a2) in attrInfo:
        if (attr1 == a1 and attr2 == a2) or (attr1 == a2 and attr2 == a1):
            attrInfo.remove((val, a1, a2))
            return
    print "failed to remove attribute pair", attr1, attr2


def addBestToCurrentProj(proj, projVal, attrInfo):
    for (val, a1, a2) in attrInfo:
        if (a1 == proj[0] and a2 not in proj) or (a2 == proj[0] and a1 not in proj) or (a1 == proj[-1] and a2 not in proj) or (a2 == proj[-1] and a1 not in proj):
            if a1 == proj[0]: return ([a2] + proj, [val] + projVal, 1)
            elif a2 == proj[0]: return ([a1] + proj, [val] + projVal, 1)
            elif a1 == proj[-1]: return (proj + [a2], projVal + [val], 1)
            else                       : return (proj + [a1], projVal + [val], 1)

    """
    for (val, a1, a2) in attrInfo:
        if a1 in proj and a2 in proj: continue
        if a1 not in proj and a2 not in proj: continue
        
        if (a1 not in proj) and (a2 in proj): placed = a2; place = a1
        else:                                  placed = a1; place = a2
            
        ind = proj.index(placed)
        if ind > 0:
            val2, exists = getAttributePairValue(place, proj[ind-1], attrInfo)
            if exists:
                proj.insert(ind, place)
                projVal[ind-1] = val2
                projVal.insert(ind-1, val)
                return (proj, projVal, 1)
        if ind < len(proj)-1:
            val2, exists = getAttributePairValue(place, proj[ind+1], attrInfo)
            if exists:
                proj.insert(ind+1, place)
                projVal[ind] = val
                projVal.insert(ind, val2)
                return (proj, projVal, 1)
    """
    return proj, projVal, 0

def computeCorrelationBetweenAttributes(data, attrList, minCorrelation = 0.0, progressCallback=None):
    correlations = []
    attrListLen = len(attrList)
    iterCount = attrListLen * (attrListLen - 1) / 2
    iter = 0
    milestones = progress_bar_milestones(iterCount)
    for i in range(len(attrList)):
        if data.domain.attributes[i].varType != orange.VarTypes.Continuous:
            continue
        for j in range(i+1, len(attrList)):
            if data.domain.attributes[j].varType != orange.VarTypes.Continuous:
                continue
            val = abs(computeCorrelation(data, attrList[i], attrList[j]))
            if val >= minCorrelation:
                correlations.append((val, attrList[i], attrList[j]))
            iter += 1
            if progressCallback and iter in milestones:
                progressCallback(100.0 * iter / iterCount)

    return sorted(correlations, reverse=True)


def computeCorrelationInsideClassesBetweenAttributes(data, attrList, minCorrelation = 0.0, progressCallback=None):
    if not data.domain.classVar or data.domain.classVar.varType == orange.VarTypes.Continuous:
        return []
    correlations = []
    attrListLen = len(attrList)
    iterCount = attrListLen * (attrListLen - 1) / 2
    iter = 0
    milestones = progress_bar_milestones(iterCount)
    for i in range(len(attrList)):
        if data.domain.attributes[i].varType != orange.VarTypes.Continuous:
            continue
        for j in range(i+1, len(attrList)):
            if data.domain.attributes[j].varType != orange.VarTypes.Continuous:
                continue
            corr, corrs, lengths = computeCorrelationInsideClasses(data, attrList[i], attrList[j])
            if corr >= minCorrelation:
                correlations.append((corr, attrList[i], attrList[j]))
            iter += 1
            if progressCallback and iter in milestones:
                progressCallback(100.0 * iter / iterCount)

    return sorted(correlations, reverse=True)


def computeCorrelationInsideClasses(data, attr1, attr2):
    if data.domain[attr1].varType != orange.VarTypes.Continuous: return None
    if data.domain[attr2].varType != orange.VarTypes.Continuous: return None

    table = data.select([attr1, attr2, data.domain.classVar])
    table = orange.Preprocessor_dropMissing(table)
    lengths = []; corrs = []
    for val in table.domain.classVar.values:
        tab = table.filter({table.domain.classVar: val})
        a1 = [tab[k][attr1].value for k in range(len(tab))]
        a2 = [tab[k][attr2].value for k in range(len(tab))]
        if len(a1) == 0: continue
        val, prob = statc.pearsonr(a1, a2)
        lengths.append(len(a1))
        corrs.append(val)
    corr = 0
    for ind in range(len(corrs)): corr += abs(corrs[ind])*lengths[ind]
    corr /= sum(lengths)
    return corr, corrs, lengths

# compute correlation between two continuous attributes
def computeCorrelation(data, attr1, attr2):
    if data.domain[attr1].varType != orange.VarTypes.Continuous: return None
    if data.domain[attr2].varType != orange.VarTypes.Continuous: return None

    table = data.select([attr1, attr2])
    table = orange.Preprocessor_dropMissing(table)
    a1 = [table[k][attr1].value for k in range(len(table))]
    a2 = [table[k][attr2].value for k in range(len(table))]

    try:
        val, prob = statc.pearsonr(a1, a2)
    except:
        val = 0.0    # possibly invalid a1 or a2

    return val

# ##########################################################################################
# ##########################################################################################


# ##########################################################################################
# #### FUNCTIONS FOR CALCULATING ATTRIBUTE ORDER USING Oblivious decision graphs
# ##########################################################################################
def replaceAttributes(index1, index2, merged, data):
    attrs = list(data.domain)
    attrs.remove(data.domain[index1])
    attrs.remove(data.domain[index2])
    domain = orange.Domain(attrs+ [merged])
    return data.select(domain)


def getFunctionalList(data):
    import orngCI

    bestQual = -10000000
    bestAttr = -1
    testAttrs = []

    dataShort = orange.Preprocessor_dropMissing(data)
    # remove continuous attributes from data
    disc = []
    for i in range(len(dataShort.domain.attributes)):
        # keep only discrete attributes that have more than one value
        if dataShort.domain.attributes[i].varType == orange.VarTypes.Discrete and len(dataShort.domain.attributes[i].values) > 1: disc.append(dataShort.domain.attributes[i].name)
    if disc == []: return []
    discData = dataShort.select(disc + [dataShort.domain.classVar.name])

    remover = orngCI.AttributeRedundanciesRemover(noMinimization = 1)
    newData = remover(discData, weight = 0)

    for attr in newData.domain.attributes: testAttrs.append(attr.name)

    # compute the best attribute combination
    for i in range(len(newData.domain.attributes)):
        vals, qual = orngCI.FeatureByMinComplexity(newData, [newData.domain.attributes[i], newData.domain.classVar])
        if qual > bestQual:
            bestQual = qual
            bestAttr = newData.domain.attributes[i].name
            mergedVals = vals
            mergedVals.name = newData.domain.classVar.name

    if bestAttr == -1: return []
    outList = [bestAttr]
    newData = replaceAttributes(bestAttr, newData.domain.classVar, mergedVals, newData)
    testAttrs.remove(bestAttr)

    while (testAttrs != []):
        bestQual = -10000000
        for attrName in testAttrs:
            vals, qual = orngCI.FeatureByMinComplexity(newData, [mergedVals, attrName])
            if qual > bestQual:
                bestqual = qual
                bestAttr = attrName

        vals, qual = orngCI.FeatureByMinComplexity(newData, [mergedVals, bestAttr])
        mergedVals = vals
        mergedVals.name = newData.domain.classVar.name
        newData = replaceAttributes(bestAttr, newData.domain.classVar, mergedVals, newData)
        outList.append(bestAttr)
        testAttrs.remove(bestAttr)

    # new attributes have "'" at the end of their names. we have to remove that in ored to identify them in the old domain
    for index in range(len(outList)):
        if outList[index][-1] == "'": outList[index] = outList[index][:-1]
    return outList


# ##########################################################################################
# POLYVIZ FUNCTIONS USED IN VIZRANK.
# ##########################################################################################

# input: array where items are arrays, e.g.: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
# returns: array, where some items are removed. it rotates items in each array and checks if the item is already in the array. it also tries to flip the items ([::-1])
# for the above case, it would return [[1,2,3]]
# used for radviz
def removeRotationDuplicates(arr, removeFlipDuplicates = 1):
    final = []
    projDict = {}
    for a in arr:
        if type(a[0]) == int:
            key = tuple([a[i]-a[i+1] for i in range(-1, len(a)-1)])
            rkey = tuple([a[i+1]-a[i] for i in range(-1, len(a)-1)])
            o = min(a)      # offset. needed in cases when we get in arr: [0,1,2] and [1,2,3]. otherwise it would remove the second element because it would look the same as the first
        else:
            key = tuple([(a[i][0]-a[i+1][0], a[i][1]-a[i+1][1]) for i in range(-1, len(a)-1)])
            rkey = tuple([(a[i+1][0]-a[i][0], a[i+1][1]-a[i][1]) for i in range(-1, len(a)-1)])
            o = 0

        rotations = [key[i:] + key[:i] for i in range(len(key))]
        if 1 in [projDict.has_key((o,key)) for key in rotations]:
            continue

        rotations = [rkey[i:] + rkey[:i] for i in range(len(rkey))]
        if removeFlipDuplicates and 1 in [projDict.has_key((o,key)) for key in rotations]:
            continue

        projDict[(o,key)] = a

        """
        found = 0
        kkey = copy.copy(key)
        
        for i in range(len(kkey)):
            kkey = tuple(list(kkey[1:]) + [kkey[0]])
            if projDict.has_key((o,kkey)):
                found = 1
                break

        # try also if there is a flipped duplicate
        if not found and removeFlipDuplicates:
            for i in range(len(rkey)):
                rkey = tuple(list(rkey[1:]) + [rkey[0]])
                if projDict.has_key((o,rkey)):
                    found = 1
                    break
        if not found: projDict[(o,key)] = a
        """
    return projDict.values()

# create possible combinations with the given set of numbers in arr
def createMixCombinations(arrs, removeFlipDuplicates):

    def addProjs(projs, count, i):
        ret = []
        perms = permutations(range(count))
        for perm in perms:
            c = copy.copy(projs)
            add = [(i, p) for p in perm]
            c = [p + add for p in c]
            ret += c
        return ret

    projs = [[]]
    for i in range(len(arrs)):
        projs = addProjs(projs, arrs[i], i)
    return removeRotationDuplicates(projs, removeFlipDuplicates)


# get a list of possible projections if we have numClasses and want to use maxProjLen attributes
# removeFlipDuplicates tries to flip the attributes in the projection and removes it if the projection already exists
# removeFlipDuplicates = 1 for radviz and =0 for polyviz
def createProjections(numClasses, maxProjLen, removeFlipDuplicates = 1):
    if _projectionListDict.has_key((numClasses, maxProjLen, removeFlipDuplicates)):
        return _projectionListDict[(numClasses, maxProjLen, removeFlipDuplicates)]

    # create array of arrays of lengths, e.g. [3,3,2] that will tell that we want 3 attrs from the 1st class, 3 from 2nd and 2 from 3rd
    if maxProjLen % numClasses != 0:
        cs = combinations(range(numClasses), maxProjLen % numClasses)
        equal = [int(maxProjLen / numClasses) for i in range(numClasses)]
        lens = [copy.copy(equal) for comb in cs]     # make array of arrays
        for i, comb in enumerate(cs):
            for val in comb: lens[i][val] += 1
    else:
        lens = [[int(maxProjLen / numClasses) for i in range(numClasses)]]

    combs = []
    seen = []
    for l in lens:
        withoutZeros = filter(None, l)          # if numClasses > maxProjLen then some values in l are 0. we have to remove this zeros and check if we have already added such combinations.
        if withoutZeros not in seen:
            tempCombs = createMixCombinations(withoutZeros, removeFlipDuplicates)
            combs += tempCombs
            seen.append(withoutZeros)

    if _differentClassPermutationsDict.has_key((numClasses, maxProjLen, removeFlipDuplicates)):
        perms = _differentClassPermutationsDict[(numClasses, maxProjLen, removeFlipDuplicates)]
    else:
        if numClasses <= maxProjLen: perms = permutations(range(numClasses))
        else:                        perms = combinations(range(numClasses), maxProjLen)
        perms = removeRotationDuplicates(perms, removeFlipDuplicates)
        _differentClassPermutationsDict[(numClasses, maxProjLen, removeFlipDuplicates)] = perms

    final = []
    for perm in perms:
        final += [[(perm[i], j) for (i,j) in comb] for comb in combs]

    _projectionListDict[(numClasses, maxProjLen, removeFlipDuplicates)] = final
    return final


if __name__=="__main__":
    """
    print "possible splits of ['a','b','c'] are ", getPossibleSplits(["a","b","c"])
    print "possible combinations of 2 elements of the array [1,2,3,4] are ", combinations([1,2,3,4],2)
    print "permutations of [1,2,3] are ", [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    print "array with two randomly switched elements of [1,2,3,4,5] is", switchTwoElements([1,2,3,4,5])

    data = orange.ExampleTable(r"E:\Development\Python23\Lib\site-packages\Orange\Datasets\microarray\cancer diagnostics\leukemia_tran.tab")
    a = data.toNumpy("ac")[0]
    c = S2NMeasure()
    c(data.domain.attributes[0].name, data)
    """
    final = createProjections(8,4)
    print final