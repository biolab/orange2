import orange
import orngFSS
import statc
from Numeric import *
from LinearAlgebra import *

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
# #### FUNCTIONS FOR CALCULATING ATTRIBUTE ORDER USING Fisher discriminant analysis
# ##########################################################################################

# #################################################################################
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
        

# #################################################################################
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
            arr = [val/largest for val in arr]

            for i in range(len(data.domain.attributes)):
                self.attrInfo[data.domain.attributes[i].name] = arr[i]

        return self.attrInfo[data.domain[attr].name]

# #################################################################################
# signal to noise measure implemented as orange.MeasureAttribute
# SLOW SLOW SLOW !!! in comparison to S2NMeasure
class S2NMeasure2:
    def __init__(self):
        self.attrInfo = {}
        self.data = None

    def __call__(self, attr, data):
        # if the data changed clear the attribute values
        if data != self.data:
            self.attrInfo = {}
            self.data = data

        if self.attrInfo == {}:
            arr = data.toNumeric("ac", 0, 1, 1e20)[0]
            arr = Numeric.transpose(arr)
            clsIndex = len(arr)-1
            cls = [Numeric.where(arr[clsIndex]== i, 1, 0) for i in range(len(data.domain.classVar.values))]

            for i in range(len(data.domain.attributes)):
                mis = []; stds = []; ts = []
                for a in cls:
                    #sel = Numeric.compress(a, arr[i]).tolist()
                    sel = Numeric.compress(a, arr[i])
                    sel = Numeric.compress(Numeric.where(sel != 1e20, 1, 0), sel).tolist()   # remove values 1e20, which represent the missing values
                    mis.append(statc.mean(sel))
                    stds.append(statc.std(sel))

                for a in range(len(data.domain.classVar.values)):
                    for b in range(a+1, len(data.domain.classVar.values)):
                        t = (mis[a] - mis[b]) / (stds[a] + stds[b])
                        ts.append(abs(t))
                
                self.attrInfo[data.domain.attributes[i].name] = sum(ts)/float(len(data.domain.classVar.values))

        return self.attrInfo[data.domain[attr].name]

# #################################################################################
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

import time

# same class as above, just that we can use it to evaluate attribute for each class value
class S2NMeasureMix(S2NMeasure):
    def __init__(self):
        S2NMeasure.__init__(self)
        self.attrInfoMix = {}
        self.dataMix = None
        self.sortedAttrList = []
        
    def __call__(self, attribute, data):
        if data.domain[attribute].varType == orange.VarTypes.Discrete:
            print "S2NMeasureMix can not evaluate discrete attributes"
            return -1
        
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
                    val = S2NMeasure.__call__(self, attrs[attrIndex], newData)
                    if statistics[0][attrIndex] == None:
                        attrValsList.append((0,attrs[attrIndex]))
                    else:
                        aves = [stat[attrIndex].avg for stat in statistics]
                        if max(aves) != aves[classVarIndex] : val = -val
                        attrValsList.append((val, attrs[attrIndex]))
                attrValsList.sort()
                attrValsList = [element[1] for element in attrValsList]     # remove the value
                attrValsList.reverse()
                cls.append(attrValsList)

            attrPositionsDict = dict([(attr, []) for attr in cls[0]])
            for arr in cls:
                for i in range(len(arr)):
                    attrPositionsDict[arr[i]].append(i)

            ableToAdd = 1
            numClasses = len(classVar.values)
            currPos = [0 for i in range(numClasses)]
            self.sortedAttrList = []
            while ableToAdd:    # sometimes some attributes are duplicated. in such cases we will add only one instance of such attribute to the list
                ableToAdd = 0
                for i in range(numClasses):
                    pos = currPos[i]
                    while pos < len(cls[i]) and cls[i][pos] == None: pos += 1
                    currPos[i] = pos + 1
                    if pos >= len(cls[i]): continue
                    ableToAdd = 1
                    
                    attr = cls[i][pos]
                    self.sortedAttrList.append(attr)
                    attrPositions = attrPositionsDict[attr]     # get indices in cls where attribute attr is placed
                    for j in range(numClasses): cls[j][attrPositions[j]] = None
                    
                    
            count = len(self.sortedAttrList)
            for (i, attr) in enumerate(self.sortedAttrList):
                self.attrInfoMix[data.domain[attr].name] = count-i

        if self.attrInfoMix.has_key(data.domain[attribute].name):
            return self.attrInfoMix[data.domain[attribute].name]
        else:
            return -1
    

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

# used by kNN optimization to evaluate attributes
def evaluateAttributes(data, contMeasure, discMeasure):
    attrs = []
    corr = MeasureCorrelation()
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
        

    
# ##############################################################################################
# ##############################################################################################



# #############################################
# SELECT ATTRIBUTES ##########################
# #############################################
def selectAttributes(data, attrContOrder, attrDiscOrder, projections = None):
    if data.domain.classVar == None or data.domain.classVar.varType != orange.VarTypes.Discrete:
        return ([attr.name for attr in data.domain.attributes], [], 0)

    shown = [data.domain.classVar.name]; hidden = []; maxIndex = 0    # initialize outputs

    # # both are RELIEF
    if attrContOrder == "ReliefF" and attrDiscOrder == "ReliefF":
        attrVals = orngFSS.attMeasure(data, orange.MeasureAttribute_relief())
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
        attrVals = orngFSS.attMeasure(dataNew, measure)
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
        attrVals = orngFSS.attMeasure(dataNew, measure)
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
    
    val, prob = statc.pearsonr(a1, a2)
    return val

def computeCorrelationBetweenAttributes(data, attrList, minCorrelation = 0.0):
    correlations = []
    for i in range(len(attrList)):
        if data.domain.attributes[i].varType != orange.VarTypes.Continuous: continue
        for j in range(i+1, len(attrList)):
            if data.domain.attributes[j].varType != orange.VarTypes.Continuous: continue
            val = abs(computeCorrelation(data, attrList[i], attrList[j]))
            if val >= minCorrelation: correlations.append((val, attrList[i], attrList[j]))

    correlations.sort()
    correlations.reverse()
    return correlations


def computeCorrelationInsideClassesBetweenAttributes(data, attrList, minCorrelation = 0.0):
    if not data.domain.classVar or data.domain.classVar.varType == orange.VarTypes.Continuous: return []
    correlations = []
    for i in range(len(attrList)):
        if data.domain.attributes[i].varType != orange.VarTypes.Continuous: continue
        for j in range(i+1, len(attrList)):
            if data.domain.attributes[j].varType != orange.VarTypes.Continuous: continue
            corr, corrs, lengths = computeCorrelationInsideClasses(data, attrList[i], attrList[j])
            if corr >= minCorrelation: correlations.append((corr, attrList[i], attrList[j]))

    correlations.sort()
    correlations.reverse()
    return correlations


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
            proj, projVal = fixIntersectingPairs(proj, projVal, attrInfo)
            for i in range(len(proj)-1):
                removeAttributePair(proj[i], proj[i+1], attrInfo)
            optimizationDlg.addProjection(sum(projVal)/len(projVal), proj)
        else:
            for i in range(len(proj)-1):
                removeAttributePair(proj[i], proj[i+1], attrInfo)


# try rotating subsequences of proj to increase value of attribute order
def fixIntersectingPairs(proj, projVal, attrInfo):
    changed = 1
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

# ##########################################################################################
# ##########################################################################################

if __name__=="__main__":
    pass
    #import orange
    #data = orange.ExampleTable(r"E:\Development\Python23\Lib\site-packages\Orange\Datasets\microarray\cancer diagnostics\leukemia_tran.tab")
    #a = data.toNumeric("ac")[0]
    #c = S2NMeasure()
    #c(data.domain.attributes[0].name, data)
