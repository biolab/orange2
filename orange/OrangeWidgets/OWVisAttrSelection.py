
import orange
import orngFSS
import statc
import orngCI
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
                for j in range(i+1, len(self.stats.keys())):
                    statI = self.stats[self.stats.keys()[i]]
                    statJ = self.stats[self.stats.keys()[j]]
                    for attribute in range(len(data.domain.attributes)):
                        if data.domain.attributes[attribute].varType != orange.VarTypes.Continuous: continue
                        val = abs(statI[attribute].avg - statJ[attribute].avg) * (statI[attribute].n + statJ[attribute].n)/(statI[attribute].n * statI[attribute].dev + statJ[attribute].n * statJ[attribute].dev)
                        #val = abs(statI[attribute].avg - statJ[attribute].avg)/(statI[attribute].dev + statJ[attribute].dev)
                        arr[attribute] += val

            # normalize values in arr so that the largest value will be 1 and others will be proportionally smaller
            largest = max(arr)
            arr = [val/largest for val in arr]

            for i in range(len(data.domain.attributes)):
                self.attrInfo[data.domain.attributes[i].name] = arr[i]

        return self.attrInfo[data.domain[attr].name]


# used by kNN optimization to evaluate attributes
def evaluateAttributes(data, contMeasure, discMeasure):
    attrs = []
    for attr in data.domain.attributes:
        if   discMeasure == None and attr.varType == orange.VarTypes.Discrete:   attrs.append((0.1, attr.name))
        elif contMeasure == None and attr.varType == orange.VarTypes.Continuous: attrs.append((0.1, attr.name))
        elif attr.varType == orange.VarTypes.Continuous: attrs.append((contMeasure(attr.name, data), attr.name))
        else:                                              attrs.append((discMeasure(attr.name, data), attr.name))
    return attrs
        

    
# ##############################################################################################
# ##############################################################################################



# #############################################
# SELECT ATTRIBUTES ##########################
# #############################################
def selectAttributes(data, graph, attrContOrder, attrDiscOrder, projections = None):
    if data.domain.classVar.varType != orange.VarTypes.Discrete:
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
    elif attrContOrder == "ReliefF" or attrContOrder == "Fisher discriminant":
        if attrContOrder == "ReliefF":   measure = orange.MeasureAttribute_relief()
        else:                               measure = MeasureFisherDiscriminant()

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
            shown.append(data.domain.classVar.name)
            attrs = getFunctionalList(data)
            for item in attrs:
                shown.append(item)
            for attr in data.domain.attributes:
                if attr.name not in shown and attr.varType == orange.VarTypes.Discrete:
                    hidden.append(attr.name)
    else:
        print "Unknown value for attribute order: ", attrDiscOrder

    return (shown, hidden, maxIndex)


def computeCorrelationBetweenAttributes(data, attrList, minCorrelation = 0.0):
    correlations = []
    for i in range(len(attrList)):
        if data.domain.attributes[i].varType != orange.VarTypes.Continuous: continue
        for j in range(i+1, len(attrList)):
            if data.domain.attributes[j].varType != orange.VarTypes.Continuous: continue
            table = data.select([attrList[i], attrList[j]])
            table = orange.Preprocessor_dropMissing(table)
            attr1 = [table[k][attrList[i]].value for k in range(len(table))]
            attr2 = [table[k][attrList[j]].value for k in range(len(table))]
            
            val, prob = statc.pearsonr(attr1, attr2)
            if abs(val) >= minCorrelation: correlations.append((abs(val), attrList[i], attrList[j]))

    correlations.sort()
    correlations.reverse()
    return correlations
    

def addBestToCurrentProj(currentProj, attrInfo):
    for (val, a1, a2) in attrInfo:
        if a1 == currentProj[0] or a2 == currentProj[0] or a1 == currentProj[-1] or a2 == currentProj[-1]:
            if a1 == currentProj[0]: return ([a2] + currentProj, (val, a1, a2), a1, val)
            elif a2 == currentProj[0]: return ([a1] + currentProj, (val, a1, a2), a2, val)
            elif a1 == currentProj[-1]: return (currentProj + [a2], (val, a1, a2), a1, val)
            else                       : return (currentProj + [a1], (val, a1, a2), a2, val)
    return (None, None, None, None)

def removeAttribute(attr, group, currentProj, attrInfo):
    if attr == group[1]: attr2 = group[2]
    else: attr2 = group[1]

    # remove attribute attr, that is fixed inside currentProj
    for i in range(len(attrInfo)-1, -1, -1):
        if attrInfo[i][1] == attr or attrInfo[i][2] == attr:
            attrInfo.remove(attrInfo[i])
        elif attrInfo[i][1] in currentProj and attrInfo[i][2] in currentProj:
            attrInfo.remove(attrInfo[i])

    return attrInfo


def optimizeAttributeOrder(attrInfo, currentProj, currentVal, numberOfAttributes, optimizationDlg, app = None):
    if len(currentProj) == numberOfAttributes:
        for attr in currentProj:
            if currentProj.count(attr) > 1: return
        optimizationDlg.addProjection(currentVal/(numberOfAttributes-1), currentProj)
        return
    elif attrInfo == []: return
    elif len(currentProj) > 0 and optimizationDlg.getWorstVal() > currentVal/(len(currentProj)-1):
        print "skipping search at depth ", len(currentProj)
        return

    if not optimizationDlg.canContinueOptimization(): return
    if app: app.processEvents()        # allow processing of other events
    
    if currentProj == []:
        newCurrentProj = [attrInfo[0][1], attrInfo[0][2]]
        newCurrentVal = attrInfo[0][0]
        group = attrInfo[0]
        newAttrInfo = list(attrInfo)
    else:
        newCurrentProj, group, attr, val = addBestToCurrentProj(currentProj, attrInfo)
        if not newCurrentProj: return    # there are no more attributes that can be added to this projection
        newCurrentVal = currentVal + val
        newAttrInfo = removeAttribute(attr, group, newCurrentProj, list(attrInfo))

    if group in attrInfo: attrInfo.remove(group)
    if group in newAttrInfo: newAttrInfo.remove(group)
    
    optimizeAttributeOrder(newAttrInfo, newCurrentProj, newCurrentVal, numberOfAttributes, optimizationDlg, app)
    if not optimizationDlg.canContinueOptimization(): return

    optimizeAttributeOrder(attrInfo, currentProj, currentVal, numberOfAttributes, optimizationDlg, app)
    if not optimizationDlg.canContinueOptimization(): return
    


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
        



