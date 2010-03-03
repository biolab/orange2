### Janez 03-02-14: Added weights
### Inform Blaz and remove this comment
import orange

class OrderAttributesByMeasure:
    def __init__(self, measure=None):
        self.measure=measure

    def __call__(self, data, weight):
        if self.measure:
            measure=self.measure
        else:
            measure=orange.MeasureAttribute_relief(m=5,k=10)

        measured=[(attr, measure(attr, data, None, weight)) for attr in data.domain.attributes]
        measured.sort(lambda x, y: cmp(x[1], y[1]))
        return [x[0] for x in measured]

def MeasureAttribute_Distance(attr = None, data = None):
    m = MeasureAttribute_DistanceClass()
    if attr != None and data != None:
        return m(attr, data)
    else:
        return m


# measure 1-D as described in Strojno ucenje (Kononenko)
class MeasureAttribute_DistanceClass(orange.MeasureAttribute):
    def __call__(self, attr, data, aprioriDist = None, weightID = None):
        import numpy
        from orngContingency import Entropy
        if attr in data.domain:  # if we receive attr as string we have to convert to variable
            attr = data.domain[attr]
        attrClassCont = orange.ContingencyAttrClass(attr, data)
        dist = []
        for vals in attrClassCont.values():
            dist += list(vals)
        classAttrEntropy = Entropy(numpy.array(dist))
        infoGain = orange.MeasureAttribute_info(attr, data)
        if classAttrEntropy > 0:
            return float(infoGain) / classAttrEntropy
        else:
            return 0

# attribute quality measure based on the minimum description length principle
def MeasureAttribute_MDL(attr = None, data = None):
    m = MeasureAttribute_MDLClass()
    if attr != None and data != None:
        return m(attr, data)
    else:
        return m

class MeasureAttribute_MDLClass(orange.MeasureAttribute):
    def __call__(self, attr, data, aprioriDist = None, weightID = None):
        attrClassCont = orange.ContingencyAttrClass(attr, data)
        classDist = orange.Distribution(data.domain.classVar, data).values()
        nCls = len(classDist)
        nEx = len(data)
        priorMDL = logMultipleCombs(nEx, classDist) + logMultipleCombs(nEx+nCls-1, [nEx, nCls-1])
        postPart1 = [logMultipleCombs(sum(attrClassCont[key]), attrClassCont[key].values()) for key in attrClassCont.keys()]
        postPart2 = [logMultipleCombs(sum(attrClassCont[key])+nCls-1, [sum(attrClassCont[key]), nCls-1]) for key in attrClassCont.keys()]
        ret = priorMDL
        for val in postPart1 + postPart2:
            ret -= val
        return ret / max(1, nEx)


# compute n! / k1! * k2! * k3! * ... kc!
# ks = [k1, k2, ...]
def logMultipleCombs(n, ks):
    import math
    m = max(ks)
    ks.remove(m)
    resArray = []
    for (start, end) in [(m+1, n+1)] + [(1, k+1) for k in ks]:
        ret = 0
        curr = 1
        for val in range(int(start), int(end)):
            curr *= val
            if curr > 1e40:
                ret += math.log(curr)
                curr = 1
        ret += math.log(curr)
        resArray.append(ret)
    ret = resArray[0]
    for val in resArray[1:]:
        ret -= val
    return ret



def mergeAttrValues(data, attrList, attrMeasure, removeUnusedValues = 1):
    import orngCI
    #data = data.select([data.domain[attr] for attr in attrList] + [data.domain.classVar])
    newData = data.select(attrList + [data.domain.classVar])
    newAttr = orngCI.FeatureByCartesianProduct(newData, attrList)[0]
    dist = orange.Distribution(newAttr, newData)
    activeValues = []
    for i in range(len(newAttr.values)):
        if dist[newAttr.values[i]] > 0: activeValues.append(i)
    currScore = attrMeasure(newAttr, newData)
    while 1:
        bestScore, bestMerge = currScore, None
        for i1, ind1 in enumerate(activeValues):
            oldInd1 = newAttr.getValueFrom.lookupTable[ind1]
            for ind2 in activeValues[:i1]:
                newAttr.getValueFrom.lookupTable[ind1] = ind2
                score = attrMeasure(newAttr, newData)
                if score >= bestScore:
                    bestScore, bestMerge = score, (ind1, ind2)
                newAttr.getValueFrom.lookupTable[ind1] = oldInd1

        if bestMerge:
            ind1, ind2 = bestMerge
            currScore = bestScore
            for i, l in enumerate(newAttr.getValueFrom.lookupTable):
                if not l.isSpecial() and int(l) == ind1:
                    newAttr.getValueFrom.lookupTable[i] = ind2
            newAttr.values[ind2] = newAttr.values[ind2] + "+" + newAttr.values[ind1]
            del activeValues[activeValues.index(ind1)]
        else:
            break

    if not removeUnusedValues:
        return newAttr

    reducedAttr = orange.EnumVariable(newAttr.name, values = [newAttr.values[i] for i in activeValues])
    reducedAttr.getValueFrom = newAttr.getValueFrom
    reducedAttr.getValueFrom.classVar = reducedAttr
    return reducedAttr



##if __name__=="__main__":
##    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\zoo.tab")
##    #newFeature, quality = FeatureByCartesianProduct(data, ["sex", "age"])
##    #MeasureAttribute_Distance()(newFeature, data)
##    #print logMultipleCombs(200, [70,30,100])
##    #import orngCI
##    #newFeature, quality = orngCI.FeatureByIM(data, ["milk", "airborne"], binary = 0, measure = MeasureAttribute_MDL())

