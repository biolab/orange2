"""
Feature scoring is normally used in feature subset selection for classification
problems.

Let start with a simple script that reads the data, uses :obj:`attMeasure` to
derive attribute scores and prints out these for the first three best scored
attributes. Same scoring function is then used to report (only) on three best
score attributes.

`fss1.py`_ (uses `voting.tab`_)::

    import orange, import orngFSS
    data = orange.ExampleTable("voting")

    print 'Score estimate for first three attributes:'
    ma = orngFSS.attMeasure(data)
    for m in ma[:3]:
        print "%5.3f %s" % (m[1], m[0])

    n = 3
    best = orngFSS.bestNAtts(ma, n)
    print '\\nBest %d attributes:' % n
    for s in best:
        print s

The script should output this::

    Attribute scores for best three attributes:
    Attribute scores for best three attributes:
    0.728 physician-fee-freeze
    0.329 adoption-of-the-budget-resolution
    0.321 synfuels-corporation-cutback

    Best 3 attributes:
    physician-fee-freeze
    adoption-of-the-budget-resolution
    synfuels-corporation-cutback</xmp>

Functions and classes for feature scoring:

.. autoclass:: Orange.feature.scoring.OrderAttributesByMeasure
   :members:

.. automethod:: Orange.feature.scoring.MeasureAttribute_Distance

.. autoclass:: Orange.feature.scoring.MeasureAttribute_DistanceClass
   :members:
   
.. automethod:: Orange.feature.scoring.MeasureAttribute_MDL

.. autoclass:: Orange.feature.scoring.MeasureAttribute_MDLClass
   :members:

.. automethod:: Orange.feature.scoring.mergeAttrValues

.. automethod:: Orange.feature.scoring.attMeasure


========================
Different Score Measures
========================

.. note: add links to gain ratio, relief and other feature scores

The following script reports on gain ratio and relief attribute
scores. Notice that for our data set the ranks of the attributes
rather match well!

`fss2.py`_ (uses `voting.tab`_)::

    import orange, orngFSS
    data = orange.ExampleTable("voting")

    print 'Relief GainRt Attribute'
    ma_def = orngFSS.attMeasure(data)
    gainRatio = orange.MeasureAttribute_gainRatio()
    ma_gr  = orngFSS.attMeasure(data, gainRatio)
    for i in range(5):
        print "%5.3f  %5.3f  %s" % (ma_def[i][1], ma_gr[i][1], ma_def[i][0])

==========
References
==========

* Kononeko: Strojno ucenje.

"""

import Orange.core as orange

######
# from orngEvalAttr.py
class OrderAttributesByMeasure:
    """Construct an instance that orders features by their scores.
    
    :param measure: an attribute measure, derived from 
      :obj:`orange.MeasureAttribute`.
    
    """
    def __init__(self, measure=None):
        self.measure = measure

    def __call__(self, data, weight):
        """Take :obj:`Orange.data.table` data table and an instance of
        :obj:`orange.MeasureAttribute` to score and order features.  

        :param data: a data table used to score features
        :type data: Orange.data.table

        :param weight: meta feature that stores weights of individual data
          instances
        :type weight: Orange.data.feature

        """
        if self.measure:
            measure=self.measure
        else:
            measure=orange.MeasureAttribute_relief(m=5,k=10)

        measured=[(attr, measure(attr, data, None, weight)) for attr in data.domain.attributes]
        measured.sort(lambda x, y: cmp(x[1], y[1]))
        return [x[0] for x in measured]

def MeasureAttribute_Distance(attr = None, data = None):
    """Instantiate :obj:`MeasureAttribute_DistanceClass` and use it to return
    the score of a given feature on given data.
    
    :param attr: feature to score
    :type attr: Orange.data.feature
    
    :param data: data table used for feature scoring
    :type data: Orange.data.table 
    
    """
    m = MeasureAttribute_DistanceClass()
    if attr != None and data != None:
        return m(attr, data)
    else:
        return m

class MeasureAttribute_DistanceClass(orange.MeasureAttribute):
    """Implement the 1-D feature distance measure described in Kononenko."""
    def __call__(self, attr, data, aprioriDist = None, weightID = None):
        """Take :obj:`Orange.data.table` data table and score the given 
        :obj:`Orange.data.feature`.

        :param attr: feature to score
        :type attr: Orange.data.feature

        :param data: a data table used to score features
        :type data: Orange.data.table

        :param aprioriDist: 
        :type aprioriDist:
        
        :param weightID: meta feature used to weight individual data instances
        :type weightID: Orange.data.feature

        """
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

def MeasureAttribute_MDL(attr = None, data = None):
    """Instantiate :obj:`MeasureAttribute_MDLClass` and use it n given data to
    return the feature's score."""
    m = MeasureAttribute_MDLClass()
    if attr != None and data != None:
        return m(attr, data)
    else:
        return m

class MeasureAttribute_MDLClass(orange.MeasureAttribute):
    """Score feature based on the minimum description length principle."""
    def __call__(self, attr, data, aprioriDist = None, weightID = None):
        """Take :obj:`Orange.data.table` data table and score the given 
        :obj:`Orange.data.feature`.

        :param attr: feature to score
        :type attr: Orange.data.feature

        :param data: a data table used to score the feature
        :type data: Orange.data.table

        :param aprioriDist: 
        :type aprioriDist:
        
        :param weightID: meta feature used to weight individual data instances
        :type weightID: Orange.data.feature

        """
        attrClassCont = orange.ContingencyAttrClass(attr, data)
        classDist = orange.Distribution(data.domain.classVar, data).values()
        nCls = len(classDist)
        nEx = len(data)
        priorMDL = _logMultipleCombs(nEx, classDist) + _logMultipleCombs(nEx+nCls-1, [nEx, nCls-1])
        postPart1 = [_logMultipleCombs(sum(attrClassCont[key]), attrClassCont[key].values()) for key in attrClassCont.keys()]
        postPart2 = [_logMultipleCombs(sum(attrClassCont[key])+nCls-1, [sum(attrClassCont[key]), nCls-1]) for key in attrClassCont.keys()]
        ret = priorMDL
        for val in postPart1 + postPart2:
            ret -= val
        return ret / max(1, nEx)

# compute n! / k1! * k2! * k3! * ... kc!
# ks = [k1, k2, ...]
def _logMultipleCombs(n, ks):
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

######
# from orngFSS
def attMeasure(data, measure = orange.MeasureAttribute_relief(k=20, m=50)):
    """Assess the quality of attributes using the given measure and return
    a sorted list of tuples (attribute name, measure).

    :param data: data table should include a discrete class.
    :type data: Orange.data.table.
    :param measure:  attribute scoring function. Derived from
      :obj:`orange.MeasureAttribute`. Defaults to Defaults to 
      :obj:`orange.MeasureAttribute_relief` with k=20 and m=50.
    :type measure: :obj:`orange.MeasureAttribute` 
    :rtype: :obj:`list` a sorted list of tuples (attribute name, score)

    """
    measl=[]
    for i in data.domain.attributes:
        measl.append((i.name, measure(i, data)))
    measl.sort(lambda x,y:cmp(y[1], x[1]))
   
#  for i in measl:
#    print "%25s, %6.3f" % (i[0], i[1])
    return measl
