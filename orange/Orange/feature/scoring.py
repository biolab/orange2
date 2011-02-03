"""
Feature scoring is used in feature subset selection for classification
problems. The goal is to find "good" features that are relevant for the given
classification task.

Here is a simple script that reads the data, uses :obj:`attMeasure` to
derive feature scores and prints out these for the first three best scored
features. Same scoring function is then used to report (only) on three best
score features.

.. _scoring-all.py: code/scoring-all.py
.. _voting.tab: code/voting.tab

`scoring-all.py`_ (uses `voting.tab`_):

.. literalinclude:: code/scoring-all.py
    :lines: 7-

The script should output this::

    Feature scores for best three features:
    0.613 physician-fee-freeze
    0.255 adoption-of-the-budget-resolution
    0.228 synfuels-corporation-cutback

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

The following script reports on gain ratio and relief feature scores.

`scoring-relief-gainRatio.py`_ (uses `voting.tab`_):

.. literalinclude:: code/scoring-relief-gainRatio.py
    :lines: 7-
    
Notice that on this data the ranks of features match rather well::
    
    Relief GainRt Feature
    0.613  0.752  physician-fee-freeze
    0.255  0.444  el-salvador-aid
    0.228  0.414  synfuels-corporation-cutback
    0.189  0.382  crime
    0.166  0.345  adoption-of-the-budget-resolution

==========
References
==========

* Kononeko: Strojno ucenje. Zalozba FE in FRI, Ljubljana, 2005.

.. _scoring-relief-gainRatio.py: code/scoring-relief-gainRatio.py
.. _voting.tab: code/voting.tab

"""

import Orange.core as orange
from orange import MeasureAttribute_gainRatio as GainRatio
from orange import MeasureAttribute as Measure
from orange import MeasureAttribute_relief as Relief
from orange import MeasureAttribute_info as InfoGain
from orange import MeasureAttribute_gini as Gini

######
# from orngEvalAttr.py
class OrderAttributesByMeasure:
    """Construct an instance that orders features by their scores.
    
    :param measure: a feature measure, derived from 
      :obj:`Orange.feature.scoring.Measure`.
    
    """
    def __init__(self, measure=None):
        self.measure = measure

    def __call__(self, data, weight):
        """Take :obj:`Orange.data.table` data table and an instance of
        :obj:`Orange.feature.scoring.Measure` to score and order features.  

        :param data: a data table used to score features
        :type data: Orange.data.table

        :param weight: meta feature that stores weights of individual data
          instances
        :type weight: Orange.data.feature

        """
        if self.measure:
            measure = self.measure
        else:
            measure = Relief(m=5,k=10)

        measured = [(attr, measure(attr, data, None, weight)) for attr in data.domain.attributes]
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
        infoGain = InfoGain(attr, data)
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
def attMeasure(data, measure=Relief(k=20, m=50)):
    """Assess the quality of features using the given measure and return
    a sorted list of tuples (feature name, measure).

    :param data: data table should include a discrete class.
    :type data: :obj:`Orange.data.table`
    :param measure:  feature scoring function. Derived from
      :obj:`Orange.feature.scoring.Measure`. Defaults to Defaults to 
      :obj:`Orange.feature.scoring.Relief` with k=20 and m=50.
    :type measure: :obj:`Orange.feature.scoring.Measure` 
    :rtype: :obj:`list` a sorted list of tuples (feature name, score)

    """
    measl=[]
    for i in data.domain.attributes:
        measl.append((i.name, measure(i, data)))
    measl.sort(lambda x,y:cmp(y[1], x[1]))
   
#  for i in measl:
#    print "%25s, %6.3f" % (i[0], i[1])
    return measl
