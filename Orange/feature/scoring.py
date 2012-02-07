import Orange.core as orange
import Orange.misc

from orange import MeasureAttribute as Score
from orange import MeasureAttributeFromProbabilities as ScoreFromProbabilities
from orange import MeasureAttribute_info as InfoGain
from orange import MeasureAttribute_gainRatio as GainRatio
from orange import MeasureAttribute_gini as Gini
from orange import MeasureAttribute_relevance as Relevance 
from orange import MeasureAttribute_cost as Cost
from orange import MeasureAttribute_relief as Relief
from orange import MeasureAttribute_MSE as MSE

######
# from orngEvalAttr.py

class OrderAttributes:
    """Orders features by their scores.
    
    .. attribute::  score
    
        A scoring method derived from :obj:`~Orange.feature.scoring.Score`.
        If :obj:`None`, :obj:`Relief` with m=5 and k=10 is used.
    
    """
    def __init__(self, score=None):
        self.score = score

    def __call__(self, data, weight):
        """Score and order all features.

        :param data: a data table used to score features
        :type data: Orange.data.Table

        :param weight: meta attribute that stores weights of instances
        :type weight: Orange.feature.Descriptor

        """
        if self.score:
            measure = self.score
        else:
            measure = Relief(m=5, k=10)

        measured = [(attr, measure(attr, data, None, weight)) for attr in data.domain.attributes]
        measured.sort(lambda x, y: cmp(x[1], y[1]))
        return [x[0] for x in measured]

OrderAttributes = Orange.misc.deprecated_members({
          "measure": "score",
}, wrap_methods=[])(OrderAttributes)

class Distance(Score):
    """The :math:`1-D` distance is defined as information gain divided
    by joint entropy :math:`H_{CA}` (:math:`C` is the class variable
    and :math:`A` the feature):

    .. math::
        1-D(C,A) = \\frac{\\mathrm{Gain}(A)}{H_{CA}}
    """

    @Orange.misc.deprecated_keywords({"aprioriDist": "apriori_dist"})
    def __new__(cls, attr=None, data=None, apriori_dist=None, weightID=None):
        self = Score.__new__(cls)
        if attr != None and data != None:
            #self.__init__(**argkw)
            return self.__call__(attr, data, apriori_dist, weightID)
        else:
            return self

    @Orange.misc.deprecated_keywords({"aprioriDist": "apriori_dist"})
    def __call__(self, attr, data, apriori_dist=None, weightID=None):
        """Score the given feature.

        :param attr: feature to score
        :type attr: Orange.feature.Descriptor

        :param data: a data table used to score features
        :type data: Orange.data.table

        :param apriori_dist: 
        :type apriori_dist:
        
        :param weightID: meta feature used to weight individual data instances
        :type weightID: Orange.feature.Descriptor

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

class MDL(Score):
    """Minimum description length principle [Kononenko1995]_. Let
    :math:`n` be the number of instances, :math:`n_0` the number of
    classes, and :math:`n_{cj}` the number of instances with feature
    value :math:`j` and class value :math:`c`. Then MDL score for the
    feature A is

    .. math::
         \mathrm{MDL}(A) = \\frac{1}{n} \\Bigg[
         \\log\\binom{n}{n_{1.},\\cdots,n_{n_0 .}} - \\sum_j
         \\log \\binom{n_{.j}}{n_{1j},\\cdots,n_{n_0 j}} \\\\
         + \\log \\binom{n+n_0-1}{n_0-1} - \\sum_j \\log
         \\binom{n_{.j}+n_0-1}{n_0-1}
         \\Bigg]
    """

    @Orange.misc.deprecated_keywords({"aprioriDist": "apriori_dist"})
    def __new__(cls, attr=None, data=None, apriori_dist=None, weightID=None):
        self = Score.__new__(cls)
        if attr != None and data != None:
            #self.__init__(**argkw)
            return self.__call__(attr, data, apriori_dist, weightID)
        else:
            return self

    @Orange.misc.deprecated_keywords({"aprioriDist": "apriori_dist"})
    def __call__(self, attr, data, apriori_dist=None, weightID=None):
        """Score the given feature.

        :param attr: feature to score
        :type attr: Orange.feature.Descriptor

        :param data: a data table used to score the feature
        :type data: Orange.data.table

        :param apriori_dist: 
        :type apriori_dist:
        
        :param weightID: meta feature used to weight individual data instances
        :type weightID: Orange.feature.Descriptor

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


@Orange.misc.deprecated_keywords({"attrList": "attr_list", "attrMeasure": "attr_score", "removeUnusedValues": "remove_unused_values"})
def merge_values(data, attr_list, attr_score, remove_unused_values = 1):
    import orngCI
    #data = data.select([data.domain[attr] for attr in attr_list] + [data.domain.classVar])
    newData = data.select(attr_list + [data.domain.class_var])
    newAttr = orngCI.FeatureByCartesianProduct(newData, attr_list)[0]
    dist = orange.Distribution(newAttr, newData)
    activeValues = []
    for i in range(len(newAttr.values)):
        if dist[newAttr.values[i]] > 0: activeValues.append(i)
    currScore = attr_score(newAttr, newData)
    while 1:
        bestScore, bestMerge = currScore, None
        for i1, ind1 in enumerate(activeValues):
            oldInd1 = newAttr.get_value_from.lookupTable[ind1]
            for ind2 in activeValues[:i1]:
                newAttr.get_value_from.lookupTable[ind1] = ind2
                score = attr_score(newAttr, newData)
                if score >= bestScore:
                    bestScore, bestMerge = score, (ind1, ind2)
                newAttr.get_value_from.lookupTable[ind1] = oldInd1

        if bestMerge:
            ind1, ind2 = bestMerge
            currScore = bestScore
            for i, l in enumerate(newAttr.get_value_from.lookupTable):
                if not l.isSpecial() and int(l) == ind1:
                    newAttr.get_value_from.lookupTable[i] = ind2
            newAttr.values[ind2] = newAttr.values[ind2] + "+" + newAttr.values[ind1]
            del activeValues[activeValues.index(ind1)]
        else:
            break

    if not remove_unused_values:
        return newAttr

    reducedAttr = orange.EnumVariable(newAttr.name, values = [newAttr.values[i] for i in activeValues])
    reducedAttr.get_value_from = newAttr.get_value_from
    reducedAttr.get_value_from.class_var = reducedAttr
    return reducedAttr

######
# from orngFSS
@Orange.misc.deprecated_keywords({"measure": "score"})
def score_all(data, score=Relief(k=20, m=50)):
    """Assess the quality of features using the given measure and return
    a sorted list of tuples (feature name, measure).

    :param data: data table should include a discrete class.
    :type data: :obj:`Orange.data.Table`
    :param score:  feature scoring function. Derived from
      :obj:`~Orange.feature.scoring.Score`. Defaults to 
      :obj:`~Orange.feature.scoring.Relief` with k=20 and m=50.
    :type measure: :obj:`~Orange.feature.scoring.Score` 
    :rtype: :obj:`list`; a sorted list of tuples (feature name, score)

    """
    measl=[]
    for i in data.domain.attributes:
        measl.append((i.name, score(i, data)))
    measl.sort(lambda x,y:cmp(y[1], x[1]))
    return measl
