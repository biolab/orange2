"""
#####################
Scoring (``scoring``)
#####################

.. index:: feature scoring

.. index:: 
   single: feature; feature scoring

Feature scoring is assessment of the usefulness of the feature for 
prediction of the dependant (class) variable.

To compute the information gain of feature "tear_rate" in the Lenses data set (loaded into `data`) use:

    >>> meas = Orange.feature.scoring.InfoGain()
    >>> print meas("tear_rate", data)
    0.548794925213

Apart from information gain you could also use other scoring methods;
:ref:`classification` and :ref:`regression`. For various
ways to call them see :ref:`callingscore`.

It is possible to construct the object and use
it on-the-fly::

    >>> print Orange.feature.scoring.InfoGain("tear_rate", data)
    0.548794925213

But constructing new instances for each feature is slow for
scoring methods that use caching, such as :obj:`Relief`.

Scoring features that are not in the domain is also possible. For
instance, discretized features can be scored without producing a
data table in advance (slow with :obj:`Relief`):

.. literalinclude:: code/scoring-info-iris.py
    :lines: 7-11

The following example computes feature scores, both with
:obj:`score_all` and by scoring each feature individually, and prints out 
the best three features. 

.. _scoring-all.py: code/scoring-all.py
.. _voting.tab: code/voting.tab

.. literalinclude:: code/scoring-all.py
    :lines: 7-

The output::

    Feature scores for best three features (with score_all):
    0.613 physician-fee-freeze
    0.255 el-salvador-aid
    0.228 synfuels-corporation-cutback

    Feature scores for best three features (scored individually):
    0.613 physician-fee-freeze
    0.255 el-salvador-aid
    0.228 synfuels-corporation-cutback

.. comment::

    The next script uses :obj:`GainRatio` and :obj:`Relief`.

    .. literalinclude:: code/scoring-relief-gainRatio.py
        :lines: 7-

    Notice that on this data the ranks of features match::
        
        Relief GainRt Feature
        0.613  0.752  physician-fee-freeze
        0.255  0.444  el-salvador-aid
        0.228  0.414  synfuels-corporation-cutback
        0.189  0.382  crime
        0.166  0.345  adoption-of-the-budget-resolution


.. _callingscore:

=======================
Calling scoring methods
=======================

To score a feature use :obj:`Score.__call__`. There are diferent
function signatures, which enable optimization. For instance,
if contingency matrix has already been computed, you can speed
up the computation by passing it to the scoring method (if it supports
that form - most do). Otherwise the scoring method will have to compute the
contingency itself.

Not all classes will accept all kinds of arguments. :obj:`Relief`,
for instance, only supports the form with instances on the input.

.. method:: Score.__call__(attribute, instances[, apriori_class_distribution][, weightID])

    :param attribute: the chosen feature, either as a descriptor, 
      index, or a name.
    :type attribute: :class:`Orange.data.variable.Variable` or int or string
    :param instances: data.
    :type instances: `Orange.data.Table`
    :param weightID: id for meta-feature with weight.

    All scoring methods need to support these parameters.

.. method:: Score.__call__(attribute, domain_contingency[, apriori_class_distribution])

    :param attribute: the chosen feature, either as a descriptor, 
      index, or a name.
    :type attribute: :class:`Orange.data.variable.Variable` or int or string
    :param domain_contingency: 
    :type domain_contingency: :obj:`Orange.statistics.contingency.Domain`

.. method:: Score.__call__(contingency, class_distribution[, apriori_class_distribution])

    :param contingency:
    :type contingency: :obj:`Orange.statistics.contingency.VarClass`
    :param class_distribution: distribution of the class
      variable. If :obj:`unknowns_treatment` is :obj:`IgnoreUnknowns`,
      it should be computed on instances where feature value is
      defined. Otherwise, class distribution should be the overall
      class distribution.
    :type class_distribution: 
      :obj:`Orange.statistics.distribution.Distribution`
    :param apriori_class_distribution: Optional and most often
      ignored. Useful if the scoring method makes any probability estimates
      based on apriori class probabilities (such as the m-estimate).
    :return: Feature score - the higher the value, the better the feature.
      If the quality cannot be scored, return :obj:`Score.Rejected`.
    :rtype: float or :obj:`Score.Rejected`.

The code below scores the same feature with :obj:`GainRatio` in
different ways.

.. literalinclude:: code/scoring-calls.py
    :lines: 7-

.. _classification:

===========================
Feature scoring in classification problems
===========================

.. Undocumented: MeasureAttribute_IM, MeasureAttribute_chiSquare, MeasureAttribute_gainRatioA, MeasureAttribute_logOddsRatio, MeasureAttribute_splitGain.

.. index:: 
   single: feature scoring; information gain

.. class:: InfoGain

    Information gain - the expected decrease of entropy. See `page on wikipedia
    <http://en.wikipedia.org/wiki/Information_gain_ratio>`_.

.. index:: 
   single: feature scoring; gain ratio

.. class:: GainRatio

    Information gain ratio - information gain divided by the entropy of the feature's
    value. Introduced in [Quinlan1986]_ in order to avoid overestimation
    of multi-valued features. It has been shown, however, that it still
    overestimates features with multiple values. See `Wikipedia
    <http://en.wikipedia.org/wiki/Information_gain_ratio>`_.

.. index:: 
   single: feature scoring; gini index

.. class:: Gini

    Gini index is the probability that two randomly chosen instances will have different
    classes. See `Gini coefficient on Wikipedia <http://en.wikipedia.org/wiki/Gini_coefficient>`_.

.. index:: 
   single: feature scoring; relevance

.. class:: Relevance

    The potential value for decision rules.

.. index:: 
   single: feature scoring; cost

.. class:: Cost

    Evaluates features based on the cost decrease achieved by knowing the value of
    feature, according to the specified cost matrix.

    .. attribute:: cost
     
        Cost matrix, see :obj:`Orange.classification.CostMatrix` for details.

    If the cost of predicting the first class of an instance that is actually in
    the second is 5, and the cost of the opposite error is 1, than an appropriate
    score can be constructed as follows::

    .. comment:: opposite error - is this term correct? TODO

        >>> meas = Orange.feature.scoring.Cost()
        >>> meas.cost = ((0, 5), (1, 0))
        >>> meas(3, data)
        0.083333350718021393

    Knowing the value of feature 3 would decrease the
    classification cost for approximately 0.083 per instance.

.. index:: 
   single: feature scoring; ReliefF

.. class:: Relief

    Assesses features' ability to distinguish between very similar
    instances from different classes. This scoring method was
    first developed by Kira and
    Rendell and then improved by Kononenko. The class :obj:`Relief`
    works on discrete and continuous classes and thus implements ReliefF
    and RReliefF.

    .. attribute:: k
    
       Number of neighbours for each instance. Default is 5.

    .. attribute:: m
    
        Number of reference instances. Default is 100. Set to -1 to take all the
        instances.

    .. attribute:: check_cached_data
    
        Check if the cached data is changed with data checksum. Slow
        on large tables.  Defaults to True. Disable it if you know that
        the data will not change.

    ReliefF is slow since it needs to find k nearest neighbours for
    each of m reference instances. As we normally compute ReliefF for
    all features in the dataset, :obj:`Relief` caches the results for
    all features, when called to score a certain feature.  When called
    again, it uses the stored results if the domain and the data table
    have not changed (data table version and the data checksum are
    compared). Caching will only work if you use the same object. 
    Constructing new instances of :obj:`Relief` fir each feature,
    like this::

        for attr in data.domain.attributes:
            print Orange.feature.scoring.Relief(attr, data)

    runs much slower than reusing the same instance::

        meas = Orange.feature.scoring.Relief()
        for attr in table.domain.attributes:
            print meas(attr, data)

    .. note::
       Relief can also compute the threshold function, that is, the feature
       quality at different thresholds for binarization.

.. autoclass:: Orange.feature.scoring.Distance
   
.. autoclass:: Orange.feature.scoring.MDL

.. _regression:

=======================
Feature scoring in regression problems
=======================

You can also use :obj:`Relief` for regression.

.. index:: 
   single: feature scoring; mean square error

.. class:: MSE

    Implements the mean square error score.

    .. attribute:: unknowns_treatment
    
        What to do with unknown values. See :obj:`Score.unknowns_treatment`.

    .. attribute:: m
    
        Parameter for m-estimate of error. Default is 0 (no m-estimate).



============
Base Classes
============

Implemented methods for scoring relevances of features to the class
are subclasses of :obj:`Score`. Those that compute statistics on
conditional distributions of class values given the feature values are
derived from :obj:`ScoreFromProbabilities`.

.. class:: Score

    Abstract base class for feature scoring. Its attributes describe which
    features it can handle and the required data.

    **Capabilities**

    .. attribute:: handles_discrete
    
        Indicates whether the scoring method can handle discrete features.

    .. attribute:: handles_continuous
    
        Indicates whether the scoring method can handle continuous features.

    .. attribute:: computes_thresholds
    
        Indicates whether the scoring method implements the :obj:`threshold_function`.

    **Input specification**

    .. attribute:: needs
    
        The type of data needed: :obj:`Generator`, :obj:`DomainContingency`,
        or :obj:`Contingency_Class`.

    .. attribute:: Generator

        Constant. Indicates that the scoring method needs an instance generator on the input (as, for example,
        :obj:`Relief`).

    .. attribute:: DomainContingency

        Constant. Indicates that the scoring method needs :obj:`Orange.statistics.contingency.Domain`.

    .. attribute:: Contingency_Class

        Constant. Indicates, that the scoring method needs the contingency
        (:obj:`Orange.statistics.contingency.VarClass`), feature
        distribution and the apriori class distribution (as most
        scoring methods).

    **Treatment of unknown values**

    .. attribute:: unknowns_treatment

        Not defined in :obj:`Score` but defined in
        classes that are able to treat unknown values. Either
        :obj:`IgnoreUnknowns`, :obj:`ReduceByUnknown`.
        :obj:`UnknownsToCommon`, or :obj:`UnknownsAsValue`.

    .. attribute:: IgnoreUnknowns

        Constant. Instances for which the feature value is unknown are removed.

    .. attribute:: ReduceByUnknown

        Constant. Features with unknown values are 
        punished. The feature quality is reduced by the proportion of
        unknown values. For impurity scores the impurity decreases
        only where the value is defined and stays the same otherwise.

    .. attribute:: UnknownsToCommon

        Constant. Undefined values are replaced by the most common value.

    .. attribute:: UnknownsAsValue

        Constant. Unknown values are treated as a separate value.

    **Methods**

    .. method:: __call__

        Abstract. See :ref:`callingscore`.

    .. method:: threshold_function(attribute, instances[, weightID])
    
        Abstract. 
        
        Assess different binarizations of the continuous feature
        :obj:`attribute`.  Return a list of tuples, where the first
        element is a threshold (between two existing values), the second
        is the quality of the corresponding binary feature, and the last
        the distribution of instancs below and above the threshold. The
        last element is optional.

        To show the computation of thresholds, we shall use the Iris data set
        (part of `scoring-info-iris.py`_, uses `iris.tab`_):

        .. literalinclude:: code/scoring-info-iris.py
            :lines: 13-15

    .. method:: best_threshold(attribute, instances)

        Return the best threshold for binarization, that is, the threshold
        with which the resulting binary feature will have the optimal
        score.

        The script below prints out the best threshold for
        binarization of an feature. ReliefF is used scoring: (part of
        `scoring-info-iris.py`_, uses `iris.tab`_):

        .. literalinclude:: code/scoring-info-iris.py
            :lines: 17-18

.. class:: ScoreFromProbabilities

    Bases: :obj:`Score`

    Abstract base class for feature scoring method that can be
    computed from contingency matrices only. It relieves the derived classes
    from having to compute the contingency matrix by defining the first two
    forms of call operator. (Well, that's not something you need to know if
    you only work in Python.)

    .. attribute:: unknowns_treatment
     
        See :obj:`Score.unknowns_treatment`.

    .. attribute:: estimator_constructor
    .. attribute:: conditional_estimator_constructor
    
        The classes that are used to estimate unconditional and
        conditional probabilities of classes, respectively. You can set
        this to, for instance, :obj:`ProbabilityEstimatorConstructor_m`
        and :obj:`ConditionalProbabilityEstimatorConstructor_ByRows`
        (with estimator constructor again set to
        :obj:`ProbabilityEstimatorConstructor_m`), respectively.
        Both default to relative frequencies.

============
Other
============

.. autoclass:: Orange.feature.scoring.OrderAttributes
   :members:

.. autofunction:: Orange.feature.scoring.merge_values

.. autofunction:: Orange.feature.scoring.score_all

.. comment .. rubric:: References

.. [Kononenko2007] Igor Kononenko, Matjaz Kukar: Machine Learning and Data Mining, 
  Woodhead Publishing, 2007.

.. [Quinlan1986] J R Quinlan: Induction of Decision Trees, Machine Learning, 1986.

.. [Breiman1984] L Breiman et al: Classification and Regression Trees, Chapman and Hall, 1984.


.. _iris.tab: code/iris.tab
.. _lenses.tab: code/lenses.tab
.. _scoring-relief-gainRatio.py: code/scoring-relief-gainRatio.py
.. _voting.tab: code/voting.tab
.. _selection-best3.py: code/selection-best3.py
.. _scoring-info-lenses.py: code/scoring-info-lenses.py
.. _scoring-info-iris.py: code/scoring-info-iris.py
.. _scoring-diff-measures.py: code/scoring-diff-measures.py

.. _scoring-regression.py: code/scoring-regression.py
.. _scoring-relief-caching: code/scoring-relief-caching

"""

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
        If None, :obj:`Relief` with m=5 and k=10 will be used.
    
    """
    def __init__(self, score=None):
        self.score = score

    def __call__(self, data, weight):
        """Score and order all features.

        :param data: a data table used to score features
        :type data: Orange.data.Table

        :param weight: meta attribute that stores weights of instances
        :type weight: Orange.data.variable.Variable

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
    """The 1-D feature distance score described in [Kononenko2007]_. TODO"""

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
        :type attr: Orange.data.variable.Variable

        :param data: a data table used to score features
        :type data: Orange.data.table

        :param apriori_dist: 
        :type apriori_dist:
        
        :param weightID: meta feature used to weight individual data instances
        :type weightID: Orange.data.variable.Variable

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
    """Score feature based on the minimum description length principle. TODO."""

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
        :type attr: Orange.data.variable.Variable

        :param data: a data table used to score the feature
        :type data: Orange.data.table

        :param apriori_dist: 
        :type apriori_dist:
        
        :param weightID: meta feature used to weight individual data instances
        :type weightID: Orange.data.variable.Variable

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
