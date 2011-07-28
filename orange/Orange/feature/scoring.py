"""
#####################
Scoring (``scoring``)
#####################

.. index:: feature scoring

.. index:: 
   single: feature; feature scoring

Features selection aims to find relevant features for the given
prediction task.

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


============
Base Classes
============

Implemented methods for scoring relevances of features to the class
are subclasses of :obj:`Measure`. Those that compute statistics on
conditional distributions of class values given the feature values are
derived from :obj:`MeasureFromProbabilities`.

.. class:: Measure

    Abstract base class for feature scoring. Its attributes describe which
    features it can handle and the required data.

    **Capabilities**

    .. attribute:: handles_discrete
    
        Indicates whether the measure can handle discrete features.

    .. attribute:: handles_continuous
    
        Indicates whether the measure can handle continuous features.

    .. attribute:: computes_thresholds
    
        Indicates whether the measure implements the :obj:`threshold_function`.

    **Input specification**

    .. attribute:: needs
    
        The type of data needed: :obj:`NeedsGenerator`, :obj:`NeedsDomainContingency`,
        or :obj:`NeedsContingency_Class`.

    .. attribute:: NeedsGenerator

        Constant. Indicates that the measure Needs an instance generator on the input (as, for example, the
        :obj:`Relief` measure).

    .. attribute:: NeedsDomainContingency

        Constant. Indicates that the measure needs :obj:`Orange.statistics.contingency.Domain`.

    .. attribute:: NeedsContingency_Class

        Constant. Indicates, that the measure needs the contingency
        (:obj:`Orange.statistics.contingency.VarClass`), feature
        distribution and the apriori class distribution (as most
        measures).

    **Treatment of unknown values**

    .. attribute:: unknowns_treatment

        Not defined in :obj:`Measure` but defined in
        classes that are able to treat unknown values. Either
        :obj:`IgnoreUnknowns`, :obj:`ReduceByUnknown`.
        :obj:`UnknownsToCommon`, or :obj:`UnknownsAsValue`.

    .. attribute:: IgnoreUnknowns

        Constant. Examples for which the feature value is unknown are removed.

    .. attribute:: ReduceByUnknown

        Constant. Features with unknown values are 
        punished. The feature quality is reduced by the proportion of
        unknown values. For impurity measures the impurity decreases
        only where the value is defined and stays the same otherwise,

    .. attribute:: UnknownsToCommon

        Constant. Undefined values are replaced by the most common value.

    .. attribute:: UnknownsAsValue

        Constant. Unknown values are treated as a separate value.

    **Methods**

    .. method:: __call__(attribute, instances[, apriori_class_distribution][, weightID])

        :param attribute: the chosen feature, either as a descriptor, 
          index, or a name.
        :type attribute: :class:`Orange.data.variable.Variable` or int or string
        :param instances: data.
        :type instances: `Orange.data.Table`
        :param weightID: id for meta-feature with weight.

        Abstract. All measures need to support `__call__` with these
        parameters.  Described below.

    .. method:: __call__(attribute, domain_contingency[, apriori_class_distribution])

        :param attribute: the chosen feature, either as a descriptor, 
          index, or a name.
        :type attribute: :class:`Orange.data.variable.Variable` or int or string
        :param domain_contingency: 
        :type domain_contingency: :obj:`Orange.statistics.contingency.Domain`

        Abstract. Described below.
        
    .. method:: __call__(contingency, class_distribution[, apriori_class_distribution])

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
          ignored. Useful if the measure makes any probability estimates
          based on apriori class probabilities (such as the m-estimate).
        :return: Feature score - the higher the value, the better the feature.
          If the quality cannot be measured, return :obj:`Measure.Rejected`.
        :rtype: float or :obj:`Measure.Rejected`.

        Abstract.

        Different forms of `__call__` enable optimization.  For instance,
        if contingency matrix has already been computed, you can speed
        up the computation by passing it to the measure (if it supports
        that form - most do). Otherwise the measure will have to compute the
        contingency itself.

        Not all classes will accept all kinds of arguments. :obj:`Relief`,
        for instance, only supports the form with instances on the input.

        The code sample below shows the use of :obj:`GainRatio` with
        different call types.

        .. literalinclude:: code/scoring-calls.py
            :lines: 7-

    .. method:: threshold_function(attribute, examples[, weightID])
    
        Abstract. 
        
        Assess different binarizations of the continuous feature
        :obj:`attribute`.  Return a list of tuples, where the first
        element is a threshold (between two existing values), the second
        is the quality of the corresponding binary feature, and the last
        the distribution of examples below and above the threshold. The
        last element is optional.

    .. method:: best_threshold

        Return the best threshold for binarization. Parameters?


    The script below shows different ways to assess the quality of astigmatic,
    tear rate and the first feature in the dataset lenses.

    .. literalinclude:: code/scoring-info-lenses.py
        :lines: 7-21

    As for many other classes in Orange, you can construct the object and use
    it on-the-fly. For instance, to measure the quality of feature
    "tear_rate", you could write simply::

        >>> print Orange.feature.scoring.Info("tear_rate", data)
        0.548794984818

    You shouldn't use this with :obj:`Relief`; see :obj:`Relief` for the explanation.

    It is also possible to score features that are not 
    in the domain. For instance, you can score discretized
    features on the fly:

    .. literalinclude:: code/scoring-info-iris.py
        :lines: 7-11

    Note that this is not possible with :obj:`Relief`, as it would be too slow.

    To show the computation of thresholds, we shall use the Iris data set.

    `scoring-info-iris.py`_ (uses `iris.tab`_):

    .. literalinclude:: code/scoring-info-iris.py
        :lines: 7-15

    If we hadn't constructed the feature in advance, we could write 
    `Orange.feature.scoring.Relief().threshold_function("petal length", data)`.
    This is not recommendable for ReliefF, since it may be a lot slower.

    The script below finds and prints out the best threshold for binarization
    of an feature, that is, the threshold with which the resulting binary
    feature will have the optimal ReliefF (or any other measure)::

        thresh, score, distr = meas.best_threshold("petal length", data)
        print "Best threshold: %5.3f (score %5.3f)" % (thresh, score)

.. class:: MeasureFromProbabilities

    Bases: :obj:`Measure`

    Abstract base class for feature quality measures that can be
    computed from contingency matrices only. It relieves the derived classes
    from having to compute the contingency matrix by defining the first two
    forms of call operator. (Well, that's not something you need to know if
    you only work in Python.)

    .. attribute:: unknowns_treatment
     
        See :obj:`Measure.unknowns_treatment`.

    .. attribute:: estimator_constructor
    .. attribute:: conditional_estimator_constructor
    
        The classes that are used to estimate unconditional and
        conditional probabilities of classes, respectively. You can set
        this to, for instance, :obj:`ProbabilityEstimatorConstructor_m`
        and :obj:`ConditionalProbabilityEstimatorConstructor_ByRows`
        (with estimator constructor again set to
        :obj:`ProbabilityEstimatorConstructor_m`), respectively.
        Both default to relative frequencies.

===========================
Measures for Classification
===========================

This script uses :obj:`GainRatio` and :obj:`Relief`.

.. literalinclude:: code/scoring-relief-gainRatio.py
    :lines: 7-

Notice that on this data the ranks of features match::
    
    Relief GainRt Feature
    0.613  0.752  physician-fee-freeze
    0.255  0.444  el-salvador-aid
    0.228  0.414  synfuels-corporation-cutback
    0.189  0.382  crime
    0.166  0.345  adoption-of-the-budget-resolution

Undocumented: MeasureAttribute_IM, MeasureAttribute_chiSquare, MeasureAttribute_gainRatioA, MeasureAttribute_logOddsRatio, MeasureAttribute_splitGain.

.. index:: 
   single: feature scoring; information gain

.. class:: InfoGain

    Measures the expected decrease of entropy.

.. index:: 
   single: feature scoring; gain ratio

.. class:: GainRatio

    Information gain divided by the entropy of the feature's
    value. Introduced by Quinlan in order to avoid overestimation of
    multi-valued features. It has been shown, however, that it
    still overestimates features with multiple values.

.. index:: 
   single: feature scoring; gini index

.. class:: Gini

    The probability that two randomly chosen examples will have different
    classes; first introduced by Breiman.

.. index:: 
   single: feature scoring; relevance

.. class:: Relevance

    The potential value for decision rules.

.. index:: 
   single: feature scoring; cost

.. class:: Cost

    Evaluates features based on the "saving" achieved by knowing the value of
    feature, according to the specified cost matrix.

    .. attribute:: cost
     
        Cost matrix, see :obj:`Orange.classification.CostMatrix` for details.

    If cost of predicting the first class of an example that is actually in
    the second is 5, and the cost of the opposite error is 1, than an appropriate
    measure can be constructed as follows::

        >>> meas = Orange.feature.scoring.Cost()
        >>> meas.cost = ((0, 5), (1, 0))
        >>> meas(3, data)
        0.083333350718021393

    Knowing the value of feature 3 would decrease the
    classification cost for approximately 0.083 per example.

.. index:: 
   single: feature scoring; ReliefF

.. class:: Relief

    Assesses features' ability to distinguish between very similar
    examples from different classes.  First developed by Kira and Rendell
    and then improved by Kononenko.

    .. attribute:: k
    
       Number of neighbours for each example. Default is 5.

    .. attribute:: m
    
        Number of reference examples. Default is 100. Set to -1 to take all the
        examples.

    .. attribute:: check_cached_data
    
        Check if the cached data is changed with data checksum. Slow
        on large tables.  Defaults to True. Disable it if you know that
        the data will not change.

    ReliefF is slow since it needs to find k nearest neighbours for each
    of m reference examples.  As we normally compute ReliefF for all
    features in the dataset, :obj:`Relief` caches the results. When called
    to score a certain feature, it computes all feature scores.
    When called again, it uses the stored results if the domain and the
    data table have not changed (data table version and the data checksum
    are compared). Caching will only work if you use the same instance.
    So, don't do this::

        for attr in data.domain.attributes:
            print Orange.feature.scoring.Relief(attr, data)

    But this::

        meas = Orange.feature.scoring.Relief()
        for attr in table.domain.attributes:
            print meas(attr, data)

    Class :obj:`Relief` works on discrete and continuous classes and thus 
    implements functionality of algorithms ReliefF and RReliefF.

    .. note::
       Relief can also compute the threshold function, that is, the feature
       quality at different thresholds for binarization.


=======================
Measures for Regression
=======================

:obj:`Relief` can be also used for regression.

.. index:: 
   single: feature scoring; mean square error

.. class:: MSE

    Implements the mean square error measure.

    .. attribute:: unknowns_treatment
    
        What to do with unknown values. See :obj:`Measure.unknowns_treatment`.

    .. attribute:: m
    
        Parameter for m-estimate of error. Default is 0 (no m-estimate).

============
Other
============

.. autoclass:: Orange.feature.scoring.OrderAttributes
   :members:

.. autofunction:: Orange.feature.scoring.Distance

.. autoclass:: Orange.feature.scoring.DistanceClass
   :members:
   
.. autofunction:: Orange.feature.scoring.MDL

.. autoclass:: Orange.feature.scoring.MDLClass
   :members:

.. autofunction:: Orange.feature.scoring.merge_values

.. autofunction:: Orange.feature.scoring.score_all

==========
References
==========

* Igor Kononeko, Matjaz Kukar: Machine Learning and Data Mining, 
  Woodhead Publishing, 2007.

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

from orange import MeasureAttribute as Measure
from orange import MeasureAttributeFromProbabilities as MeasureFromProbabilities
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
    
    .. attribute::  measure
    
        A measure derived from :obj:`~Orange.feature.scoring.Measure`.
        If None, :obj:`Relief` will be used.
    
    """
    def __init__(self, measure=None):
        self.measure = measure

    def __call__(self, data, weight):
        """Score and order all features.

        :param data: a data table used to score features
        :type data: Orange.data.table

        :param weight: meta attribute that stores weights of instances
        :type weight: Orange.data.variable

        """
        if self.measure:
            measure = self.measure
        else:
            measure = Relief(m=5,k=10)

        measured = [(attr, measure(attr, data, None, weight)) for attr in data.domain.attributes]
        measured.sort(lambda x, y: cmp(x[1], y[1]))
        return [x[0] for x in measured]

def Distance(attr=None, data=None):
    """Instantiate :obj:`DistanceClass` and use it to return
    the score of a given feature on given data.
    
    :param attr: feature to score
    :type attr: Orange.data.variable
    
    :param data: data table used for feature scoring
    :type data: Orange.data.table 
    
    """
    m = DistanceClass()
    if attr != None and data != None:
        return m(attr, data)
    else:
        return m

class DistanceClass(Measure):
    """The 1-D feature distance measure described in Kononenko."""

    @Orange.misc.deprecated_keywords({"aprioriDist": "apriori_dist"})
    def __call__(self, attr, data, apriori_dist=None, weightID=None):
        """Take :obj:`Orange.data.table` data table and score the given 
        :obj:`Orange.data.variable`.

        :param attr: feature to score
        :type attr: Orange.data.variable

        :param data: a data table used to score features
        :type data: Orange.data.table

        :param apriori_dist: 
        :type apriori_dist:
        
        :param weightID: meta feature used to weight individual data instances
        :type weightID: Orange.data.variable

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

def MDL(attr=None, data=None):
    """Instantiate :obj:`MDLClass` and use it n given data to
    return the feature's score."""
    m = MDLClass()
    if attr != None and data != None:
        return m(attr, data)
    else:
        return m

class MDLClass(Measure):
    """Score feature based on the minimum description length principle."""

    @Orange.misc.deprecated_keywords({"aprioriDist": "apriori_dist"})
    def __call__(self, attr, data, apriori_dist=None, weightID=None):
        """Take :obj:`Orange.data.table` data table and score the given 
        :obj:`Orange.data.variable`.

        :param attr: feature to score
        :type attr: Orange.data.variable

        :param data: a data table used to score the feature
        :type data: Orange.data.table

        :param apriori_dist: 
        :type apriori_dist:
        
        :param weightID: meta feature used to weight individual data instances
        :type weightID: Orange.data.variable

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


@Orange.misc.deprecated_keywords({"attrList": "attr_list", "attrMeasure": "attr_measure", "removeUnusedValues": "remove_unused_values"})
def merge_values(data, attr_list, attr_measure, remove_unused_values = 1):
    import orngCI
    #data = data.select([data.domain[attr] for attr in attr_list] + [data.domain.classVar])
    newData = data.select(attr_list + [data.domain.class_var])
    newAttr = orngCI.FeatureByCartesianProduct(newData, attr_list)[0]
    dist = orange.Distribution(newAttr, newData)
    activeValues = []
    for i in range(len(newAttr.values)):
        if dist[newAttr.values[i]] > 0: activeValues.append(i)
    currScore = attr_measure(newAttr, newData)
    while 1:
        bestScore, bestMerge = currScore, None
        for i1, ind1 in enumerate(activeValues):
            oldInd1 = newAttr.get_value_from.lookupTable[ind1]
            for ind2 in activeValues[:i1]:
                newAttr.get_value_from.lookupTable[ind1] = ind2
                score = attr_measure(newAttr, newData)
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
def score_all(data, measure=Relief(k=20, m=50)):
    """Assess the quality of features using the given measure and return
    a sorted list of tuples (feature name, measure).

    :param data: data table should include a discrete class.
    :type data: :obj:`Orange.data.table`
    :param measure:  feature scoring function. Derived from
      :obj:`Orange.feature.scoring.Measure`. Defaults to 
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
