"""
.. index:: feature scoring

.. index:: 
   single: feature; feature scoring

Features selection aims to find relevant features for the given
prediction task.

The following example uses :obj:`attMeasure` to derive feature scores
and prints out the three best features.

.. _scoring-all.py: code/scoring-all.py
.. _voting.tab: code/voting.tab

`scoring-all.py`_ (uses `voting.tab`_):

.. literalinclude:: code/scoring-all.py
    :lines: 7-

The output::

    Feature scores for best three features:
    0.613 physician-fee-freeze
    0.255 adoption-of-the-budget-resolution
    0.228 synfuels-corporation-cutback

============
Base Classes
============

Orange implements several methods for scoring relevance of features to
the class. All are subclasses of :obj:`Measure`. The most common compute
statistics on conditional distributions of class values given the feature
values; these are derived from :obj:`MeasureFromProbabilities`.

.. class:: Measure

    Abstract base class for feature scoring. Its attributes describe which
    features it can handle and the required data.

    .. attribute:: handles_discrete
    
        Indicates whether the measure can handle discrete features.

    .. attribute:: handles_continuous
    
        Indicates whether the measure can handle continuous features.

    .. attribute:: computes_thresholds
    
        Indicated whether the measure implements the :obj:`threshold_function`.

    .. attribute:: needs
    
        The kind of data needed. Either

        * :obj:`NeedsGenerator`; an instance generator (as, for example,
          Relief)

        * :obj:`NeedsDomainContingency; needs
          :obj:`Orange.statistics.contingency.Domain`,a

        * :obj:`NeedsContingency_Class`; needs the contingency
          (:obj:`Orange.statistics.contingency.VarClass`), feature
          distribution and the apriori class distribution (as most
          measures).

    .. attribute:: unknowns_treatment

        Not defined in :obj:`Measure` but defined in
        classes that are able to treat unknown values. Possible values:
        
        * ignored (:obj:`Measure.IgnoreUnknowns`);
          examples for which the feature value is unknown are removed,

        * punished (:obj:`Measure.ReduceByUnknown`); the feature quality is
          reduced by the proportion of unknown values. For impurity measures
          the impurity decreases only where the value is defined and stays 
          the same otherwise,

        * imputed (:obj:`Measure.UnknownsToCommon`); undefined values are
          replaced by the most common value,

        * treated as a separate value (:obj:`Measure.UnknownsAsValue`).

    .. method:: __call__(attribute, examples[, apriori_class_distribution][, weightID])
    .. method:: __call__(attribute, domain_contingency[, apriori_class_distribution])
    .. method:: __call__(contingency, class_distribution[, apriori_class_distribution])

        :param attribute: the choosen feature, either as a descriptor, 
          index, or a name.
        :type attribute: :class:`Orange.data.variable.Variable` or int or string
        :param examples: data.
        :type examples: `Orange.data.Table`
        :param apriori_class_distribution: Optional and most often
          ignored. Useful if the measure makes any probability estimates
          based on apriori class probabilities (such as the m-estimate).
        :param weightID: id for meta-feature with weight.
        :param domain_contingency: Not sure.
        :type domain_contingency: :obj:`Orange.statistics.contingency.Domain`
        :param distribution: Not sure.
        :type distribution: :obj:`Orange.statistics.distribution.Distribution`


        Abstract. Return a float: the higher the value, the better the feature
        If the quality cannot be measured, return :obj:`Measure.Rejected`. 

        All measures need to support the first form, with the data on the input.

        Not all classes will accept all kinds of arguments. Relief, for instance,
        cannot be computed from contingencies alone. Besides, the feature and
        the class need to be of the correct type for a particular measure.

        Different forms of the call enable optimization.  For instance,
        if contingency matrix has already been computed, you can speed
        ab the computation by passing it to the measure (if it supports
        that form - most do). Otherwise the measurea will compute the
        contingency itself.

        Data is given either as examples (and, optionally, id for
        meta-feature with weight), contingency tables or distributions
        for all attributes. In the latter form, what is given as
        the class distribution depends upon what you do with unknown
        values (if there are any).  If :obj:`unknowns_treatment` is
        :obj:`IgnoreUnknowns`, the class distribution should be computed
        on examples for which the feature value is defined. Otherwise,
        class distribution should be the overall class distribution.


    .. method:: threshold_function(attribute, examples[, weightID])
    
        Abstract. Assess different binarizations of the continuous feature
        :obj:`attribute`.  Return a list of tuples, where the first
        element is a threshold (between two existing values), the second
        is the quality of the corresponding binary feature, and the last
        the distribution of examples below and above the threshold. The
        last element is optional.

    .. method:: best_threshold

        

    The script below shows different ways to assess the quality of astigmatic,
    tear rate and the first feature (whichever it is) in the dataset lenses.

    .. literalinclude:: code/scoring-info-lenses.py
        :lines: 7-21

    As for many other classes in Orange, you can construct the object and use
    it on-the-fly. For instance, to measure the quality of feature
    "tear_rate", you could write simply::

        >>> print Orange.feature.scoring.Info("tear_rate", data)
        0.548794984818

    You shouldn't use this shortcut with ReliefF, though; see the explanation
    in the section on ReliefF.

    XXXXXXXX It is also possible to assess the quality of features that do not exist
    in the features. For instance, you can assess the quality of discretized
    features without constructing a new domain and dataset that would include
    them.

    `scoring-info-iris.py`_ (uses `iris.tab`_):

    .. literalinclude:: code/scoring-info-iris.py
        :lines: 7-11

    The quality of the new feature d1 is assessed on data, which does not
    include the new feature at all. (Note that ReliefF won't do that since
    it would be too slow. ReliefF requires the feature to be present in the
    dataset.)

    Finally, you can compute the quality of meta-features. The following
    script adds a meta-feature to an example table, initializes it to random
    values and measures its information gain.

    `scoring-info-lenses.py`_ (uses `lenses.tab`_):

    .. literalinclude:: code/scoring-info-lenses.py
        :lines: 54-

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

    Abstract base class for feature quality measures that can be
    computed from contingency matrices only. It relieves the derived classes
    from having to compute the contingency matrix by defining the first two
    forms of call operator. (Well, that's not something you need to know if
    you only work in Python.) Additional feature of this class is that you can
    set probability estimators. If none are given, probabilities and
    conditional probabilities of classes are estimated by relative frequencies.

    .. attribute:: unknowns_treatment
     
        Defines what to do with unknown values. See the possibilities described above.

    .. attribute:: estimator_constructor
    .. attribute:: conditional_estimator_constructor
    
        The classes that are used to estimate unconditional and conditional
        probabilities of classes, respectively. You can set this to, for instance, 
        :obj:`ProbabilityEstimatorConstructor_m` and 
        :obj:`ConditionalProbabilityEstimatorConstructor_ByRows`
        (with estimator constructor again set to 
        :obj:`ProbabilityEstimatorConstructor_m`), respectively.

===========================
Measures for Classification
===========================

This script scores features with gain ratio and relief.

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

See  `scoring-info-lenses.py`_, `scoring-info-iris.py`_,
`scoring-diff-measures.py`_ and `scoring-regression.py`_
for examples on their use.

Found in Orange:
'MeasureAttribute_IM', 'MeasureAttribute_chiSquare', 'MeasureAttribute_gainRatioA', 'MeasureAttribute_logOddsRatio', 'MeasureAttribute_splitGain'

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
    
        Check if the cached data changed. Defaults to True. Best left alone.

    ReliefF is slow since it needs to find k nearest
    neighbours for each of m reference examples.
    Since we normally compute ReliefF for all features in the dataset,
    :obj:`Relief` caches the results. When it is called to compute a quality of
    certain feature, it computes qualities for all features in the dataset.
    When called again, it uses the stored results if the data has not changeddomain
    is still the same and the example table has not changed. Checking is done by
    comparing the data table version :obj:`Orange.data.Table` for details) and then
    computing a checksum of the data and comparing it with the previous checksum.
    The latter can take some time on large tables, so you may want to disable it
    by setting `checkCachedData` to :obj:`False`. In most cases it will do no harm,
    except when the data is changed in such a way that it passed unnoticed by the 
    version' control, in which cases the computed ReliefFs can be false. Hence:
    disable it if you know that the data does not change or if you know what kind
    of changes are detected by the version control.

    Caching will only have an effect if you use the same instance for all
    features in the domain. So, don't do this::

        for attr in data.domain.attributes:
            print Orange.feature.scoring.Relief(attr, data)

    In this script, cached data dies together with the instance of :obj:`Relief`,
    which is constructed and destructed for each feature separately. It's way
    faster to go like this::

        meas = Orange.feature.scoring.Relief()
        for attr in table.domain.attributes:
            print meas(attr, data)

    When called for the first time, meas will compute ReliefF for all features
    and the subsequent calls simply return the stored data.

    Class :obj:`Relief` works on discrete and continuous classes and thus 
    implements functionality of algorithms ReliefF and RReliefF.

    .. note::
       ReliefF can also compute the threshold function, that is, the feature
       quality at different thresholds for binarization.


=======================
Measures for Regression
=======================

:obj:`Relief` (described for classification) can be also used for regression.

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
class OrderAttributesByMeasure:
    """Construct an instance that orders features by their scores.
    
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

def MeasureAttribute_Distance(attr=None, data=None):
    """Instantiate :obj:`MeasureAttribute_DistanceClass` and use it to return
    the score of a given feature on given data.
    
    :param attr: feature to score
    :type attr: Orange.data.variable
    
    :param data: data table used for feature scoring
    :type data: Orange.data.table 
    
    """
    m = MeasureAttribute_DistanceClass()
    if attr != None and data != None:
        return m(attr, data)
    else:
        return m

class MeasureAttribute_DistanceClass(orange.MeasureAttribute):
    """The 1-D feature distance measure described in Kononenko."""

    def __call__(self, attr, data, aprioriDist=None, weightID=None):
        """Take :obj:`Orange.data.table` data table and score the given 
        :obj:`Orange.data.variable`.

        :param attr: feature to score
        :type attr: Orange.data.variable

        :param data: a data table used to score features
        :type data: Orange.data.table

        :param aprioriDist: 
        :type aprioriDist:
        
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

def MeasureAttribute_MDL(attr=None, data=None):
    """Instantiate :obj:`MeasureAttribute_MDLClass` and use it n given data to
    return the feature's score."""
    m = MeasureAttribute_MDLClass()
    if attr != None and data != None:
        return m(attr, data)
    else:
        return m

class MeasureAttribute_MDLClass(orange.MeasureAttribute):
    """Score feature based on the minimum description length principle."""

    def __call__(self, attr, data, aprioriDist=None, weightID=None):
        """Take :obj:`Orange.data.table` data table and score the given 
        :obj:`Orange.data.variable`.

        :param attr: feature to score
        :type attr: Orange.data.variable

        :param data: a data table used to score the feature
        :type data: Orange.data.table

        :param aprioriDist: 
        :type aprioriDist:
        
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
