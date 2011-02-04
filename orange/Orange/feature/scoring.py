"""

.. index:: feature scoring

.. index:: 
   single: feature; feature scoring

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

============
Base Classes
============

There are a number of different measures for assessing the relevance of 
features with respect to much information they contain about the 
corresponding class. These procedures are also known as feature scoring. 
Orange implements several methods that all stem from
:obj:`Orange.feature.scoring.Measure`. The most of common ones compute
certain statistics on conditional distributions of class values given
the feature values; in Orange, these are derived from
:obj:`Orange.feature.scoring.MeasureAttributeFromProbabilities`.

.. class:: Measure

    This is the base class for a wide range of classes that measure quality of
    features. The class itself is, naturally, abstract. Its fields merely
    describe what kinds of features it can handle and what kind of data it 
    requires.

    .. attribute:: handlesDiscrete
    
    Tells whether the measure can handle discrete features.

    .. attribute:: handlesContinuous
    
    Tells whether the measure can handle continuous features.

    .. attribute:: computesThresholds
    
    Tells whether the measure implements the :obj:`thresholdFunction`.

    .. attribute:: needs
    
    Tells what kind of data the measure needs. This can be either 
    :obj:`NeedsGenerator`, :obj:`NeedsDomainContingency`, 
    :obj:`NeedsContingency_Class`. The first need an instance generator
    (Relief is an example of such measure), the second can compute the quality
    from :obj:`Orange.statistics.distributions.DomainContingency` and the
    latter only needs the contingency
    (:obj:`Orange.statistics.distributions.ContingencyAttrClass`) the 
    feature distribution and the apriori class distribution. Most measures
    only need the latter.

    Several (but not all) measures can treat unknown feature values in
    different ways, depending on field :obj:`unknownsTreatment` (this field is
    not defined in :obj:`Measure` but in many derived classes). Undefined 
    values can be:
    
    * ignored (:obj:`Measure.IgnoreUnknowns`); this has the same effect as if 
      the example for which the feature value is unknown are removed.

    * punished (:obj:`Measure.ReduceByUnknown`); the feature quality is
      reduced by the proportion of unknown values. In impurity measures, this
      can be interpreted as if the impurity is decreased only on examples for
      which the value is defined and stays the same for the others, and the
      feature quality is the average impurity decrease.
      
    * imputed (:obj:`Measure.UnknownsToCommon`); here, undefined values are
      replaced by the most common feature value. If you want a more clever
      imputation, you should do it in advance.

    * treated as a separate value (:obj:`MeasureAttribute.UnknownsAsValue`)

    The default treatment is :obj:`ReduceByUnknown`, which is optimal in most
    cases and does not make additional presumptions (as, for instance,
    :obj:`UnknownsToCommon` which supposes that missing values are not for
    instance, results of measurements that were not performed due to
    information extracted from the other features). Use other treatments if
    you know that they make better sense on your data.

    The only method supported by all measures is the call operator to which we
    pass the data and get the number representing the quality of the feature.
    The number does not have any absolute meaning and can vary widely for
    different feature measures. The only common characteristic is that
    higher the value, better the feature. If the feature is so bad that 
    it's quality cannot be measured, the measure returns
    :obj:`Measure.Rejected`. None of the measures described here do so.

    There are different sets of arguments that the call operator can accept.
    Not all classes will accept all kinds of arguments. Relief, for instance,
    cannot be computed from contingencies alone. Besides, the feature and
    the class need to be of the correct type for a particular measure.

    There are three call operators just to make your life simpler and faster.
    When working with the data, your method might have already computed, for
    instance, contingency matrix. If so and if the quality measure you use is
    OK with that (as most measures are), you can pass the contingency matrix
    and the measure will compute much faster. If, on the other hand, you only
    have examples and haven't computed any statistics on them, you can pass
    examples (and, optionally, an id for meta-feature with weights) and the
    measure will compute the contingency itself, if needed.

    .. method:: __call__(attribute, examples[, apriori class distribution][, weightID])
    .. method:: __call__(attribute, domain contingency[, apriori class distribution])
    .. method:: __call__(contingency, class distribution[, apriori class distribution])

        :param attribute: gives the feature whose quality is to be assessed.
          This can be either a descriptor, an index into domain or a name. In
          the first form, if the feature is given by descriptor, it doesn't
          need to be in the domain. It needs to be computable from the
          feature in the domain, though.
          
        Data is given either as examples (and, optionally, id for 
        meta-feature with weight), domain contingency
        (:obj:`Orange.statistics.distributions.DomainContingency`) (a list of
        contingencies) or distribution (:obj:`Orange.statistics.distributions`)
        matrix and :obj:`Orange.statistics.distributions.Distribution`. If 
        you use the latter form, what you should give as the class distribution
        depends upon what you do with unknown values (if there are any).
        If :obj:`unknownsTreatment` is :obj:`IgnoreUnknowns`, the class
        distribution should be computed on examples for which the feature
        value is defined. Otherwise, class distribution should be the overall
        class distribution.

        The optional argument with apriori class distribution is
        most often ignored. It comes handy if the measure makes any probability
        estimates based on apriori class probabilities (such as m-estimate).

    .. method:: thresholdFunction(attribute, examples[, weightID])
    
    This function computes the qualities for different binarizations of the
    continuous feature :obj:`attribute`. The feature should of course be
    continuous. The result of a function is a list of tuples, where the first
    element represents a threshold (all splits in the middle between two
    existing feature values), the second is the measured quality for a
    corresponding binary feature and the last one is the distribution which
    gives the number of examples below and above the threshold. The last
    element, though, may be missing; generally, if the particular measure can
    get the distribution without any computational burden, it will do so and
    the caller can use it. If not, the caller needs to compute it itself.

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

    It is also possible to assess the quality of features that do not exist
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
    `Orange.feature.scoring.Relief().thresholdFunction("petal length", data)`.
    This is not recommendable for ReliefF, since it may be a lot slower.

    The script below finds and prints out the best threshold for binarization
    of an feature, that is, the threshold with which the resulting binary
    feature will have the optimal ReliefF (or any other measure)::

        thresh, score, distr = meas.bestThreshold("petal length", data)
        print "Best threshold: %5.3f (score %5.3f)" % (thresh, score)

.. class:: MeasureAttributeFromProbabilities

    This is the abstract base class for feature quality measures that can be
    computed from contingency matrices only. It relieves the derived classes
    from having to compute the contingency matrix by defining the first two
    forms of call operator. (Well, that's not something you need to know if
    you only work in Python.) Additional feature of this class is that you can
    set probability estimators. If none are given, probabilities and
    conditional probabilities of classes are estimated by relative frequencies.

    .. attribute:: unknownsTreatment
     
    Defines what to do with unknown values. See the possibilities described above.

    .. attribute:: estimatorConstructor
    .. attribute:: conditionalEstimatorConstructor
    
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

The following section describes the feature quality measures suitable for 
discrete features and outcomes. 
See  `scoring-info-lenses.py`_, `scoring-info-iris.py`_,
`scoring-diff-measures.py`_ and `scoring-regression.py`_
for more examples on their use.

.. index:: 
   single: feature scoring; information gain

.. class:: InfoGain

    The most popular measure, information gain :obj:`Info` measures the expected
    decrease of the entropy.

.. index:: 
   single: feature scoring; gain ratio

.. class:: GainRatio

    Gain ratio :obj:`GainRatio` was introduced by Quinlan in order to avoid
    overestimation of multi-valued features. It is computed as information
    gain divided by the entropy of the feature's value. (It has been shown,
    however, that such measure still overstimates the features with multiple
    values.)

.. index:: 
   single: feature scoring; gini index

.. class:: Gini

    Gini index :obj:`Gini` was first introduced by Breiman and can be interpreted
    as the probability that two randomly chosen examples will have different
    classes.

.. index:: 
   single: feature scoring; relevance

.. class:: Relevance

    Relevance of features :obj:`Relevance` is a measure that discriminate
    between features on the basis of their potential value in the formation of
    decision rules.

.. index:: 
   single: feature scoring; cost

.. class:: Cost

    Evaluates features based on the "saving" achieved by knowing the value of
    feature, according to the specified cost matrix.

    .. attribute:: cost
     
    Cost matrix, see :obj:`Orange.classification.CostMatrix` for details.

    If cost of predicting the first class for an example that is actually in
    the second is 5, and the cost of the opposite error is 1, than an appropriate
    measure can be constructed and used for feature 3 as follows::

        >>> meas = Orange.feature.scoring.Cost()
        >>> meas.cost = ((0, 5), (1, 0))
        >>> meas(3, data)
        0.083333350718021393

    This tells that knowing the value of feature 3 would decrease the
    classification cost for appx 0.083 per example.

.. index:: 
   single: feature scoring; ReliefF

.. class:: Relief

    ReliefF :obj:`Relief` was first developed by Kira and Rendell and then
    substantially generalized and improved by Kononenko. It measures the
    usefulness of features based on their ability to distinguish between
    very similar examples belonging to different classes.

    .. attribute:: k
    
    Number of neighbours for each example. Default is 5.

    .. attribute:: m
    
    Number of reference examples. Default is 100. Set to -1 to take all the
    examples.

    .. attribute:: checkCachedData
    
    A flag best left alone unless you know what you do.

Computation of ReliefF is rather slow since it needs to find k nearest
neighbours for each of m reference examples (or all examples, if m is set to
-1). Since we normally compute ReliefF for all features in the dataset,
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

Finally, here is an example which shows what can happen if you disable the 
computation of checksums::

    table = Orange.data.Table("iris")
    r1 = Orange.feature.scoring.Relief()
    r2 = Orange.feature.scoring.Relief(checkCachedData = False)

    print "%.3f\\t%.3f" % (r1(0, table), r2(0, table))
    for ex in table:
        ex[0] = 0
    print "%.3f\\t%.3f" % (r1(0, table), r2(0, table))

The first print prints out the same number, 0.321 twice. Then we annulate the
first feature. r1 notices it and returns -1 as it's ReliefF,
while r2 does not and returns the same number, 0.321, which is now wrong.

======================
Measure for Regression
======================

Except for ReliefF, the only feature quality measure available for regression
problems is based on a mean square error.

.. index:: 
   single: feature scoring; mean square error

.. class:: MSE

    Implements the mean square error measure.

    .. attribute:: unknownsTreatment
    
    Tells what to do with unknown feature values. See description on the top
    of this page.

    .. attribute:: m
    
    Parameter for m-estimate of error. Default is 0 (no m-estimate).

==========
References
==========

* Kononeko: Strojno ucenje. Zalozba FE in FRI, Ljubljana, 2005.

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
        :type weight: Orange.data.Feature

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
