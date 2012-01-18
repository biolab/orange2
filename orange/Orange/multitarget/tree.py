"""
.. index:: Multi-target Tree Learner

***************************************
Multi-target Tree Learner
***************************************

To use the tree learning algorithm for multi-target data, standard
orange trees (:class:`Orange.classification.tree.TreeLearner`) can be used.
Only the :obj:`~Orange.classification.tree.TreeLearner.measure` for feature
scoring and the :obj:`~Orange.classification.tree.TreeLearner.node_learner`
components have to be chosen so that they work on multi-target data domains.

This module provides one such measure (:class:`MultitargetVariance`) that
can be used and a helper class :class:`MultiTreeLearner` which extends
:class:`~Orange.classification.tree.TreeLearner` and is the same in all
aspects except for different (multi-target) defaults for
:obj:`~Orange.classification.tree.TreeLearner.measure` and
:obj:`~Orange.classification.tree.TreeLearner.node_learner`.

Examples
========

The following example demonstrates how to build a prediction model with
MultitargetTreeLearner and use it to predict (multiple) class values for
a given instance (:download:`multitarget.py <code/multitarget.py>`,
uses :download:`emotions.tab <code/emotions.tab>`):

.. literalinclude:: code/multitarget.py
   :lines: 1-4, 10-12


.. index:: Multi-target Variance 
.. autoclass:: Orange.multitarget.tree.MultitargetVariance
   :members:
   :show-inheritance:

.. index:: Multi-target Tree Learner
.. autoclass:: Orange.multitarget.tree.MultiTreeLearner
   :members:
   :show-inheritance:

.. index:: Multi-target Tree Classifier
.. autoclass:: Orange.multitarget.tree.MultiTree
   :members:
   :show-inheritance:

"""

from operator import itemgetter

import numpy as np
import Orange


def weighted_variance(X, weights=None):
    """Computes the variance using a weighted distance to the centroid."""
    if not weights:
        weights = [1] * len(X[0])
    X = X * np.array(weights)
    return np.sum(np.sum((X - np.mean(X, 0))**2, 1))

class MultitargetVariance(Orange.feature.scoring.Score):
    """
    A multi-target score that ranks features based on the variance of the
    subsets. A weighted distance can be used to compute the variance.
    """

    def __init__(self, weights=None):
        """
        :param weights: Weights of the features used when computing distances.
                        If None, all weights are set to 1.
        :type weigts: list
        """

        # Types of classes allowed
        self.handles_discrete = True
        ### TODO: for discrete classes with >2 values entropy should be used instead of variance
        self.handles_continuous = True
        # Can handle continuous features
        self.computes_thresholds = True
        # Needs instances
        self.needs = Orange.feature.scoring.Score.Generator

        self.weights = weights


    def threshold_function(self, feature, data, cont_distrib=None, weightID=0):
        """
        Evaluates possible splits of a continuous feature into a binary one
        and scores them.
        
        :param feature: Continuous feature to be split.
        :type feature: :class:`Orange.data.variable`

        :param data: The data set to be split using the given continuous feature.
        :type data: :class:`Orange.data.Table`

        :return: :obj:`list` of :obj:`tuples <tuple>` [(threshold, score, None),]
        """

        f = data.domain[feature]
        values = sorted(set(ins[f].value for ins in data))
        ts = [(v1 + v2) / 2. for v1, v2 in zip(values, values[1:])]
        if len(ts) > 40:
            ts = ts[::len(ts)/20]
        scores = []
        for t in ts:
            bf = Orange.feature.discretization.IntervalDiscretizer(
                points=[t]).construct_variable(f)
            dom2 = Orange.data.Domain([bf], class_vars=data.domain.class_vars)
            data2 = Orange.data.Table(dom2, data)
            scores.append((t, self.__call__(bf, data2)))
        return scores

    def best_threshold(self, feature, data):
        """
        Computes the best threshold for a split of a continuous feature.

        :param feature: Continuous feature to be split.
        :type feature: :class:`Orange.data.variable`

        :param data: The data set to be split using the given continuous feature.
        :type data: :class:`Orange.data.Table`

        :return: :obj:`tuple` (threshold, score, None)
        """

        scores = self.threshold_function(feature, data)
        threshold, score = max(scores, key=itemgetter(1))
        return (threshold, score, None)

    def __call__(self, feature, data, apriori_class_distribution=None, weightID=0):
        """
        :param feature: The feature to be scored.
        :type feature: :class:`Orange.data.variable`

        :param data: The data set on which to score the feature.
        :type data: :class:`Orange.data.Table`

        :return: :obj:`float`
        """

        split = dict((ins[feature].value, []) for ins in data)
        for ins in data:
            split[ins[feature].value].append(ins.get_classes())
        score = -sum(weighted_variance(x, self.weights) * len(x) for x in split.values())
        return score


class MultiTreeLearner(Orange.classification.tree.TreeLearner):
    """
    MultiTreeLearner is a multi-target version of a tree learner. It is the
    same as :class:`~Orange.classification.tree.TreeLearner`, except for the
    default values of two parameters:
    
    .. attribute:: measure
        
        A multi-target score is used by default: :class:`MultitargetVariance`.

    .. attribute:: node_learner
        
        Standard trees use :class:`~Orange.classification.majority.MajorityLearner`
        to construct prediction models in the leaves of the tree.
        MultiTreeLearner uses the multi-target equivalent which can be 
        obtained simply by wrapping the majority learner:

        :class:`Orange.multitarget.MultitargetLearner` (:class:`Orange.classification.majority.MajorityLearner()`).

    """

    def __init__(self, **kwargs):
        """
        The constructor simply passes all given arguments to
        :class:`~Orange.classification.tree.TreeLearner`'s constructor
        :obj:`Orange.classification.tree.TreeLearner.__init__`.
        """
        
        measure = MultitargetVariance()
        node_learner = Orange.multitarget.MultitargetLearner(
            Orange.classification.majority.MajorityLearner())
        Orange.classification.tree.TreeLearner.__init__(
            self, measure=measure, node_learner=node_learner, **kwargs)

    def __call__(self, data, weight=0):
        """
        :param data: Data instances to learn from.
        :type data: :class:`Orange.data.Table`

        :param weight: Id of meta attribute with weights of instances.
        :type weight: :obj:`int`
        """
        
        # TreeLearner does not work on class-less domains,
        # so we set the class if necessary
        if data.domain.class_var is None:
            data2 = Orange.data.Table(Orange.data.Domain(
                data.domain.attributes, data.domain.class_vars[0],
                class_vars=data.domain.class_vars), data)
        tree = Orange.classification.tree.TreeLearner.__call__(
            self, data2, weight)
        return MultiTree(base_classifier=tree)

class MultiTree(Orange.classification.tree.TreeClassifier):
    """
    MultiTree classifier is almost the same as the base class it extends
    (:class:`~Orange.classification.tree.TreeClassifier`). Only the
    :obj:`__call__` method is modified so it works with multi-target data.
    """

    def __call__(self, instance, return_type=Orange.core.GetValue):
        """
        :param instance: Instance to be classified.
        :type instance: :class:`Orange.data.Instance`

        :param return_type: One of
            :class:`Orange.classification.Classifier.GetValue`,
            :class:`Orange.classification.Classifier.GetProbabilities` or
            :class:`Orange.classification.Classifier.GetBoth`
        """

        node = self.descender(self.tree, instance)[0]
        return node.node_classifier(instance, return_type)


if __name__ == '__main__':
    data = Orange.data.Table('test-pls')
    print 'Actual classes:\n', data[0].get_classes()
    
    majority = Orange.classification.majority.MajorityLearner()
    mt_majority = Orange.multitarget.MultitargetLearner(majority)
    c_mtm = mt_majority(data)
    print 'Majority predictions:\n', c_mtm(data[0])

    mt_tree = MultiTreeLearner(max_depth=3)
    c_mtt = mt_tree(data)
    print 'Multi-target Tree predictions:\n', c_mtt(data[0])
