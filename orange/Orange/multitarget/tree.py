from operator import itemgetter

import numpy as np
import Orange
from Orange.multitarget import data_label_mask


def weighted_variance(X, weights=None):
    """Computes the variance using a weighted distance between vectors"""
    if weights:
        X = X * np.array(weights)
    return np.sum((X - np.mean(X, 0))**2, 1).sum()

class MultitargetVariance(Orange.feature.scoring.Score):
    def __init__(self, weights=None):
        # Types of classes allowed
        self.handles_discrete = True
        self.handles_continuous = True
        # Can handle continuous features
        self.computes_thresholds = True
        # Needs instances
        self.needs = Orange.feature.scoring.Score.Generator

        self.weights = weights


    def threshold_function(self, feature, data, cont_distrib=None, weightID=0):
        f = data.domain[feature]
        label_mask = data_label_mask(data.domain)
        classes = [v for v, label in zip(data.domain, label_mask) if label]
        values = sorted(set(ins[f].value for ins in data))
        ts = [(v1 + v2) / 2. for v1, v2 in zip(values, values[1:])]
        if len(ts) > 40:
            ts = ts[::len(ts)/20]
        scores = []
        for t in ts:
            bf = Orange.feature.discretization.IntervalDiscretizer(
                points=[t]).construct_variable(f)
            data2 = data.select([bf] + classes)
            scores.append((t, self.__call__(bf, data2)))
        return scores

    def best_threshold(self, feature, data):
        scores = self.threshold_function(feature, data)
        threshold, score = max(scores, key=itemgetter(1))
        return (threshold, score, None)

    def __call__(self, feature, data, apriori_class_distribution=None, weightID=0):
        if data.domain[feature].attributes.has_key('label'):
            return float('-inf')
        label_mask = data_label_mask(data.domain)
        classes = [v for v, label in zip(data.domain, label_mask) if label]
        split = dict((ins[feature].value, []) for ins in data)
        for ins in data:
            # TODO: does not work when there are missing class values
            split[ins[feature].value].append([float(ins[c]) for c in classes])
        score = -sum(weighted_variance(x, self.weights) * len(x) for x in split.values())
        return score


class MultiTreeLearner(Orange.classification.tree.TreeLearner):
    """
    MultiTreeLearner is a multitarget equivalent of the TreeLearner.
    It is the same as Orange.classification.tree.TreeLearner, except for
    the default values of two parameters:
    measure: MultitargetVariance
    node_learner: Orange.multitarget.MultitargetLearner(Orange.classification.majority.MajorityLearner())
    """

    def __init__(self, measure=MultitargetVariance(), 
                 node_learner=Orange.multitarget.MultitargetLearner(
                     Orange.classification.majority.MajorityLearner()),
                 **kwargs):
        Orange.classification.tree.TreeLearner.__init__(
            self, measure=measure, node_learner=node_learner, **kwargs)

    def __call__(self, data, weight=0):
        # TreeLearner does not work on class-less domains,
        # so we set the class if necessary
        if data.domain.class_var is None:
            for var in data.domain:
                if var.attributes.has_key('label'):
                    data = Orange.data.Table(Orange.data.Domain(data.domain, var),
                                             data)
                    break

        tree = Orange.classification.tree.TreeLearner.__call__(self, data, weight)
        return MultiTree(base_classifier=tree)

class MultiTree(Orange.classification.tree.TreeClassifier):
    """MultiTree classifier"""

    def __call__(self, instance, return_type=Orange.core.GetValue):
        node = self.descender(self.tree, instance)[0]
        return node.node_classifier(instance, return_type)

