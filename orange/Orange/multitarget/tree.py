from operator import itemgetter

import numpy as np
import Orange


def weighted_variance(X, weights=None):
    """Computes the variance using a weighted distance between vectors"""
    global foo
    foo = X

    if not weights:
        weights = [1] * len(X[0])

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
        scores = self.threshold_function(feature, data)
        threshold, score = max(scores, key=itemgetter(1))
        return (threshold, score, None)

    def __call__(self, feature, data, apriori_class_distribution=None, weightID=0):
        split = dict((ins[feature].value, []) for ins in data)
        for ins in data:
            split[ins[feature].value].append(ins.get_classes())
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
            data2 = Orange.data.Table(Orange.data.Domain(
                data.domain.attributes, data.domain.class_vars[0],
                class_vars=data.domain.class_vars), data)
        tree = Orange.classification.tree.TreeLearner.__call__(self, data2, weight)
        return MultiTree(base_classifier=tree)

class MultiTree(Orange.classification.tree.TreeClassifier):
    """MultiTree classifier"""

    def __call__(self, instance, return_type=Orange.core.GetValue):
        node = self.descender(self.tree, instance)[0]
        return node.node_classifier(instance, return_type)


if __name__ == '__main__':
    data = Orange.data.Table('emotions')
    print 'Actual classes:\n', data[0].get_classes()
    mt = MultiTreeLearner(max_depth=2)
    c = mt(data)
    print 'Predicted classes:\n', c(data[0])

