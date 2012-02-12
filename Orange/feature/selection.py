__docformat__ = 'restructuredtext'

import Orange.core as orange

from Orange.feature.scoring import score_all


def best_n(scores, N):
    """Return the best N features (without scores) from the list returned
    by :obj:`Orange.feature.scoring.score_all`.

    :param scores: a list such as the one returned by
      :obj:`Orange.feature.scoring.score_all`
    :type scores: list
    :param N: number of features to select.
    :type N: int
    :rtype: :obj:`list`

    """
    return [x[0] for x in sorted(scores)[:N]]

bestNAtts = best_n


def above_threshold(scores, threshold=0.0):
    """Return features (without scores) from the list returned by
    :obj:`Orange.feature.scoring.score_all` with score above or
    equal to a specified threshold.

    :param scores: a list such as one returned by
      :obj:`Orange.feature.scoring.score_all`
    :type scores: list
    :param threshold: threshold for selection. Defaults to 0.
    :type threshold: float
    :rtype: :obj:`list`

    """
    return [x[0] for x in scores if x[1] > threshold]


attsAboveThreshold = above_threshold


def select_best_n(data, scores, N):
    """Construct and return a new data table that includes a
    class and only N best features from a list scores.

    :param data: an example table
    :type data: Orange.data.table
    :param scores: a list such as the one returned by
      :obj:`Orange.feature.scoring.score_all`
    :type scores: list
    :param N: number of features to select
    :type N: int
    :rtype: new data table

    """
    return data.select(best_n(scores, N) + [data.domain.classVar.name])

selectBestNAtts = select_best_n


def select_above_threshold(data, scores, threshold=0.0):
    """Construct and return a new data table that includes a class and
    features from the list returned by
    :obj:`Orange.feature.scoring.score_all` that have the score above or
    equal to a specified threshold.

    :param data: an example table
    :type data: Orange.data.table
    :param scores: a list such as the one returned by
      :obj:`Orange.feature.scoring.score_all`
    :type scores: list
    :param threshold: threshold for selection. Defaults to 0.
    :type threshold: float
    :rtype: new data table

    """
    return data.select(above_threshold(scores, threshold) + \
                       [data.domain.classVar.name])

selectAttsAboveThresh = select_above_threshold


def select_relief(data, measure=orange.MeasureAttribute_relief(k=20, m=50), margin=0):
    """Iteratively remove the worst scored feature until no feature
    has a score below the margin. The filter procedure was originally
    designed for measures such as Relief, which are context dependent,
    i.e., removal of features may change the scores of other remaining
    features. The score is thus recomputed in each iteration.

    :param data: an data table
    :type data: Orange.data.table
    :param measure: a feature scorer (derived from
      :obj:`Orange.feature.scoring.Measure`)
    :param margin: margin for removal. Defaults to 0.
    :type margin: float

    """
    measl = score_all(data, measure)
    while len(data.domain.attributes) > 0 and measl[-1][1] < margin:
        data = select_best_n(data, measl, len(data.domain.attributes) - 1)
        measl = score_all(data, measure)
    return data

filterRelieff = select_relief


class FilterAboveThreshold(object):
    """Store filter parameters that are later called with the data to
    return the data table with only selected features.

    This class uses :obj:`select_above_threshold`.

    :param measure: an attribute measure (derived from
      :obj:`Orange.feature.scoring.Measure`). Defaults to
      :obj:`Orange.feature.scoring.Relief` for k=20 and m=50.
    :param threshold: score threshold for attribute selection. Defaults to 0.
    :type threshold: float

    Some examples of how to use this class are::

        filter = Orange.feature.selection.FilterAboveThreshold(threshold=.15)
        new_data = filter(data)
        new_data = Orange.feature.selection.FilterAboveThreshold(data)
        new_data = Orange.feature.selection.FilterAboveThreshold(data, threshold=.1)
        new_data = Orange.feature.selection.FilterAboveThreshold(data, threshold=.1,
                   measure=Orange.feature.scoring.Gini())

    """
    def __new__(cls, data=None,
                measure=orange.MeasureAttribute_relief(k=20, m=50),
                threshold=0.0):

        if data is None:
            self = object.__new__(cls)
            return self
        else:
            self = cls(measure=measure, threshold=threshold)
            return self(data)

    def __init__(self, measure=orange.MeasureAttribute_relief(k=20, m=50), \
                 threshold=0.0):

        self.measure = measure
        self.threshold = threshold

    def __call__(self, data):
        """Return data table features with scores above given threshold.

        :param data: data table
        :type data: Orange.data.table

        """
        ma = score_all(data, self.measure)
        return select_above_threshold(data, ma, self.threshold)

FilterAttsAboveThresh = FilterAboveThreshold
FilterAttsAboveThresh_Class = FilterAboveThreshold


class FilterBestN(object):
    """Store filter parameters that are later called with the data to
    return the data table with only selected features.

    :param measure: an attribute measure (derived from
      :obj:`Orange.feature.scoring.Measure`). Defaults to
      :obj:`Orange.feature.scoring.Relief` for k=20 and m=50.
    :param n: number of best features to return. Defaults to 5.
    :type n: int

    """
    def __new__(cls, data=None,
                measure=orange.MeasureAttribute_relief(k=20, m=50),
                n=5):

        if data is None:
            self = object.__new__(cls)
            return self
        else:
            self = cls(measure=measure, n=n)
            return self(data)

    def __init__(self, measure=orange.MeasureAttribute_relief(k=20, m=50),
                 n=5):
        self.measure = measure
        self.n = n

    def __call__(self, data):
        ma = score_all(data, self.measure)
        self.n = min(self.n, len(data.domain.attributes))
        return select_best_n(data, ma, self.n)

FilterBestNAtts = FilterBestN
FilterBestNAtts_Class = FilterBestN


class FilterRelief(object):
    """Store filter parameters that are later called with the data to
    return the data table with only selected features.

    :param measure: an attribute measure (derived from
      :obj:`Orange.feature.scoring.Measure`). Defaults to
      :obj:`Orange.feature.scoring.Relief` for k=20 and m=50.
    :param margin: margin for Relief scoring. Defaults to 0.
    :type margin: float

    """
    def __new__(cls, data=None,
                measure=orange.MeasureAttribute_relief(k=20, m=50),
                margin=0):

        if data is None:
            self = object.__new__(cls)
            return self
        else:
            self = cls(measure=measure, margin=margin)
            return self(data)

    def __init__(self, measure=orange.MeasureAttribute_relief(k=20, m=50),
                 margin=0):
        self.measure = measure
        self.margin = margin

    def __call__(self, data):
        return select_relief(data, self.measure, self.margin)

FilterRelief_Class = FilterRelief

##############################################################################
# wrapped learner


class FilteredLearner(object):
    """Return the learner that wraps :obj:`Orange.classification.baseLearner` 
    and a data selection method.

    When calling the learner with a data table, data is first filtered and
    then passed to :obj:`Orange.classification.baseLearner`. This comes handy
    when one wants to test the schema of feature-subset-selection-and-learning
    by some repetitive evaluation method, e.g., cross validation.

    :param filter: defatuls to
      :obj:`Orange.feature.selection.FilterAboveThreshold`
    :type filter: Orange.feature.selection.FilterAboveThreshold

    Here is an example of how to build a wrapper around naive Bayesian learner
    and use it on a data set::

        nb = Orange.classification.bayes.NaiveBayesLearner()
        learner = Orange.feature.selection.FilteredLearner(nb,
            filter=Orange.feature.selection.FilterBestN(n=5), name='filtered')
        classifier = learner(data)

    """
    def __new__(cls, baseLearner, data=None, weight=0,
                filter=FilterAboveThreshold(), name='filtered'):

        if data is None:
            self = object.__new__(cls)
            return self
        else:
            self = cls(baseLearner, filter=filter, name=name)
            return self(data, weight)

    def __init__(self, baseLearner, filter=FilterAboveThreshold(),
                 name='filtered'):
        self.baseLearner = baseLearner
        self.filter = filter
        self.name = name

    def __call__(self, data, weight=0):
        # filter the data and then learn
        fdata = self.filter(data)
        model = self.baseLearner(fdata, weight)
        return FilteredClassifier(classifier=model, domain=model.domain)

FilteredLearner_Class = FilteredLearner


class FilteredClassifier:
    """A classifier returned by FilteredLearner."""
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example, resultType=orange.GetValue):
        return self.classifier(example, resultType)

    def atts(self):
        return self.domain.attributes
