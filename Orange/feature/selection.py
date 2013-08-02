__docformat__ = 'restructuredtext'

from operator import itemgetter

import Orange.data
import Orange.core as orange

from Orange.feature.scoring import score_all


def _select_features_subset(data, features):
    """Select the `features` from the `data`.

    .. note::
        The `features` must be a subset of the `data.domain.features`.

    """
    def as_descriptor(arg):
        """Ensure `arg` is an descriptor from `data.domain`"""
        return data.domain[arg]

    features = map(as_descriptor, features)
    domain = Orange.data.Domain(features, data.domain.class_var,
                                class_vars=data.domain.class_vars)
    domain.add_metas(data.domain.get_metas())
    return Orange.data.Table(domain, data)


def top_rated(scores, n, highest_best=True):
    """Return n top-rated features from the list of scores.

    :param list scores:
        A list such as the one returned by :func:`.score_all`
    :param int n: Number of features to select.
    :param bool highest_best:
        If true, the features that are scored higher are preferred.
    :rtype: :obj:`list`

    """
    return [f for f, score in
            sorted(scores, key=itemgetter(1), reverse=highest_best)[:n]]

bestNAtts = top_rated


def above_threshold(scores, threshold=0.0):
    """Return features (without scores) with scores above or
    equal to a specified threshold.

    :param list scores:
        A list such as one returned by :func:`.score_all`
    :param float threshold: Threshold for selection.
    :rtype: :obj:`list`

    """
    return [f for f, score in scores if score >= threshold]


attsAboveThreshold = above_threshold


def select(data, scores, n):
    """Construct and return a new data table that includes a
    class and only the best features from a list scores.

    :param data: a data table
    :type data: :obj:`Orange.data.Table`
    :param scores: a list such as the one returned by
      :obj:`~Orange.feature.scoring.score_all`
    :type scores: list
    :param n: number of features to select
    :type n: int
    :rtype: :obj:`Orange.data.Table`

    """
    features = top_rated(scores, n)
    return _select_features_subset(data, features)

selectBestNAtts = select
select_best_n = select


def select_above_threshold(data, scores, threshold=0.0):
    """Construct and return a new data table that includes a class and
    features from the list returned by
    :obj:`~Orange.feature.scoring.score_all` with higher or equal score
    to a given threshold.

    :param data: a data table
    :type data: :obj:`Orange.data.Table`
    :param scores: a list such as the one returned by
      :obj:`~Orange.feature.scoring.score_all`
    :type scores: list
    :param threshold: threshold for selection
    :type threshold: float
    :rtype: :obj:`Orange.data.Table`
    """
    features = above_threshold(scores, threshold)
    return _select_features_subset(data, features)

selectAttsAboveThresh = select_above_threshold


def select_relief(data, measure=orange.MeasureAttribute_relief(k=20, m=50), margin=0):
    """Iteratively remove the worst scored feature until no feature
    has a score below the margin. The filter procedure was originally
    designed for measures such as Relief, which are context dependent,
    i.e., removal of features may change the scores of other remaining
    features. The score is thus recomputed in each iteration.

    :param data: a data table
    :type data: :obj:`Orange.data.Table`
    :param measure: a feature scorer
    :type measure: :obj:`Orange.feature.scoring.Score`
    :param margin: margin for removal
    :type margin: float

    """
    measl = score_all(data, measure)
    while len(data.domain.attributes) > 0 and measl[-1][1] < margin:
        data = select(data, measl, len(data.domain.attributes) - 1)
        measl = score_all(data, measure)
    return data

filterRelieff = select_relief


class FilterAboveThreshold(object):
    """A wrapper around :obj:`select_above_threshold`; the
    constructor stores the parameters of the feature selection
    procedure that are then applied when the the selection
    is called with the actual data.

    :param measure: a feature scorer
    :type measure: :obj:`Orange.feature.scoring.Score`
    :param threshold: threshold for selection. Defaults to 0.
    :type threshold: float
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
        """Return data table features that have scores above given
        threshold.

        :param data: data table
        :type data: Orange.data.Table

        """
        ma = score_all(data, self.measure)
        return select_above_threshold(data, ma, self.threshold)

FilterAttsAboveThresh = FilterAboveThreshold
FilterAttsAboveThresh_Class = FilterAboveThreshold


class FilterBestN(object):
    """A wrapper around :obj:`select`; the
    constructor stores the filter parameters that are applied when the
    function is called.

    :param measure: a feature scorer
    :type measure: :obj:`Orange.feature.scoring.Score`
    :param n: number of features to select
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
        return select(data, ma, self.n)

FilterBestNAtts = FilterBestN
FilterBestNAtts_Class = FilterBestN


class FilterRelief(object):
    """A class wrapper around :obj:`select_best_n`; the
    constructor stores the filter parameters that are applied when the
    function is called.

    :param measure: a feature scorer
    :type measure: :obj:`Orange.feature.scoring.Score`
    :param margin: margin for Relief scoring
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
    """A feature selection wrapper around base learner. When provided data,
     this learner applies a given feature selection method and then calls
     the base learner.

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
