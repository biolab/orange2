"""
#########################
Selection (``selection``)
#########################

.. index:: feature selection

.. index::
   single: feature; feature selection

Some machine learning methods perform better if they learn only from a
selected subset of the most informative or "best" features.

This so-called filter approach can boost the performance
of learner in terms of predictive accuracy, speed-up induction and
simplicity of resulting models. Feature scores are estimated before
modeling, without knowing  which machine learning method will be
used to construct a predictive model.

:download:`Example script:<code/selection-best3.py>`

.. literalinclude:: code/selection-best3.py
    :lines: 7-

The script should output this::

    Best 3 features:
    physician-fee-freeze
    el-salvador-aid
    synfuels-corporation-cutback

.. autoclass:: Orange.feature.selection.FilterAboveThreshold
   :members:

.. autoclass:: Orange.feature.selection.FilterBestN
   :members:

.. autoclass:: Orange.feature.selection.FilterRelief
   :members:

.. autoclass:: Orange.feature.selection.FilteredLearner
   :members:

.. autoclass:: Orange.feature.selection.FilteredClassifier
   :members:

These functions support the design of feature subset selection for
classification problems.

.. automethod:: Orange.feature.selection.best_n

.. automethod:: Orange.feature.selection.above_threshold

.. automethod:: Orange.feature.selection.select_best_n

.. automethod:: Orange.feature.selection.select_above_threshold

.. automethod:: Orange.feature.selection.select_relief

.. rubric:: Examples

The following script defines a new Naive Bayes classifier, that
selects five best features from the data set before learning.
The new classifier is wrapped-up in a special class (see
<a href="../ofb/c_pythonlearner.htm">Building your own learner</a>
lesson in <a href="../ofb/default.htm">Orange for Beginners</a>). The
script compares this filtered learner with one that uses a complete
set of features.

:download:`selection-bayes.py<code/selection-bayes.py>`

.. literalinclude:: code/selection-bayes.py
    :lines: 7-

Interestingly, and somehow expected, feature subset selection
helps. This is the output that we get::

    Learner      CA
    Naive Bayes  0.903
    with FSS     0.940

We can do all of  he above by wrapping the learner using
<code>FilteredLearner</code>, thus
creating an object that is assembled from data filter and a base learner. When
given a data table, this learner uses attribute filter to construct a new
data set and base learner to construct a corresponding
classifier. Attribute filters should be of the type like
<code>orngFSS.FilterAboveThresh</code> or
<code>orngFSS.FilterBestN</code> that can be initialized with the
arguments and later presented with a data, returning new reduced data
set.

The following code fragment replaces the bulk of code
from previous example, and compares naive Bayesian classifier to the
same classifier when only a single most important attribute is
used.

:download:`selection-filtered-learner.py<code/selection-filtered-learner.py>`

.. literalinclude:: code/selection-filtered-learner.py
    :lines: 13-16

Now, let's decide to retain three features (change the code in <a
href="fss4.py">fss4.py</a> accordingly!), but observe how many times
an attribute was used. Remember, 10-fold cross validation constructs
ten instances for each classifier, and each time we run
FilteredLearner a different set of features may be
selected. <code>orngEval.CrossValidation</code> stores classifiers in
<code>results</code> variable, and <code>FilteredLearner</code>
returns a classifier that can tell which features it used (how
convenient!), so the code to do all this is quite short.

.. literalinclude:: code/selection-filtered-learner.py
    :lines: 25-

Running :download:`selection-filtered-learner.py <code/selection-filtered-learner.py>` with three features selected each
time a learner is run gives the following result::

    Learner      CA
    bayes        0.903
    filtered     0.956

    Number of times features were used in cross-validation:
     3 x el-salvador-aid
     6 x synfuels-corporation-cutback
     7 x adoption-of-the-budget-resolution
    10 x physician-fee-freeze
     4 x crime

Experiment yourself to see, if only one attribute is retained for
classifier, which attribute was the one most frequently selected over
all the ten cross-validation tests!

==========
References
==========

* K. Kira and L. Rendell. A practical approach to feature selection. In
  D. Sleeman and P. Edwards, editors, Proc. 9th Int'l Conf. on Machine
  Learning, pages 249{256, Aberdeen, 1992. Morgan Kaufmann Publishers.

* I. Kononenko. Estimating attributes: Analysis and extensions of RELIEF.
  In F. Bergadano and L. De Raedt, editors, Proc. European Conf. on Machine
  Learning (ECML-94), pages  171-182. Springer-Verlag, 1994.

* R. Kohavi, G. John: Wrappers for Feature Subset Selection, Artificial
  Intelligence, 97 (1-2), pages 273-324, 1997

"""

__docformat__ = 'restructuredtext'

import Orange.core as orange

from Orange.feature.scoring import score_all


def best_n(scores, N):
    """Return the best N features (without scores) from the list returned
    by :obj:`Orange.feature.scoring.score_all`.

    :param scores: a list such as returned by
      :obj:`Orange.feature.scoring.score_all`
    :type scores: list
    :param N: number of best features to select.
    :type N: int
    :rtype: :obj:`list`

    """
    return map(lambda x:x[0], scores[:N])

bestNAtts = best_n


def above_threshold(scores, threshold=0.0):
    """Return features (without scores) from the list returned by
    :obj:`Orange.feature.scoring.score_all` with score above or
    equal to a specified threshold.

    :param scores: a list such as one returned by
      :obj:`Orange.feature.scoring.score_all`
    :type scores: list
    :param threshold: score threshold for attribute selection. Defaults to 0.
    :type threshold: float
    :rtype: :obj:`list`

    """
    pairs = filter(lambda x, t=threshold: x[1] > t, scores)
    return map(lambda x: x[0], pairs)

attsAboveThreshold = above_threshold


def select_best_n(data, scores, N):
    """Construct and return a new set of examples that includes a
    class and only N best features from a list scores.

    :param data: an example table
    :type data: Orange.data.table
    :param scores: a list such as one returned by
      :obj:`Orange.feature.scoring.score_all`
    :type scores: list
    :param N: number of features to select
    :type N: int
    :rtype: :class:`Orange.data.table` holding N best features

    """
    return data.select(best_n(scores, N) + [data.domain.classVar.name])

selectBestNAtts = select_best_n


def select_above_threshold(data, scores, threshold=0.0):
    """Construct and return a new set of examples that includes a class and
    features from the list returned by
    :obj:`Orange.feature.scoring.score_all` that have the score above or
    equal to a specified threshold.

    :param data: an example table
    :type data: Orange.data.table
    :param scores: a list such as one returned by
      :obj:`Orange.feature.scoring.score_all`
    :type scores: list
    :param threshold: score threshold for attribute selection. Defaults to 0.
    :type threshold: float
    :rtype: :obj:`list` first N features (without measures)

    """
    return data.select(above_threshold(scores, threshold) + \
                       [data.domain.classVar.name])

selectAttsAboveThresh = select_above_threshold


def select_relief(data, measure=orange.MeasureAttribute_relief(k=20, m=50), margin=0):
    """Take the data set and use an attribute measure to remove the worst
    scored attribute (those below the margin). Repeats, until no attribute has
    negative or zero score.

    .. note:: Notice that this filter procedure was originally designed for \
    measures such as Relief, which are context dependent, i.e., removal of \
    features may change the scores of other remaining features. Hence the \
    need to re-estimate score every time an attribute is removed.

    :param data: an data table
    :type data: Orange.data.table
    :param measure: an attribute measure (derived from
      :obj:`Orange.feature.scoring.Measure`). Defaults to
      :obj:`Orange.feature.scoring.Relief` for k=20 and m=50.
    :param margin: if score is higher than margin, attribute is not removed.
      Defaults to 0.
    :type margin: float

    """
    measl = score_all(data, measure)
    while len(data.domain.attributes) > 0 and measl[-1][1] < margin:
        data = select_best_n(data, measl, len(data.domain.attributes) - 1)
#        print 'remaining ', len(data.domain.attributes)
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
            self = object.__new__(cls, measure=measure, threshold=threshold)
            return self
        else:
            self = cls(measure=measure, threshold=threshold)
            return self(data)

    def __init__(self, measure=orange.MeasureAttribute_relief(k=20, m=50), \
                 threshold=0.0):

        self.measure = measure
        self.threshold = threshold

    def __call__(self, data):
        """Take data and return features with scores above given threshold.

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
            self = object.__new__(cls, measure=measure, n=n)
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
            self = object.__new__(cls, measure=measure, margin=margin)
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
            self = object.__new__(cls, baseLearner, filter=filter, name=name)
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
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example, resultType=orange.GetValue):
        return self.classifier(example, resultType)

    def atts(self):
        return self.domain.attributes
