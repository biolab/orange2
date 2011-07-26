"""
#########################
Selection (``selection``)
#########################

.. index:: feature selection

.. index:: 
   single: feature; feature selection

Some machine learning methods may perform better if they learn only from a 
selected subset of "best" features. 

The performance of some machine learning method can be improved by learning 
only from a selected subset of data, which includes the most informative or 
"best" features. This so-called filter approaches can boost the performance 
of learner both in terms of predictive accuracy, speed-up induction, and
simplicity of resulting models. Feature scores are estimated prior to the
modelling, that is, without knowing of which machine learning method will be
used to construct a predictive model.

`selection-best3.py`_ (uses `voting.tab`_):

.. literalinclude:: code/selection-best3.py
    :lines: 7-

The script should output this::

    Best 3 features:
    physician-fee-freeze
    el-salvador-aid
    synfuels-corporation-cutback

.. automethod:: Orange.feature.selection.FilterAttsAboveThresh

.. autoclass:: Orange.feature.selection.FilterAttsAboveThresh_Class
   :members:

.. automethod:: Orange.feature.selection.FilterBestNAtts

.. autoclass:: Orange.feature.selection.FilterBestNAtts_Class
   :members:

.. automethod:: Orange.feature.selection.FilterRelief

.. autoclass:: Orange.feature.selection.FilterRelief_Class
   :members:

.. automethod:: Orange.feature.selection.FilteredLearner

.. autoclass:: Orange.feature.selection.FilteredLearner_Class
   :members:

.. autoclass:: Orange.feature.selection.FilteredClassifier
   :members:

These functions support in the design of feature subset selection for
classification problems.

.. automethod:: Orange.feature.selection.bestNAtts

.. automethod:: Orange.feature.selection.attsAboveThreshold

.. automethod:: Orange.feature.selection.selectBestNAtts

.. automethod:: Orange.feature.selection.selectAttsAboveThresh

.. automethod:: Orange.feature.selection.filterRelieff

.. rubric:: Examples

Following is a script that defines a new classifier that is based
on naive Bayes and prior to learning selects five best features from
the data set. The new classifier is wrapped-up in a special class (see
<a href="../ofb/c_pythonlearner.htm">Building your own learner</a>
lesson in <a href="../ofb/default.htm">Orange for Beginners</a>). The
script compares this filtered learner naive Bayes that uses a complete
set of features.

`selection-bayes.py`_ (uses `voting.tab`_):

.. literalinclude:: code/selection-bayes.py
    :lines: 7-

Interestingly, and somehow expected, feature subset selection
helps. This is the output that we get::

    Learner      CA
    Naive Bayes  0.903
    with FSS     0.940

Now, a much simpler example. Although perhaps educational, we can do all of 
the above by wrapping the learner using <code>FilteredLearner</code>, thus 
creating an object that is assembled from data filter and a base learner. When
given the data, this learner uses attribute filter to construct a new
data set and base learner to construct a corresponding
classifier. Attribute filters should be of the type like
<code>orngFSS.FilterAttsAboveThresh</code> or
<code>orngFSS.FilterBestNAtts</code> that can be initialized with the
arguments and later presented with a data, returning new reduced data
set.

The following code fragment essentially replaces the bulk of code
from previous example, and compares naive Bayesian classifier to the
same classifier when only a single most important attribute is
used.

`selection-filtered-learner.py`_ (uses `voting.tab`_):

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

Running `selection-filtered-learner.py`_ with three features selected each
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

.. _selection-best3.py: code/selection-best3.py
.. _selection-bayes.py: code/selection-bayes.py
.. _selection-filtered-learner.py: code/selection-filtered-learner.py
.. _voting.tab: code/voting.tab

"""

__docformat__ = 'restructuredtext'

import Orange.core as orange

from Orange.feature.scoring import attMeasure

# from orngFSS
def bestNAtts(scores, N):
    """Return the best N features (without scores) from the list returned
    by function :obj:`Orange.feature.scoring.attMeasure`.
    
    :param scores: a list such as one returned by 
      :obj:`Orange.feature.scoring.attMeasure`
    :type scores: list
    :param N: number of best features to select. 
    :type N: int
    :rtype: :obj:`list`

    """
    return map(lambda x:x[0], scores[:N])

def attsAboveThreshold(scores, threshold=0.0):
    """Return features (without scores) from the list returned by
    :obj:`Orange.feature.scoring.attMeasure` with score above or
    equal to a specified threshold.
    
    :param scores: a list such as one returned by
      :obj:`Orange.feature.scoring.attMeasure`
    :type scores: list
    :param threshold: score threshold for attribute selection. Defaults to 0.
    :type threshold: float
    :rtype: :obj:`list`

    """
    pairs = filter(lambda x, t=threshold: x[1] > t, scores)
    return map(lambda x:x[0], pairs)

def selectBestNAtts(data, scores, N):
    """Construct and return a new set of examples that includes a
    class and only N best features from a list scores.
    
    :param data: an example table
    :type data: Orange.data.table
    :param scores: a list such as one returned by 
      :obj:`Orange.feature.scoring.attMeasure`
    :type scores: list
    :param N: number of features to select
    :type N: int
    :rtype: :class:`Orange.data.table` holding N best features

    """
    return data.select(bestNAtts(scores, N)+[data.domain.classVar.name])


def selectAttsAboveThresh(data, scores, threshold=0.0):
    """Construct and return a new set of examples that includes a class and 
    features from the list returned by 
    :obj:`Orange.feature.scoring.attMeasure` that have the score above or 
    equal to a specified threshold.
    
    :param data: an example table
    :type data: Orange.data.table
    :param scores: a list such as one returned by
      :obj:`Orange.feature.scoring.attMeasure`    
    :type scores: list
    :param threshold: score threshold for attribute selection. Defaults to 0.
    :type threshold: float
    :rtype: :obj:`list` first N features (without measures)
  
    """
    return data.select(attsAboveThreshold(scores, threshold)+[data.domain.classVar.name])

def filterRelieff(data, measure=orange.MeasureAttribute_relief(k=20, m=50), margin=0):
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
    measl = attMeasure(data, measure)
    while len(data.domain.attributes)>0 and measl[-1][1]<margin:
        data = selectBestNAtts(data, measl, len(data.domain.attributes)-1)
#        print 'remaining ', len(data.domain.attributes)
        measl = attMeasure(data, measure)
    return data

##############################################################################
# wrappers

def FilterAttsAboveThresh(data=None, **kwds):
    filter = apply(FilterAttsAboveThresh_Class, (), kwds)
    if data:
        return filter(data)
    else:
        return filter
  
class FilterAttsAboveThresh_Class:
    """Stores filter's parameters and can be later called with the data to
    return the data table with only selected features. 
    
    This class is used in the function :obj:`selectAttsAboveThresh`.
    
    :param measure: an attribute measure (derived from 
      :obj:`Orange.feature.scoring.Measure`). Defaults to 
      :obj:`Orange.feature.scoring.Relief` for k=20 and m=50.  
    :param threshold: score threshold for attribute selection. Defaults to 0.
    :type threshold: float
     
    Some examples of how to use this class are::

        filter = Orange.feature.selection.FilterAttsAboveThresh(threshold=.15)
        new_data = filter(data)
        new_data = Orange.feature.selection.FilterAttsAboveThresh(data)
        new_data = Orange.feature.selection.FilterAttsAboveThresh(data, threshold=.1)
        new_data = Orange.feature.selection.FilterAttsAboveThresh(data, threshold=.1,
                   measure=Orange.feature.scoring.Gini())

    """
    def __init__(self, measure=orange.MeasureAttribute_relief(k=20, m=50), 
               threshold=0.0):
        self.measure = measure
        self.threshold = threshold

    def __call__(self, data):
        """Take data and return features with scores above given threshold.
        
        :param data: an data table
        :type data: Orange.data.table

        """
        ma = attMeasure(data, self.measure)
        return selectAttsAboveThresh(data, ma, self.threshold)

def FilterBestNAtts(data=None, **kwds):
    """Similarly to :obj:`FilterAttsAboveThresh`, wrap around class
    :obj:`FilterBestNAtts_Class`.
    
    :param measure: an attribute measure (derived from 
      :obj:`Orange.feature.scoring.Measure`). Defaults to 
      :obj:`Orange.feature.scoring.Relief` for k=20 and m=50.  
    :param n: number of best features to return. Defaults to 5.
    :type n: int

    """
    filter = apply(FilterBestNAtts_Class, (), kwds)
    if data: return filter(data)
    else: return filter
  
class FilterBestNAtts_Class:
    def __init__(self, measure=orange.MeasureAttribute_relief(k=20, m=50), n=5):
        self.measure = measure
        self.n = n
    def __call__(self, data):
        ma = attMeasure(data, self.measure)
        self.n = min(self.n, len(data.domain.attributes))
        return selectBestNAtts(data, ma, self.n)

def FilterRelief(data=None, **kwds):
    """Similarly to :obj:`FilterBestNAtts`, wrap around class 
    :obj:`FilterRelief_Class`.
    
    :param measure: an attribute measure (derived from 
      :obj:`Orange.feature.scoring.Measure`). Defaults to 
      :obj:`Orange.feature.scoring.Relief` for k=20 and m=50.  
    :param margin: margin for Relief scoring. Defaults to 0.
    :type margin: float

    """    
    filter = apply(FilterRelief_Class, (), kwds)
    if data:
        return filter(data)
    else:
        return filter
  
class FilterRelief_Class:
    def __init__(self, measure=orange.MeasureAttribute_relief(k=20, m=50), margin=0):
        self.measure = measure
        self.margin = margin
    def __call__(self, data):
        return filterRelieff(data, self.measure, self.margin)

##############################################################################
# wrapped learner

def FilteredLearner(baseLearner, examples = None, weight = None, **kwds):
    """Return the corresponding learner that wraps 
    :obj:`Orange.classification.baseLearner` and a data selection method. 
    
    When such learner is presented a data table, data is first filtered and 
    then passed to :obj:`Orange.classification.baseLearner`. This comes handy 
    when one wants to test the schema of feature-subset-selection-and-learning
    by some repetitive evaluation method, e.g., cross validation. 
    
    :param filter: defatuls to
      :obj:`Orange.feature.selection.FilterAttsAboveThresh`
    :type filter: Orange.feature.selection.FilterAttsAboveThresh

    Here is an example of how to build a wrapper around naive Bayesian learner
    and use it on a data set::

        nb = Orange.classification.bayes.NaiveBayesLearner()
        learner = Orange.feature.selection.FilteredLearner(nb, 
                  filter=Orange.feature.selection.FilterBestNAtts(n=5), name='filtered')
        classifier = learner(data)

    """
    learner = apply(FilteredLearner_Class, [baseLearner], kwds)
    if examples:
        return learner(examples, weight)
    else:
        return learner

class FilteredLearner_Class:
    def __init__(self, baseLearner, filter=FilterAttsAboveThresh(), name='filtered'):
        self.baseLearner = baseLearner
        self.filter = filter
        self.name = name
    def __call__(self, data, weight=0):
        # filter the data and then learn
        fdata = self.filter(data)
        model = self.baseLearner(fdata, weight)
        return FilteredClassifier(classifier = model, domain = model.domain)

class FilteredClassifier:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
    def __call__(self, example, resultType = orange.GetValue):
        return self.classifier(example, resultType)
    def atts(self):
        return self.domain.attributes  
