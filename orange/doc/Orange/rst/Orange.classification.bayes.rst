.. automodule:: Orange.classification.bayes

.. index:: naive Bayes classifier

.. index::
   single: classification; naive Bayes classifier

**********************************
Naive Bayes classifier (``bayes``)
**********************************

A `Naive Bayes classifier <http://en.wikipedia.org/wiki/Naive_Bayes_classifier>`_
is a simple probabilistic classifier that estimates conditional probabilities of the dependant variable
from training data and uses them for classification of new data instances. The algorithm is very
fast if all features in the training data set are discrete. If a number of features are continuous,
though, the algorithm runs slower due to time spent to estimate continuous conditional distributions.

The following example demonstrates a straightforward invocation of
this algorithm (`bayes-run.py`_, uses `titanic.tab`_):

.. literalinclude:: code/bayes-run.py
   :lines: 7-

.. index:: Naive Bayesian Learner
.. autoclass:: Orange.classification.bayes.NaiveLearner
   :members:
   :show-inheritance:

.. autoclass:: Orange.classification.bayes.NaiveClassifier
   :members:
   :show-inheritance:


Examples
========

:obj:`NaiveLearner` can estimate probabilities using relative frequencies or
m-estimate (`bayes-mestimate.py`_, uses `lenses.tab`_):

.. literalinclude:: code/bayes-mestimate.py
    :lines: 7-

Observing conditional probabilities in an m-estimate based classifier shows a
shift towards the second class - as compared to probabilities above, where
relative frequencies were used. Note that the change in error estimation did
not have any effect on apriori probabilities
(`bayes-thresholdAdjustment.py`_, uses `adult-sample.tab`_):

.. literalinclude:: code/bayes-thresholdAdjustment.py
    :lines: 7-

Setting adjustThreshold parameter can sometimes improve the results. Those are
the classification accuracies of 10-fold cross-validation of a normal naive
bayesian classifier, and one with an adjusted threshold::

    [0.7901746265516516, 0.8280138859667578]

Probabilities for continuous features are estimated with \
:class:`Orange.statistics.estimate.Loess`.
(`bayes-plot-iris.py`_, uses `iris.tab`_):

.. literalinclude:: code/bayes-plot-iris.py
    :lines: 4-

.. image:: code/bayes-iris.png
   :scale: 50 %

If petal lengths are shorter, the most probable class is "setosa". Irises with
middle petal lengths belong to "versicolor", while longer petal lengths indicate
for "virginica". Critical values where the decision would change are at about
5.4 and 6.3.


.. _bayes-run.py: code/bayes-run.py
.. _bayes-thresholdAdjustment.py: code/bayes-thresholdAdjustment.py
.. _bayes-mestimate.py: code/bayes-mestimate.py
.. _bayes-plot-iris.py: code/bayes-plot-iris.py
.. _adult-sample.tab: code/adult-sample.tab
.. _iris.tab: code/iris.tab
.. _titanic.tab: code/iris.tab
.. _lenses.tab: code/lenses.tab