.. automodule:: Orange.classification.bayes

.. index:: naive Bayes classifier

.. index::
   single: classification; naive Bayes classifier

**********************************
Naive Bayes classifier (``bayes``)
**********************************

A `Naive Bayes classifier
<http://en.wikipedia.org/wiki/Naive_Bayes_classifier>`_ is a
probabilistic classifier that estimates conditional probabilities of the
dependant variable from training data and uses them for classification
of new data instances. The algorithm is very fast for discrete features, but
runs slower for continuous features.

The following example demonstrates a straightforward invocation of
this algorithm:

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
m-estimate:

.. literalinclude:: code/bayes-mestimate.py
    :lines: 7-

Conditional probabilities in an m-estimate based classifier show a
shift towards the second class - as compared to probabilities above, where
relative frequencies were used. The change in error estimation did
not have any effect on apriori probabilities:

.. literalinclude:: code/bayes-thresholdAdjustment.py
    :lines: 7-

Setting :obj:`~NaiveLearner.adjust_threshold` can improve the results.
The classification accuracies of 10-fold cross-validation of a normal naive
bayesian classifier, and one with an adjusted threshold::

    [0.7901746265516516, 0.8280138859667578]

Probability distributions for continuous features are estimated with \
:class:`~Orange.statistics.estimate.Loess`.

.. literalinclude:: code/bayes-plot-iris.py
    :lines: 4-

.. image:: files/bayes-iris.png
   :scale: 50 %

If petal lengths are shorter, the most probable class is "setosa". Irises with
middle petal lengths belong to "versicolor", while longer petal lengths indicate
for "virginica". Critical values where the decision would change are at about
5.4 and 6.3.
