.. automodule:: Orange.classification.bayes

.. index:: naive Bayes classifier

.. index::
   single: classification; naive Bayes classifier

**********************************
Naive Bayes classifier (``bayes``)
**********************************

The most primitive Bayesian classifier is :obj:`NaiveLearner`.
`Naive Bayes classification algorithm <http://en.wikipedia.org/wiki/Naive_Bayes_classifier>`_
estimates conditional probabilities from training data and uses them
for classification of new data instances. The algorithm learns very fast if all features
in the training data set are discrete. If a number of features are continues, though, the
algorithm runs slower due to time spent to estimate continuous conditional distributions.

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
:class:`ProbabilityEstimatorConstructor_loess`.
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

Implementation details
======================

The following two classes are implemented in C++ (*bayes.cpp*). They are not
intended to be used directly. Here we provide implementation details for those
interested.

Orange.core.BayesLearner
------------------------
Fields estimatorConstructor, conditionalEstimatorConstructor and
conditionalEstimatorConstructorContinuous are empty (None) by default.

If estimatorConstructor is left undefined, p(C) will be estimated by relative
frequencies of examples (see ProbabilityEstimatorConstructor_relative).
When conditionalEstimatorConstructor is left undefined, it will use the same
constructor as for estimating unconditional probabilities (estimatorConstructor
is used as an estimator in ConditionalProbabilityEstimatorConstructor_ByRows).
That is, by default, both will use relative frequencies. But when
estimatorConstructor is set to, for instance, estimate probabilities by
m-estimate with m=2.0, the same estimator will be used for estimation of
conditional probabilities, too.
P(c|vi) for continuous attributes are, by default, estimated with loess (a
variant of locally weighted linear regression), using
ConditionalProbabilityEstimatorConstructor_loess.
The learner first constructs an estimator for p(C). It tries to get a
precomputed distribution of probabilities; if the estimator is capable of
returning it, the distribution is stored in the classifier's field distribution
and the just constructed estimator is disposed. Otherwise, the estimator is
stored in the classifier's field estimator, while the distribution is left
empty.

The same is then done for conditional probabilities. Different constructors are
used for discrete and continuous attributes. If the constructed estimator can
return all conditional probabilities in form of Contingency, the contingency is
stored and the estimator disposed. If not, the estimator is stored. If there
are no contingencies when the learning is finished, the resulting classifier's
conditionalDistributions is None. Alternatively, if all probabilities are
stored as contingencies, the conditionalEstimators fields is None.

Field normalizePredictions is copied to the resulting classifier.

Orange.core.BayesClassifier
---------------------------
Class NaiveClassifier represents a naive bayesian classifier. Probability of
class C, knowing that values of features :math:`F_1, F_2, ..., F_n` are
:math:`v_1, v_2, ..., v_n`, is computed as :math:`p(C|v_1, v_2, ..., v_n) = \
p(C) \\cdot \\frac{p(C|v_1)}{p(C)} \\cdot \\frac{p(C|v_2)}{p(C)} \\cdot ... \
\\cdot \\frac{p(C|v_n)}{p(C)}`.

Note that when relative frequencies are used to estimate probabilities, the
more usual formula (with factors of form :math:`\\frac{p(v_i|C)}{p(v_i)}`) and
the above formula are exactly equivalent (without any additional assumptions of
independency, as one could think at a first glance). The difference becomes
important when using other ways to estimate probabilities, like, for instance,
m-estimate. In this case, the above formula is much more appropriate.

When computing the formula, probabilities p(C) are read from distribution, which
is of type Distribution, and stores a (normalized) probability of each class.
When distribution is None, BayesClassifier calls estimator to assess the
probability. The former method is faster and is actually used by all existing
methods of probability estimation. The latter is more flexible.

Conditional probabilities are computed similarly. Field conditionalDistribution
is of type DomainContingency which is basically a list of instances of
Contingency, one for each attribute; the outer variable of the contingency is
the attribute and the inner is the class. Contingency can be seen as a list of
normalized probability distributions. For attributes for which there is no
contingency in conditionalDistribution a corresponding estimator in
conditionalEstimators is used. The estimator is given the attribute value and
returns distributions of classes.

If neither, nor pre-computed contingency nor conditional estimator exist, the
attribute is ignored without issuing any warning. The attribute is also ignored
if its value is undefined; this cannot be overriden by estimators.

Any field (distribution, estimator, conditionalDistributions,
conditionalEstimators) can be None. For instance, BayesLearner normally
constructs a classifier which has either distribution or estimator defined.
While it is not an error to have both, only distribution will be used in that
case. As for the other two fields, they can be both defined and used
complementarily; the elements which are missing in one are defined in the
other. However, if there is no need for estimators, BayesLearner will not
construct an empty list; it will not construct a list at all, but leave the
field conditionalEstimators empty.

If you only need probabilities of individual class call BayesClassifier's
method p(class, example) to compute the probability of this class only. Note
that this probability will not be normalized and will thus, in general, not
equal the probability returned by the call operator.