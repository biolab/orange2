.. automodule:: Orange.evaluation.reliability

.. index:: Reliability Estimation

.. index::
   single: reliability; Reliability Estimation for Regression

########################################
Reliability estimation (``reliability``)
########################################

*************************************
Reliability Estimation for Regression
*************************************

Reliability assessment statistically predicts reliability of single
predictions. Most of implemented algorithms are taken from Comparison of
approaches for estimating reliability of individual regression predictions,
Zoran Bosnić, 2008.

The following example shows basic usage of reliability estimation methods:

.. literalinclude:: code/reliability-basic.py

The important points of this example are:
 * construction of reliability estimators using classes,
   implemented in this module,
 * construction of a reliability learner that bonds a regular learner
   (:class:`~Orange.classification.knn.kNNLearner` in this case) with
   reliability estimators,
 * calling the constructed classifier with
   :obj:`Orange.classification.Classifier.GetBoth` option to obtain class
   probabilities; :obj:`probability` is the object that gets appended the
   :obj:`reliability_estimate` attribute, an instance of
   :class:`Orange.evaluation.reliability.Estimate`, by the reliability learner.

It is also possible to do reliability estimation on whole data
table, not only on single instance. Next example demonstrates usage of a
cross-validation technique for reliability estimation. Reliability estimations
for first 10 instances get printed:

.. literalinclude:: code/reliability-run.py

Reliability Methods
===================

Sensitivity Analysis (SAvar and SAbias)
---------------------------------------
.. autoclass:: SensitivityAnalysis

Variance of bagged models (BAGV)
--------------------------------
.. autoclass:: BaggingVariance

Local cross validation reliability estimate (LCV)
-------------------------------------------------
.. autoclass:: LocalCrossValidation

Local modeling of prediction error (CNK)
----------------------------------------
.. autoclass:: CNeighbours

Bagging variance c-neighbours (BVCK)
------------------------------------

.. autoclass:: BaggingVarianceCNeighbours

Mahalanobis distance
--------------------

.. autoclass:: Mahalanobis

Mahalanobis to center
---------------------

.. autoclass:: MahalanobisToCenter

Reliability estimation wrappers
===============================

.. autoclass:: Learner
    :members:

.. autoclass:: Classifier
    :members:

Reliability estimation results
==============================

.. autoclass:: Estimate
    :members:
    :show-inheritance:

There is a dictionary named :obj:`METHOD_NAME` that maps reliability estimation
method IDs (ints) to method names (strings).

In this module, there are also two constants for distinguishing signed and
absolute reliability estimation measures::

  SIGNED = 0
  ABSOLUTE = 1

Reliability estimation scoring methods
======================================

.. autofunction:: get_pearson_r

.. autofunction:: get_pearson_r_by_iterations

.. autofunction:: get_spearman_r

Example of usage
================

.. literalinclude:: code/reliability-long.py
    :lines: 1-16

This script prints out Pearson's R coefficient between reliability estimates
and actual prediction errors, and a corresponding p-value, for each of the
reliability estimation measures used by default. ::

  Estimate               r       p
  SAvar absolute        -0.077   0.454
  SAbias signed         -0.165   0.105
  SAbias absolute       -0.099   0.333
  BAGV absolute          0.104   0.309
  CNK signed             0.233   0.021
  CNK absolute           0.057   0.579
  LCV absolute           0.069   0.504
  BVCK_absolute          0.092   0.368
  Mahalanobis absolute   0.091   0.375


References
==========

Bosnić, Z., Kononenko, I. (2007) `Estimation of individual prediction
reliability using local sensitivity analysis. <http://www.springerlink
.com/content/e27p2584387532g8/>`_ *Applied Intelligence* 29(3), pp. 187-203.

Bosnić, Z., Kononenko, I. (2008) `Comparison of approaches for estimating
reliability of individual regression predictions. <http://www.sciencedirect
.com/science/article/pii/S0169023X08001080>`_ *Data & Knowledge Engineering*
67(3), pp. 504-516.

Bosnić, Z., Kononenko, I. (2010) `Automatic selection of reliability estimates
for individual regression predictions. <http://journals.cambridge
.org/abstract_S0269888909990154>`_ *The Knowledge Engineering Review* 25(1),
pp. 27-47.

