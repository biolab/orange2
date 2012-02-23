.. automodule:: Orange.evaluation.scoring

############################
Method scoring (``scoring``)
############################

.. index: scoring

Scoring plays and integral role in evaluation of any prediction model. Orange
implements various scores for evaluation of classification,
regression and multi-label models. Most of the methods needs to be called
with an instance of :obj:`~Orange.evaluation.testing.ExperimentResults`.

.. literalinclude:: code/scoring-example.py

==============
Classification
==============

Calibration scores
==================
Many scores for evaluation of the classification models measure whether the
model assigns the correct class value to the test instances. Many of these
scores can be computed solely from the confusion matrix constructed manually
with the :obj:`confusion_matrices` function. If class variable has more than
two values, the index of the value to calculate the confusion matrix for should
be passed as well.

.. autoclass:: CA
.. autofunction:: sens
.. autofunction:: spec
.. autofunction:: PPV
.. autofunction:: NPV
.. autofunction:: precision
.. autofunction:: recall
.. autofunction:: F1
.. autofunction:: Falpha
.. autofunction:: MCC
.. autofunction:: AP
.. autofunction:: IS
.. autofunction:: confusion_chi_square

Discriminatory scores
=====================
Scores that measure how good can the prediction model separate instances with
different classes are called discriminatory scores.

.. autofunction:: Brier_score

.. autoclass:: AUC
    :members: by_weighted_pairs, by_pairs,
              weighted_one_against_all, one_against_all, single_class, pair,
              matrix

.. autofunction:: AUCWilcoxon

.. autofunction:: compute_ROC

.. autofunction:: confusion_matrices

.. autoclass:: ConfusionMatrix


Comparison of Algorithms
========================

.. autofunction:: McNemar

.. autofunction:: McNemar_of_two

==========
Regression
==========

Several alternative measures, as given below, can be used to evaluate
the sucess of numeric prediction:

.. image:: files/statRegression.png

.. autofunction:: MSE

.. autofunction:: RMSE

.. autofunction:: MAE

.. autofunction:: RSE

.. autofunction:: RRSE

.. autofunction:: RAE

.. autofunction:: R2

The following code (:download:`statExamples.py <code/statExamples.py>`) uses most of the above measures to
score several regression methods.

The code above produces the following output::

    Learner   MSE     RMSE    MAE     RSE     RRSE    RAE     R2
    maj       84.585  9.197   6.653   1.002   1.001   1.001  -0.002
    rt        40.015  6.326   4.592   0.474   0.688   0.691   0.526
    knn       21.248  4.610   2.870   0.252   0.502   0.432   0.748
    lr        24.092  4.908   3.425   0.285   0.534   0.515   0.715

=================
Ploting functions
=================

.. autofunction:: graph_ranks

The following script (:download:`statExamplesGraphRanks.py <code/statExamplesGraphRanks.py>`) shows hot to plot a graph:

.. literalinclude:: code/statExamplesGraphRanks.py

Code produces the following graph:

.. image:: files/statExamplesGraphRanks1.png

.. autofunction:: compute_CD

.. autofunction:: compute_friedman

=================
Utility Functions
=================

.. autofunction:: split_by_iterations


.. _mt-scoring:

============
Multi-target
============

:doc:`Multi-target <Orange.multitarget>` classifiers predict values for
multiple target classes. They can be used with standard
:obj:`~Orange.evaluation.testing` procedures (e.g.
:obj:`~Orange.evaluation.testing.Evaluation.cross_validation`), but require
special scoring functions to compute a single score from the obtained
:obj:`~Orange.evaluation.testing.ExperimentResults`.
Since different targets can vary in importance depending on the experiment,
some methods have options to indicate this e.g. through weights or customized
distance functions. These can also be used for normalization in case target
values do not have the same scales.

.. autofunction:: mt_flattened_score
.. autofunction:: mt_average_score

The whole procedure of evaluating multi-target methods and computing
the scores (RMSE errors) is shown in the following example
(:download:`mt-evaluate.py <code/mt-evaluate.py>`). Because we consider
the first target to be more important and the last not so much we will
indicate this using appropriate weights.

.. literalinclude:: code/mt-evaluate.py

Which outputs::

    Weighted RMSE scores:
        Majority    0.8228
          MTTree    0.3949
             PLS    0.3021
           Earth    0.2880

==========================
Multi-label classification
==========================

Multi-label classification requires different metrics than those used in
traditional single-label classification. This module presents the various
metrics that have been proposed in the literature. Let :math:`D` be a
multi-label evaluation data set, conisting of :math:`|D|` multi-label examples
:math:`(x_i,Y_i)`, :math:`i=1..|D|`, :math:`Y_i \\subseteq L`. Let :math:`H`
be a multi-label classifier and :math:`Z_i=H(x_i)` be the set of labels
predicted by :math:`H` for example :math:`x_i`.

.. autofunction:: mlc_hamming_loss
.. autofunction:: mlc_accuracy
.. autofunction:: mlc_precision
.. autofunction:: mlc_recall

The following script demonstrates the use of those evaluation measures:

.. literalinclude:: code/mlc-evaluate.py

The output should look like this::

    loss= [0.9375]
    accuracy= [0.875]
    precision= [1.0]
    recall= [0.875]

References
==========

Boutell, M.R., Luo, J., Shen, X. & Brown, C.M. (2004), 'Learning multi-label scene classification',
Pattern Recogintion, vol.37, no.9, pp:1757-71

Godbole, S. & Sarawagi, S. (2004), 'Discriminative Methods for Multi-labeled Classification', paper
presented to Proceedings of the 8th Pacific-Asia Conference on Knowledge Discovery and Data Mining
(PAKDD 2004)

Schapire, R.E. & Singer, Y. (2000), 'Boostexter: a bossting-based system for text categorization',
Machine Learning, vol.39, no.2/3, pp:135-68.
