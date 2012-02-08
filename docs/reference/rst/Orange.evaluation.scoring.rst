.. automodule:: Orange.evaluation.scoring

############################
Method scoring (``scoring``)
############################

.. index: scoring

Scoring plays and integral role in evaluation of any prediction model. Orange
implements various scores for evaluation of classification,
regression and multi-label models. Most of the methods needs to be called
with an instance of :obj:`ExperimentResults`.

.. literalinclude:: code/statExample0.py

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

.. autosingleton:: CA
.. autoclass:: CAClass
    :members:
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

.. autofunction:: AUC

    .. attribute:: AUC.ByWeightedPairs (or 0)

        Computes AUC for each pair of classes (ignoring instances of all other
        classes) and averages the results, weighting them by the number of
        pairs of instances from these two classes (e.g. by the product of
        probabilities of the two classes). AUC computed in this way still
        behaves as concordance index, e.g., gives the probability that two
        randomly chosen instances from different classes will be correctly
        recognized (this is of course true only if the classifier knows
        from which two classes the instances came).

    .. attribute:: AUC.ByPairs (or 1)

        Similar as above, except that the average over class pairs is not
        weighted. This AUC is, like the binary, independent of class
        distributions, but it is not related to concordance index any more.

    .. attribute:: AUC.WeightedOneAgainstAll (or 2)

        For each class, it computes AUC for this class against all others (that
        is, treating other classes as one class). The AUCs are then averaged by
        the class probabilities. This is related to concordance index in which
        we test the classifier's (average) capability for distinguishing the
        instances from a specified class from those that come from other classes.
        Unlike the binary AUC, the measure is not independent of class
        distributions.

    .. attribute:: AUC.OneAgainstAll (or 3)

        As above, except that the average is not weighted.

   In case of multiple folds (for instance if the data comes from cross
   validation), the computation goes like this. When computing the partial
   AUCs for individual pairs of classes or singled-out classes, AUC is
   computed for each fold separately and then averaged (ignoring the number
   of instances in each fold, it's just a simple average). However, if a
   certain fold doesn't contain any instances of a certain class (from the
   pair), the partial AUC is computed treating the results as if they came
   from a single-fold. This is not really correct since the class
   probabilities from different folds are not necessarily comparable,
   yet this will most often occur in a leave-one-out experiments,
   comparability shouldn't be a problem.

   Computing and printing out the AUC's looks just like printing out
   classification accuracies (except that we call AUC instead of
   CA, of course)::

       AUCs = Orange.evaluation.scoring.AUC(res)
       for l in range(len(learners)):
           print "%10s: %5.3f" % (learners[l].name, AUCs[l])

   For vehicle, you can run exactly this same code; it will compute AUCs
   for all pairs of classes and return the average weighted by probabilities
   of pairs. Or, you can specify the averaging method yourself, like this::

       AUCs = Orange.evaluation.scoring.AUC(resVeh, Orange.evaluation.scoring.AUC.WeightedOneAgainstAll)

   The following snippet tries out all four. (We don't claim that this is
   how the function needs to be used; it's better to stay with the default.)::

       methods = ["by pairs, weighted", "by pairs", "one vs. all, weighted", "one vs. all"]
       print " " *25 + "  \tbayes\ttree\tmajority"
       for i in range(4):
           AUCs = Orange.evaluation.scoring.AUC(resVeh, i)
           print "%25s: \t%5.3f\t%5.3f\t%5.3f" % ((methods[i], ) + tuple(AUCs))

   As you can see from the output::

                                   bayes   tree    majority
              by pairs, weighted:  0.789   0.871   0.500
                        by pairs:  0.791   0.872   0.500
           one vs. all, weighted:  0.783   0.800   0.500
                     one vs. all:  0.783   0.800   0.500

.. autofunction:: AUC_single

.. autofunction:: AUC_pair

.. autofunction:: AUC_matrix

The remaining functions, which plot the curves and statistically compare
them, require that the results come from a test with a single iteration,
and they always compare one chosen class against all others. If you have
cross validation results, you can either use split_by_iterations to split the
results by folds, call the function for each fold separately and then sum
the results up however you see fit, or you can set the ExperimentResults'
attribute number_of_iterations to 1, to cheat the function - at your own
responsibility for the statistical correctness. Regarding the multi-class
problems, if you don't chose a specific class, Orange.evaluation.scoring will use the class
attribute's baseValue at the time when results were computed. If baseValue
was not given at that time, 1 (that is, the second class) is used as default.

We shall use the following code to prepare suitable experimental results::

    ri2 = Orange.core.MakeRandomIndices2(voting, 0.6)
    train = voting.selectref(ri2, 0)
    test = voting.selectref(ri2, 1)
    res1 = Orange.evaluation.testing.learnAndTestOnTestData(learners, train, test)


.. autofunction:: AUCWilcoxon

.. autofunction:: compute_ROC


.. autofunction:: confusion_matrices

.. autoclass:: ConfusionMatrix


Comparison of Algorithms
------------------------

.. autofunction:: McNemar

.. autofunction:: McNemar_of_two

==========
Regression
==========

General Measure of Quality
==========================

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

=====================================
Scoring for multilabel classification
=====================================

Multi-label classification requries different metrics than those used in traditional single-label
classification. This module presents the various methrics that have been proposed in the literature.
Let :math:`D` be a multi-label evaluation data set, conisting of :math:`|D|` multi-label examples
:math:`(x_i,Y_i)`, :math:`i=1..|D|`, :math:`Y_i \\subseteq L`. Let :math:`H` be a multi-label classifier
and :math:`Z_i=H(x_i)` be the set of labels predicted by :math:`H` for example :math:`x_i`.

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
