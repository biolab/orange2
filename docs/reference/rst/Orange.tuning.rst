.. py:currentmodule:: Orange.classification.majority

###############################
Tuning (``tuning``)
###############################

.. automodule:: Orange.tuning

.. index:: tuning

Wrappers for Tuning Parameters and Thresholds

Classes for two very useful purposes: tuning learning algorithm's parameters
using internal validation and tuning the threshold for classification into
positive class.

*****************
Tuning parameters
*****************

Two classes support tuning parameters.
:obj:`~Tune1Parameter` for fitting a single parameter and
:obj:`~TuneMParameters` fitting multiple parameters at once,
trying all possible combinations. When called with data and, optionally, id
of meta attribute with weights, they find the optimal setting of arguments
using cross validation. The classes can also be used as ordinary learning
algorithms - they are in fact derived from
:obj:`~Orange.classification.Learner`.

Both classes have a common parent, :obj:`~TuneParameters`,
and a few common attributes.

.. autoclass:: TuneParameters
   :members:

.. autoclass:: Tune1Parameter
   :members:

.. autoclass:: TuneMParameters
   :members:

**************************
Setting Optimal Thresholds
**************************

Some models may perform well in terms of AUC which measures the ability to
distinguish between instances of two classes, but have low classifications
accuracies. The reason may be in the threshold: in binary problems, classifiers
usually classify into the more probable class, while sometimes, when class
distributions are highly skewed, a modified threshold would give better
accuracies. Here are two classes that can help.

.. autoclass:: ThresholdLearner
   :members:

.. autoclass:: ThresholdClassifier
   :members:

Examples
========

This is how you use the learner.

part of :download:`optimization-thresholding1.py <code/optimization-thresholding1.py>`

.. literalinclude:: code/optimization-thresholding1.py

The output::

    W/out threshold adjustement: 0.633
    With adjusted thredhold: 0.659
    With threshold at 0.80: 0.449

part of :download:`optimization-thresholding2.py <code/optimization-thresholding2.py>`

.. literalinclude:: code/optimization-thresholding2.py

The script first divides the data into training and testing subsets. It trains
a naive Bayesian classifier and than wraps it into
:obj:`~ThresholdClassifiers` with thresholds of .2, .5 and
.8. The three models are tested on the left-out data, and we compute the
confusion matrices from the results. The printout::

    0.20: TP 60.000, TN 1.000
    0.50: TP 42.000, TN 24.000
    0.80: TP 2.000, TN 43.000

shows how the varying threshold changes the balance between the number of true
positives and negatives.

.. autoclass:: PreprocessedLearner
   :members: