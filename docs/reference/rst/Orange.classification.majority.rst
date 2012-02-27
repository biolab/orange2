.. py:currentmodule:: Orange.classification.majority

***********************
Majority (``majority``)
***********************

.. index:: majority classifier
   pair: classification; majority classifier

Accuracy of classifiers is often compared with the "default accuracy",
that is, the accuracy of a classifier which classifies all instances
to the majority class. The training of such classifier consists of
computing the class distribution and its modus. The model is
represented as an instance of
:obj:`Orange.classification.ConstantClassifier`.

.. class:: MajorityLearner

    MajorityLearner has two components, which are seldom used.

    .. attribute:: estimator_constructor
    
        An estimator constructor that can be used for estimation of
        class probabilities. If left ``None``, probability of each class is
        estimated as the relative frequency of instances belonging to
        this class.
        
    .. attribute:: apriori_distribution
    
        Apriori class distribution that is passed to estimator
        constructor if one is given.

Example
========

This "learning algorithm" will most often be used as a baseline,
that is, to determine if some other learning algorithm provides
any information about the class (:download:`majority-classification.py <code/majority-classification.py>`):

.. literalinclude:: code/majority-classification.py
    :lines: 7-
