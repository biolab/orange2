"""
********
Majority
********

.. index:: classification; majority

Accuracy of classifiers is often compared to the "default accuracy",
that is, the accuracy of a classifier which classifies all instances
to the majority class. To fit into the standard schema, even this
algorithm is provided in form of the usual learner-classifier pair.
Learning is done by :obj:`MajorityLearner` and the classifier it
constructs is an instance of :obj:`ConstantClassifier`.

.. class:: MajorityLearner

    MajorityLearner will most often be used as is, without setting any
    features. Nevertheless, it has two.

    .. attribute:: estimatorConstructor
    
        An estimator constructor that can be used for estimation of
        class probabilities. If left None, probability of each class is
        estimated as the relative frequency of instances belonging to
        this class.
        
    .. attribute:: aprioriDistribution
    
        Apriori class distribution that is passed to estimator
        constructor if one is given.

.. class:: ConstantClassifier

    ConstantClassifier always classifies to the same class and reports
    same class probabilities.

    .. attribute:: defaultVal
    
        Value that is returned by the classifier.
    
    .. attribute:: defaultDistribution

        Class probabilities returned by the classifier.

The ConstantClassifier's constructor can be called without arguments,
with value (for defaultVal), variable (for classVar). If the value is
given and is of type orange.Value (alternatives are an integer index
of a discrete value or a continuous value), its field variable is will
either be used for initializing classVar if variable is not given as
an argument, or checked against the variable argument, if it is given. 

.. rubric:: Examples

This "learning algorithm" will most often be used as a baseline,
that is, to determine if some other learning algorithm provides
any information about the class. Here's a simple example.

`majority-classification.py`_ (uses: `monks-1.tab`_):

.. literalinclude:: code/majority-classification.py
    :lines: 7-

.. _majority-classification.py: code/majority-classification.py
.. _monks-1.tab: code/monks-1.tab

"""

from Orange.core import MajorityLearner
from Orange.core import DefaultClassifier as ConstantClassifier
