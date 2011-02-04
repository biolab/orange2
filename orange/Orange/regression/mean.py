"""

.. index:: regression; mean

Accuracy of classifiers is often compared to the "default accuracy".
For regression, that is the accuracy of a classifier which predicts for
all instances the mean value of all observed class values in the
training data. To fit into the standard schema, even this algorithm
is provided in form of the usual learner-classifier pair.
Learning is done by :obj:`MeanLearner` and the classifier it
constructs is an instance of :obj:`ConstantClassifier`.

This is the regression counterpart of the
:obj:`Orange.classification.majority.MajorityLearner`, which can be
used for classification problems.

Examples
========

This "learning algorithm" will most often be used to establish
whether some other learning algorithm is better than "nothing".
Here's a simple example.

`mean-regression.py`_ (uses: `housing.tab`_):

.. literalinclude:: code/mean-regression.py
    :lines: 7-

.. _mean-regression.py: code/mean-regression.py
.. _housing.tab: code/housing.tab

"""

from Orange.core import MajorityLearner as MeanLearner
from Orange.core import DefaultClassifier as ConstantClassifier
