################
Mean (``mean``)
################

.. py:currentmodule:: Orange.regression.mean

.. index:: regression; mean

Accuracy of a regressor is often compared with the accuracy achieved
by always predicting the averag value. The "learning algorithm"
computes the average and represents it with a regressor of type
:obj:`Orange.classification.ConstantClassifier`.

.. rubric:: Examples

The following example compares the mean squared error of always
predicting the average with the error of a tree learner.

:download:`mean-regression.py <code/mean-regression.py>`:

.. literalinclude:: code/mean-regression.py
    :lines: 7-
