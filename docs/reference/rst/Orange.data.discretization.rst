.. py:currentmodule:: Orange.data

###################################
Discretization (``discretization``)
###################################

.. index:: discretization

.. index::
   single: data; discretization

Continues features in the data can be discretized using a uniform discretization method. The approach will consider
only continues features, and replace them in the data set with corresponding categorical features:

.. literalinclude:: code/discretization-table.py

Discretization introduces new categorical features and computes their values in accordance to
a discretization method::

    Original data set:
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']

    Discretized data set:
    ['<=5.45', '>3.15', '<=2.45', '<=0.80', 'Iris-setosa']
    ['<=5.45', '(2.85, 3.15]', '<=2.45', '<=0.80', 'Iris-setosa']
    ['<=5.45', '>3.15', '<=2.45', '<=0.80', 'Iris-setosa']

The procedure uses feature discretization classes as define in XXX and applies them on entire data sets.
The suported discretization methods are:

* equal width discretization, where the domain of continuous feature is split to intervals of the same
  width equal-sized intervals (:class:`EqualWidth`),
* equal frequency discretization, where each intervals contains equal number of data instances (:class:`EqualFreq`),
* entropy-based, as originally proposed by [FayyadIrani1993]_ that infers the intervals to minimize
  within-interval entropy of class distributions (:class:`Entropy`),
* bi-modal, using three intervals to optimize the difference of the class distribution in
  the middle with the distribution outside it (:class:`BiModal`),
* fixed, with the user-defined cut-off points.

The above script used the default discretization method (equal frequency with three intervals). This can be
changed while some selected discretization approach as demonstrated below:

.. literalinclude:: code/discretization-table-method.py
    :lines: 3-5

Classes
=======

Some functions and classes that can be used for
categorization of continuous features. Besides several general classes that
can help in this task, we also provide a function that may help in
entropy-based discretization (Fayyad & Irani), and a wrapper around classes for
categorization that can be used for learning.

.. autoclass:: Orange.feature.discretization.DiscretizedLearner_Class

.. autoclass:: DiscretizeTable

.. rubric:: Example

FIXME. A chapter on `feature subset selection <../ofb/o_fss.htm>`_ in Orange
for Beginners tutorial shows the use of DiscretizedLearner. Other
discretization classes from core Orange are listed in chapter on
`categorization <../ofb/o_categorization.htm>`_ of the same tutorial.
