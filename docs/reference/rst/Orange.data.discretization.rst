.. py:currentmodule:: Orange.data.discretization

########################################
Data discretization (``discretization``)
########################################

.. index:: discretization

.. index::
   single: data; discretization

Continues features in the data can be discretized using a uniform discretization method. Discretization considers
only continues features, and replaces them in the new data set with corresponding categorical features:

.. literalinclude:: code/discretization-table.py

Discretization introduces new categorical features with discretized values::

    Original data set:
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']
    [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']

    Discretized data set:
    ['<=5.45', '>3.15', '<=2.45', '<=0.80', 'Iris-setosa']
    ['<=5.45', '(2.85, 3.15]', '<=2.45', '<=0.80', 'Iris-setosa']
    ['<=5.45', '>3.15', '<=2.45', '<=0.80', 'Iris-setosa']

Data discretization uses feature discretization classes from :doc:`Orange.feature
.discretization` and applies them on entire data set. The suported discretization methods are:

* equal width discretization, where the domain of continuous feature is split to intervals of the same
  width equal-sized intervals (uses :class:`Orange.feature.discretization.EqualWidth`),
* equal frequency discretization, where each intervals contains equal number of data instances (uses
  :class:`Orange.feature.discretization.EqualFreq`),
* entropy-based, as originally proposed by [FayyadIrani1993]_ that infers the intervals to minimize
  within-interval entropy of class distributions (uses :class:`Orange.feature.discretization.Entropy`),
* bi-modal, using three intervals to optimize the difference of the class distribution in
  the middle with the distribution outside it (uses :class:`Orange.feature.discretization.BiModal`),
* fixed, with the user-defined cut-off points.

.. FIXME give a corresponding class for fixed discretization

Default discretization method (equal frequency with three intervals) can be replaced with other
discretization approaches as demonstrated below:

.. literalinclude:: code/discretization-table-method.py
    :lines: 3-5

Entropy-based discretization is special as it may infer new features that are constant and have only one value. Such
features are redundant and provide no information about the class are. By default,
:class:`DiscretizeTable` would remove them, a way performing feature subset selection. The effect of removal of
non-informative features is also demonstrated in the following script:

.. literalinclude:: code/discretization-entropy.py
    :lines: 3-

In the sampled dat set above three features were discretized to a constant and thus removed::

    Redundant features (3 of 13):
    cholesterol, rest SBP, age

.. note::
    Entropy-based and bi-modal discretization require class-labeled data sets.

Data discretization classes
===========================

.. .. autoclass:: Orange.feature.discretization.DiscretizedLearner_Class

.. autoclass:: DiscretizeTable

.. A chapter on `feature subset selection <../ofb/o_fss.htm>`_ in Orange
   for Beginners tutorial shows the use of DiscretizedLearner. Other
   discretization classes from core Orange are listed in chapter on
   `categorization <../ofb/o_categorization.htm>`_ of the same tutorial. -> should put in classification/wrappers

.. [FayyadIrani1993] UM Fayyad and KB Irani. Multi-interval discretization of continuous valued
  attributes for classification learning. In Proc. 13th International Joint Conference on Artificial Intelligence, pages
  1022--1029, Chambery, France, 1993.