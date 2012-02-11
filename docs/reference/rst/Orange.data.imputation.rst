.. py:currentmodule:: Orange.data.imputation

.. index:: imputation

.. index::
   single: table; value imputation

***************************
Imputation (``imputation``)
***************************

Imputation replaces missing feature values with appropriate values. The
example below shows how to replace the missing values with variables'
averages:

.. literalinclude:: code/imputation-data.py
   :lines: 7-

The output of this code is::

    Original data set:
    ['M', 1818, 'HIGHWAY', ?, 2, 'N', 'THROUGH', 'WOOD', 'SHORT', 'S', 'WOOD']
    ['A', 1819, 'HIGHWAY', 1037, 2, 'N', 'THROUGH', 'WOOD', 'SHORT', 'S', 'WOOD']
    ['A', 1829, 'AQUEDUCT', ?, 1, 'N', 'THROUGH', 'WOOD', '?', 'S', 'WOOD']

    Imputed data set:
    ['M', 1818, 'HIGHWAY', 1300, 2, 'N', 'THROUGH', 'WOOD', 'SHORT', 'S', 'WOOD']
    ['A', 1819, 'HIGHWAY', 1037, 2, 'N', 'THROUGH', 'WOOD', 'SHORT', 'S', 'WOOD']
    ['A', 1829, 'AQUEDUCT', 1300, 1, 'N', 'THROUGH', 'WOOD', 'MEDIUM', 'S', 'WOOD']

The function uses feature imputation methods from
:doc:`Orange.feature.imputation` and applies them on entire data set. The
supported methods are:

* imputation of minimal, maximal, average value (uses :class:`Orange.feature.imputation.Defaults`),
* imputation of random value (uses :class:`Orange.feature.imputation.Random`),
* imputation based on a predictive model (uses :class:`Orange.feature.imputation.Model`),
* imputation where missing value is treated as a value (uses :class:`Orange.feature.imputation.AsValue`).
