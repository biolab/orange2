.. py:currentmodule:: Orange.classification.svm

.. index:: support vector machines (SVM)
.. index:
   single: classification; support vector machines (SVM)
   
*********************************
Support Vector Machines (``svm``)
*********************************

The module for `Support Vector Machine`_ (SVM) classification is based
on the popular `LibSVM`_ and `LIBLINEAR`_ libraries. It provides several
learning algorithms:

- :obj:`SVMLearner`, a general SVM learner;
- :obj:`SVMLearnerEasy`, which is similar to the `svm-easy.py` script
    from the LibSVM distribution and helps with the data normalization and
    parameter tuning;
- :obj:`LinearSVMLearner`, a fast learner useful for data sets with a large
    number of features.
          
SVM learners (from `LibSVM`_)
=============================

:class:`SVMLearner` uses the standard `LibSVM`_ learner. It supports
several built-in kernel types and user-defined kernels functions
written in Python. The kernel type is denoted by constants ``Linear``,
``Polynomial``, ``RBF``, ``Sigmoid`` and ``Custom`` defined in
``Orange.classification.svm.kernels``.  A custom kernel function must
accept two data instances and return a float. See
:ref:`kernel-wrapper` for examples.
    
The class also supports several types of optimization: ``C_SVC``,
``Nu_SVC`` (default), ``OneClass``, ``Epsilon_SVR`` and ``Nu_SVR``
(defined in ``Orange.classification.svm.SVMLearner``).

Class :obj:`SVMLearner` works on non-sparse data and
:class:`SVMLearnerSparse` class works on sparse data sets, for
instance data from the `basket` format).

.. autoclass:: Orange.classification.svm.SVMLearner
    :members:

.. autoclass:: Orange.classification.svm.SVMLearnerSparse
    :members:
    :show-inheritance:
    
.. autoclass:: Orange.classification.svm.SVMLearnerEasy
    :members:
    :show-inheritance:

The example below compares the performances of :obj:`SVMLearnerEasy`
with automatic data preprocessing and parameter tuning and
:obj:`SVMLearner` with the default :obj:`~SVMLearner.nu` and
:obj:`~SVMLearner.gamma`:
    
.. literalinclude:: code/svm-easy.py


   
Linear SVM learners (from `LIBLINEAR`_)
=======================================

Linear SVM learners are more suitable for large scale problems since
they are significantly faster then :class:`SVMLearner` and its
subclasses. A down side is that they support only a linear kernel and
can not estimate probabilities.
   
.. autoclass:: Orange.classification.svm.LinearSVMLearner
   :members:
   
.. autoclass:: Orange.classification.svm.MultiClassSVMLearner
   :members:
   
   
SVM Based feature selection and scoring
=======================================

.. autoclass:: Orange.classification.svm.RFE

.. autoclass:: Orange.classification.svm.ScoreSVMWeights
    :show-inheritance:
 
 
Utility functions
=================

.. automethod:: Orange.classification.svm.max_nu

.. automethod:: Orange.classification.svm.get_linear_svm_weights

.. automethod:: Orange.classification.svm.table_to_svm_format

The following example shows how to get linear SVM weights:
    
.. literalinclude:: code/svm-linear-weights.py    


.. _kernel-wrapper:

Kernel wrappers
===============

Kernel wrappers are helper classes for building custom kernels for use
with :class:`SVMLearner` and subclasses. They take and transform one
or two Python functions (attributes :obj:`wrapped` or :obj:`wrapped1`
and :obj:`wrapped2`). The function must be a positive definite kernel
that takes two arguments of type :class:`Orange.data.Instance` and
returns a float.

.. autoclass:: Orange.classification.svm.kernels.KernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.DualKernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.RBFKernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.PolyKernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.AdditionKernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.MultiplicationKernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.CompositeKernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.SparseLinKernel
   :members:

Example:

.. literalinclude:: code/svm-custom-kernel.py

.. _`Support Vector Machine`: http://en.wikipedia.org/wiki/Support_vector_machine
.. _`LibSVM`: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
.. _`LIBLINEAR`: http://www.csie.ntu.edu.tw/~cjlin/liblinear/
