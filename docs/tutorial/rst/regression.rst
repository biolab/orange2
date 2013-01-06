Regression
==========

.. index:: regression

From the interface point of view, regression methods in Orange are very similar to classification. Both intended for supervised data mining, they require class-labeled data. Just like in classification, regression is implemented with learners and regression models (regressors). Regression learners are objects that accept data and return regressors. Regression models are given data items to predict the value of continuous class:

.. literalinclude:: code/regression.py


Handful of Regressors
---------------------

.. index::
   single: regression; tree

Let us start with regression trees. Below is an example script that builds the tree from data on housing prices and prints out the tree in textual form:

.. literalinclude:: code/regression-tree.py
   :lines: 3-

The script outputs the tree::
   
   RM<=6.941: 19.9
   RM>6.941
   |    RM<=7.437
   |    |    CRIM>7.393: 14.4
   |    |    CRIM<=7.393
   |    |    |    DIS<=1.886: 45.7
   |    |    |    DIS>1.886: 32.7
   |    RM>7.437
   |    |    TAX<=534.500: 45.9
   |    |    TAX>534.500: 21.9

Following is initialization of few other regressors and their prediction of the first five data instances in housing price data set:

.. index::
   single: regression; mars
   single: regression; linear

.. literalinclude:: code/regression-other.py
   :lines: 3-

Looks like the housing prices are not that hard to predict::

   y    lin  mars tree
   21.4 24.8 23.0 20.1
   15.7 14.4 19.0 17.3
   36.5 35.7 35.6 33.8

Cross Validation
----------------

Just like for classification, the same evaluation module (``Orange.evaluation``) is available for regression. Its testing submodule includes procedures such as cross-validation, leave-one-out testing and similar, and functions in scoring submodule can assess the accuracy from the testing:

.. literalinclude:: code/regression-other.py
   :lines: 3-

.. index: 
   single: regression; root mean squared error

`MARS <http://en.wikipedia.org/wiki/Multivariate_adaptive_regression_splines>`_ has the lowest root mean squared error::

   Learner  RMSE
   lin      4.83
   mars     3.84
   tree     5.10

