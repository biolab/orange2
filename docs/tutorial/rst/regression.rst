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
   single: regression; linear

.. literalinclude:: code/regression-other.py
   :lines: 3-

Looks like the housing prices are not that hard to predict::

   y    lin  rf   tree
   12.7 11.3 15.3 19.1
   13.8 20.2 14.1 13.1
   19.3 20.8 20.7 23.3


Cross Validation
----------------

Just like for classification, the same evaluation module (``Orange.evaluation``) is available for regression. Its testing submodule includes procedures such as cross-validation, leave-one-out testing and similar, and functions in scoring submodule can assess the accuracy from the testing:

.. literalinclude:: code/regression-cv.py
   :lines: 3-

.. index: 
   single: regression; root mean squared error

Random forest has the lowest root mean squared error::

   Learner  RMSE
   lin      4.83
   rf       3.73
   tree     5.10
