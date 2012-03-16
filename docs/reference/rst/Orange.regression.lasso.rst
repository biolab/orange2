############################
Lasso regression (``lasso``)
############################

.. automodule:: Orange.regression.lasso

.. index:: regression

.. _`Lasso regression. Regression shrinkage and selection via the lasso`:
    http://www-stat.stanford.edu/~tibs/lasso/lasso.pdf


`The Lasso <http://www-stat.stanford.edu/~tibs/lasso/lasso.pdf>`_ is a shrinkage
and selection method for linear regression. It minimizes the usual sum of squared
errors, with a bound on the sum of the absolute values of the coefficients. 

To fit the regression parameters on housing data set use the following code:

.. literalinclude:: code/lasso-example.py
   :lines: 9,10,11

.. autoclass:: LassoRegressionLearner
    :members:

.. autoclass:: LassoRegression
    :members:


.. autoclass:: LassoRegressionLearner
    :members:

.. autoclass:: LassoRegression
    :members:

Utility functions
-----------------

.. autofunction:: center

.. autofunction:: get_bootstrap_sample

.. autofunction:: permute_responses


========
Examples
========

To predict values of the response for the first five instances
use the code

.. literalinclude:: code/lasso-example.py
   :lines: 14,15

Output

::

    Actual: 24.00, predicted: 24.58 
    Actual: 21.60, predicted: 23.30 
    Actual: 34.70, predicted: 24.98 
    Actual: 33.40, predicted: 24.78 
    Actual: 36.20, predicted: 24.66 

To see the fitted regression coefficients, print the model

.. literalinclude:: code/lasso-example.py
   :lines: 17

The output

::

    Variable  Coeff Est  Std Error          p
     Intercept     22.533
          CRIM     -0.000      0.023      0.480      
         INDUS     -0.010      0.023      0.300      
            RM      1.303      0.994      0.000   ***
           AGE     -0.002      0.000      0.320      
       PTRATIO     -0.191      0.209      0.050     .
         LSTAT     -0.126      0.105      0.000   ***
    Signif. codes:  0 *** 0.001 ** 0.01 * 0.05 . 0.1 empty 1


    For 7 variables the regression coefficient equals 0: 
    ZN
    CHAS
    NOX
    DIS
    RAD
    TAX
    B

shows that some of the regression coefficients are equal to 0.    

