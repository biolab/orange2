############################
Lasso regression (``lasso``)
############################

.. automodule:: Orange.regression.lasso

.. index:: regression

.. _`Lasso regression. Regression shrinkage and selection via the lasso`:
    http://www-stat.stanford.edu/~tibs/lasso/lasso.pdf


`The lasso <http://www-stat.stanford.edu/~tibs/lasso/lasso.pdf>`_
(least absolute shrinkage and selection operator) is a regularized
version of least squares regression.
It minimizes the sum of squared errors while also penalizing the
:math:`L_1` norm (sum of absolute values) of the coefficients.

Concretely, the function that is minimized in Orange is:

.. math:: \frac{1}{n}\|Xw - y\|_2^2 + \frac{\lambda}{m} \|w\|_1

Where :math:`X` is a :math:`n \times m` data matrix, :math:`y` the vector
of class values and :math:`w` the regression coefficients to be estimated.

.. autoclass:: LassoRegressionLearner
    :members:
    :show-inheritance:

.. autoclass:: LassoRegression
    :members:
    :show-inheritance:

Utility functions
-----------------

.. autofunction:: get_bootstrap_sample

.. autofunction:: permute_responses


========
Examples
========

To fit the regression parameters on housing data set use the following code:

.. literalinclude:: code/lasso-example.py
   :lines: 9,10,11

To predict values of the response for the first five instances:

.. literalinclude:: code/lasso-example.py
   :lines: 15,16

Output::

    Actual: 24.00, predicted: 30.45
    Actual: 21.60, predicted: 25.60
    Actual: 34.70, predicted: 31.48
    Actual: 33.40, predicted: 30.18
    Actual: 36.20, predicted: 29.59

To see the fitted regression coefficients, print the model:

.. literalinclude:: code/lasso-example.py
   :lines: 19

Output::

      Variable  Coeff Est  Std Error          p
     Intercept     22.533
          CRIM     -0.023      0.024      0.050     .
          CHAS      1.970      1.331      0.040     *
           NOX     -4.226      2.944      0.010     *
            RM      4.270      0.934      0.000   ***
           DIS     -0.373      0.170      0.010     *
       PTRATIO     -0.798      0.117      0.000   ***
             B      0.007      0.003      0.020     *
         LSTAT     -0.519      0.102      0.000   ***
    Signif. codes:  0 *** 0.001 ** 0.01 * 0.05 . 0.1 empty 1

    For 5 variables the regression coefficient equals 0:
    ZN, INDUS, AGE, RAD, TAX

Note that some of the regression coefficients are equal to 0.    

