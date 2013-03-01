.. _Linear Regression:

Linear Regression Learner
=========================

.. image:: ../../../../Orange/OrangeWidgets/icons/Unknown.png
	:alt: Linear Regression

Learns a linear function of its input data.

Channels
--------

Input
	- Data (Table)
		Input data table

Output
	- Learner
		The learning algorithm with the supplied parameters

	- Predictor
		Trained regressor

	- Model  Statisics
		A data table containing trained model statistics


Signal ``Predictor`` and ``Model Statistics`` send the output
signal only if input signal ``Data`` is present.

Description
-----------

Linear Regression widget construct a learner/predictor that learns a
linear function from its input data. Furthermore `Lasso`_ and `Ridge`_
regularization parameters can be specified.

.. image:: images/LinearRegression.png
	:alt: Linear Regression interface

.. rst-class:: stamp-list

    1. The learner/predictor name
    2. Train an ordinary least squares or ridge regression model
    3. If ``Ridge lambda`` is checked the learner will build a ridge regression
       model with 4 as the ``lambda`` parameter.
    4. Ridge lambda parameter.
    5. Use `Lasso`_ regularization.
    6. The Lasso bound (bound on the beta vector L1 norm)
    7. Tolerance (any beta value lower then this will be forced to 0)

.. _`Lasso`: http://en.wikipedia.org/wiki/Least_squares#LASSO_method

.. _`Ridge`: http://en.wikipedia.org/wiki/Ridge_regression
