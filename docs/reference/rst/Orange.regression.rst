###########################
Regression (``regression``)
###########################

Orange implements a set of methods for regression modeling, that is,
where the outcome - dependent variable is real-valued:

.. toctree::
   :maxdepth: 1

   Orange.regression.linear
   Orange.regression.lasso
   Orange.regression.pls
   Orange.regression.earth
   Orange.regression.tree
   Orange.regression.mean

Notice that the dependent variable is in this documentation and in the
implementation referred to as `class variable`. See also the documentation
on :doc:`Orange.classification` for information on how to fit models
and use them for prediction.

*************************
Base class for regression
*************************

All regression learners are inherited from `BaseRegressionLearner`.

.. automodule:: Orange.regression.base
