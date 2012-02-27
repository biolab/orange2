###########################
Regression (``regression``)
###########################

Orange uses the term `classification` to also denote the
regression. For instance, the dependent variable is called a `class
variable` even when it is continuous, and models are generally called
classifiers. A part of the reason is that classification and
regression rely on the same set of basic classes.

Please see the documentation on :doc:`Orange.classification` for
information on how to fit models in general.

Orange contains a number of regression models which are listed below.

.. toctree::
   :maxdepth: 1

   Orange.regression.mean
   Orange.regression.linear
   Orange.regression.lasso
   Orange.regression.pls
   Orange.regression.earth
   Orange.regression.tree

.. automodule:: Orange.regression.base