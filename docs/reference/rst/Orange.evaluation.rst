#############################
Evaluation (``evaluation``)
#############################

Evaluation of prediction modules is split into two parts. Module
:obj:`Orange.evaluation.testing` contains procedures that sample data,
train learning algorithms and test models. All procedures return
results as an instance of
:obj:`~Orange.evaluation.testing.ExperimentResults` that is described
below. Module :obj:`Orange.evaluation.scoring` uses such data to
compute various performance scores like classification accuracy and
AUC.

There is a third module available as an add-on, which is unrelated to this
scheme,:obj:`Orange.evaluation.reliability`, that assesses the reliability
of individual predictions.

.. toctree::
   :maxdepth: 1

   Orange.evaluation.testing
   Orange.evaluation.scoring

Classes for storing the experimental results
--------------------------------------------


The following two classes are used for storing the results of experiments by :obj:`Orange.evaluation.testing` and computing of scores by :obj:`Orange.evaluation.scoring`. Instances of this class seldom need to be constructed and used outside of these two modules.

.. py:currentmodule:: Orange.evaluation.testing

.. autoclass:: ExperimentResults(iterations, classifier_names, class_values=None, weights=None, base_class=-1)
    :members:

.. autoclass:: TestedExample
    :members:
