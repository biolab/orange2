.. _Logistic Regression:

Logistic Regression Learner
===========================

.. image:: ../../../../Orange/OrangeWidgets/Classify/icons/LogisticRegression.svg

Logistic Regression Learner

Signals
-------

Inputs:

   - Data
      A table with training instances


Outputs:

   - Learner
      The logistic regression learning algorithm with settings as
      specified in the dialog.

   - Logistic Regression Classifier
      Trained classifier (a subtype of Classifier)


Signal :obj:`Logistic Regression Classifier` sends data only if the learning
data (signal :obj:`Data`) is present.

Description
-----------

This widget provides a graphical interface to the logistic regression
classifier.

As all widgets for classification, this widget provides a learner and
classifier on the output. Learner is a learning algorithm with settings
as specified by the user. It can be fed into widgets for testing learners,
for instance :ref:`Test Learners`. Classifier is a logistic regression
classifier (a subtype of a general classifier), built from the training
examples on the input. If examples are not given, there is no classifier
on the output.

.. image:: images/LogisticRegression.png
   :alt: Logistic Regression Widget


.. rst-class:: stamp-list::

   1. Learner can be given a name under which it will appear in, say,
      :ref:`Test Learners`. The default name is "Logistic regression".

   2. Set the regularization type (L1 or L2 weight penalty).

   3. Set error cost paramter (higher cost means less regularization).

   4. Normalize the features before training.


Examples
--------

The widget is used just as any other widget for inducing classifier. See,
for instance, the example for the :ref:`Naive Bayes`.
