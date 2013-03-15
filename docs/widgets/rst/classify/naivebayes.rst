.. _Naive Bayes:

Naive Bayesian Learner
======================

.. image:: ../icons/NaiveBayes.png

Naive Bayesian Learner

Signals
-------

Inputs:


   - Examples (ExampleTable)
      A table with training examples


Outputs:

   - Learner
      The naive Bayesian learning algorithm with settings as specified in
      the dialog.

   - Naive Bayesian Classifier
      Trained classifier (a subtype of Classifier)


Signal :obj:`Naive Bayesian Classifier` sends data only if the learning
data (signal :obj:`Examples` is present.

Description
-----------

This widget provides a graphical interface to the Naive Bayesian classifier.

As all widgets for classification, this widget provides a learner and
classifier on the output. Learner is a learning algorithm with settings
as specified by the user. It can be fed into widgets for testing learners,
for instance :ref:`Test Learners`. Classifier is a Naive Bayesian Classifier
(a subtype of a general classifier), built from the training examples on the
input. If examples are not given, there is no classifier on the output.

.. image:: images/NaiveBayes.png
   :alt: NaiveBayes Widget

Learner can be given a name under which it will appear in, say,
:ref:`Test Learners`. The default name is "Naive Bayes".

Next come the probability estimators. :obj:`Prior` sets the method used for
estimating prior class probabilities from the data. You can use either
:obj:`Relative frequency` or the :obj:`Laplace estimate`.
:obj:`Conditional (for discrete)` sets the method for estimating conditional
probabilities, besides the above two, conditional probabilities can be
estimated using the :obj:`m-estimate`; in this case the value of m should be
given as the :obj:`Parameter for m-estimate`. By setting it to
:obj:`<same as above>` the classifier will use the same method as for
estimating prior probabilities.

Conditional probabilities for continuous attributes are estimated using
LOESS. :obj:`Size of LOESS window` sets the proportion of points in the
window; higher numbers mean more smoothing.
:obj:`LOESS sample points` sets the number of points in which the function
is sampled.

If the class is binary, the classification accuracy may be increased
considerably by letting the learner find the optimal classification
threshold (option :obj:`Adjust threshold`). The threshold is computed from
the training data. If left unchecked, the usual threshold of 0.5 is used.

When you change one or more settings, you need to push :obj:`Apply`;
this will put the new learner on the output and, if the training examples
are given, construct a new classifier and output it as well.


Examples
--------

There are two typical uses of this widget. First, you may want to induce
the model and check what it looks like in a :ref:`Nomogram`.

.. image:: images/NaiveBayes-SchemaClassifier.png
   :alt: Naive Bayesian Classifier - Schema with a Classifier

The second schema compares the results of Naive Bayesian learner with
another learner, a C4.5 tree.

.. image:: images/C4.5-SchemaLearner.png
   :alt: Naive Bayesian Classifier - Schema with a Learner
