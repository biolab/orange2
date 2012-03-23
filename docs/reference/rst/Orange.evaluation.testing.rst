.. index:: Testing, Sampling
.. automodule:: Orange.evaluation.testing

==================================
Sampling and Testing (``testing``)
==================================

Module :obj:`Orange.evaluation.testing` contains methods for
cross-validation, leave-one out, random sampling and learning
curves. These procedures split the data onto training and testing set
and use the training data to induce models; models then make
predictions for testing data. Predictions are collected in
:obj:`ExperimentResults`, together with the actual classes and some
other data. The latter can be given to functions
:obj:`~Orange.evaluation.scoring` that compute the performance scores
of models.

.. literalinclude:: code/testing-example.py

The following call makes 100 iterations of 70:30 test and stores all the
induced classifiers. ::

     res = Orange.evaluation.testing.proportion_test(learners, iris, 0.7, 100, store_classifiers=1)

Different evaluation techniques are implemented as instance methods of
:obj:`Evaluation` class. For ease of use, an instance of this class is
created at module loading time and instance methods are exposed as functions
in :obj:`Orange.evaluation.testing`.

Randomness in tests
===================

If evaluation method uses random sampling, parameter
``random_generator`` can be used to either provide either a random
seed or an instance of :obj:`~Orange.misc.Random`. If omitted, a new
instance of random generator is constructed for each call of the
method with random seed 0.

.. note::

    Running the same script twice will generally give the same
    results.

For conducting a repeatable set of experiments, construct an instance
of :obj:`~Orange.misc.Random` and pass it to all of them. This way,
all methods will use different random numbers, but they will be the
same for each run of the script.

For truly random number, set seed to a random number generated with
python random generator. Since python's random generator is reset each
time python is loaded with current system time as seed, results of the
script will be different each time you run it.

.. autoclass:: Evaluation

   .. automethod:: cross_validation

   .. automethod:: leave_one_out

   .. automethod:: proportion_test

   .. automethod:: test_with_indices

   .. automethod:: one_fold_with_indices

   .. automethod:: learn_and_test_on_learn_data

   .. automethod:: learn_and_test_on_test_data

   .. automethod:: learning_curve(learners, examples, cv_indices=None, proportion_indices=None, proportions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], preprocessors=(), random_generator=0, callback=None)

   .. automethod:: learning_curve_n(learners, examples, folds=10, proportions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], stratification=StratifiedIfPossible, preprocessors=(), random_generator=0, callback=None)

   .. automethod:: learning_curve_with_test_data(learners, learn_set, test_set, times=10, proportions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], stratification=StratifiedIfPossible, preprocessors=(), random_generator=0, store_classifiers=False, store_examples=False)

   .. automethod:: test_on_data

