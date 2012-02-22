.. index:: Testing, Sampling
.. automodule:: Orange.evaluation.testing

==================================
Sampling and Testing (``testing``)
==================================

There are many ways to test prediction models on data. Orange includes
methods for cross-validation, leave-one out, random sampling and learning
curves. This methods handle learning of models and prediction of new
examples; they return :obj:`ExperimentResults` which can be passed to
:obj:`~Orange.evaluation.scoring` functions to evaluate model.

.. literalinclude:: code/testing-example.py

Different evaluation techniques are implemented as instance methods of
:obj:`Evaluation` class. For ease of use, an instance of this class is
created at module loading time and instance methods are exposed as functions
with the same name in Orange.evaluation.testing namespace.

.. autoclass:: Evaluation
   :members:

.. autoclass:: ExperimentResults
    :members:

.. autoclass:: TestedExample
    :members:

Generating random numbers
=========================

Many evaluation

*stratified*
    Tells whether to stratify the random selections. Its default value is
    :obj:`orange.StratifiedIfPossible` which stratifies selections
    if the class variable is discrete and has no unknown values.

*random_generator*
    If evaluation method relies on randomness, parameter ``random_generator``
    can be used to either provide a random seed or an instance of
    :obj:`~Orange.misc.Random` which will be used to generate random numbers.

    By default, a new instance of random generator is constructed for each
    call of the method with random seed 0.

    If you use more than one method that is based on randomness,
    you can construct an instance of :obj:`~Orange.misc.Random` and pass it
    to all of them. This way, all methods will use different random numbers,
    but this numbers will be the same for each run of the script.

    For truly random number, set seed to a random number generated with
    python random generator. Since python's random generator is reset each
    time python is loaded with current system time as seed,
    results of the script will be different each time you run it.

*preprocessors*
    A list of preprocessors. It consists of tuples ``(c, preprocessor)``,
    where ``c`` determines whether the preprocessor will be applied
    to the learning set (``"L"``), test set (``"T"``) or to both
    (``"B"``). The latter is applied first, when the example set is still
    undivided. The ``"L"`` and ``"T"`` preprocessors are applied on the
    separated subsets. Preprocessing testing examples is allowed only
    on experimental procedures that do not report the TestedExample's
    in the same order as examples in the original set. The second item
    in the tuple, preprocessor can be either a pure Orange or a pure
    Python preprocessor, that is, any function or callable class that
    accepts a table of examples and weight, and returns a preprocessed
    table and weight.

    This example will demonstrate the devastating effect of 100% class
    noise on learning. ::

        classnoise = orange.Preprocessor_addClassNoise(proportion=1.0)
        res = Orange.evaluation.testing.proportion_test(learners, data, 0.7, 100, pps = [("L", classnoise)])

*store_classifiers (keyword argument)*
    If this flag is set, the testing procedure will store the constructed
    classifiers. For each iteration of the test (eg for each fold in
    cross validation, for each left out example in leave-one-out...),
    the list of classifiers is appended to the ExperimentResults'
    field classifiers.

    The script below makes 100 repetitions of 70:30 test and store the
    classifiers it induces. ::

        res = Orange.evaluation.testing.proportion_test(learners, data, 0.7,
        100, store_classifiers=1)


Knowing classes :obj:`TestedExample` that stores results of testing
for a single test example and :obj:`ExperimentResults` that stores a list of
TestedExamples along with some other data on experimental procedures
and classifiers used, is important if you would like to write your own
measures of quality of models, compatible the sampling infrastructure
provided by Orange. If not, you can skip the remainder of this page.



References
==========

Salzberg, S. L. (1997). On comparing classifiers: Pitfalls to avoid
and a recommended approach. Data Mining and Knowledge Discovery 1,
pages 317-328.

