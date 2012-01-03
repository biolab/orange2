.. index:: Testing, Sampling
.. automodule:: Orange.evaluation.testing

==================================
Sampling and Testing (``testing``)
==================================

This module includes functions for data sampling and splitting, and for
testing learners. It implements cross-validation, leave-one out, random
sampling and learning curves. All functions return their results in the same
format - an instance of :obj:`ExperimentResults`, described at the end of the
page, or, in case of learning curves, a list of :obj:`ExperimentResults`. This
object(s) can be passed to statistical function for model evaluation
(classification accuracy, Brier score, ROC analysis...) available in
module :obj:`Orange.evaluation.scoring`.

Your scripts will thus basically conduct experiments using methods of
:obj:`Evaluation` class and functions in  :obj:`Orange.evaluation.testing`,
covered on this page and then evaluate the results by functions in
:obj:`Orange.evaluation.scoring`. For those interested in writing their own
statistical measures of the quality of models,
description of :obj:`TestedExample` and :obj:`ExperimentResults` are
available at the end of this page.

.. note:: Orange has been "de-randomized". Running the same script twice
    will generally give the same results, unless special care is taken to
    randomize it. This is opposed to the previous versions where special
    care needed to be taken to make experiments repeatable.
    See arguments :obj:`randseed` and :obj:`randomGenerator` for the
    explanation.

Example scripts in this section suppose that the data is loaded and a
list of learning algorithms is prepared.

part of :download:`testing-test.py <code/testing-test.py>` (uses :download:`voting.tab <code/voting.tab>`)

.. literalinclude:: code/testing-test.py
    :start-after: import random
    :end-before: def printResults(res)

After testing is done, classification accuracies can be computed and
printed by the following function.

.. literalinclude:: code/testing-test.py
    :pyobject: printResults

Common Arguments
================

Many function in this module use a set of common arguments, which we define here.

*learners*
    A list of learning algorithms. These can be either pure Orange objects
    (such as :obj:`Orange.classification.bayes.NaiveLearner`) or Python
    classes or functions written in pure Python (anything that can be
    called with the same arguments and results as Orange's classifiers
    and performs similar function).

*examples, learnset, testset*
    Examples, given as an :obj:`Orange.data.Table` (some functions need an undivided
    set of examples while others need examples that are already split
    into two sets). If examples are weighted, pass them as a tuple
    ``(examples, weightID)``. Weights are respected by learning and testing,
    but not by sampling. When selecting 10% of examples, this means 10%
    by number, not by weights. There is also no guarantee that sums
    of example weights will be (at least roughly) equal for folds in
    cross validation.

*strat*
    Tells whether to stratify the random selections. Its default value is
    :obj:`orange.StratifiedIfPossible` which stratifies selections
    if the class variable is discrete and has no unknown values.

*randseed (obsolete: indicesrandseed), randomGenerator*
    Random seed (``randseed``) or random generator (``randomGenerator``) for
    random selection of examples. If omitted, random seed of 0 is used and
    the same test will always select the same examples from the example
    set. There are various slightly different ways to randomize it.

    *
      Set ``randomGenerator`` to :obj:`orange.globalRandom`. The function's
      selection will depend upon Orange's global random generator that
      is reset (with random seed 0) when Orange is imported. The Script's
      output will therefore depend upon what you did after Orange was
      first imported in the current Python session. ::

          res = Orange.evaluation.testing.proportion_test(learners, data, 0.7,
              randomGenerator=orange.globalRandom)

    *
      Construct a new :obj:`orange.RandomGenerator`. The code below,
      for instance, will produce different results in each iteration,
      but overall the same results each time it's run.

      .. literalinclude:: code/testing-test.py
        :start-after: but the same each time the script is run
        :end-before: # End

    *
      Set the random seed (argument ``randseed``) to a random
      number. Python has a global random generator that is reset when
      Python is loaded, using the current system time for a seed. With this,
      results will be (in general) different each time the script is run.


      .. literalinclude:: code/testing-test.py
        :start-after: proportionsTest that will give different results each time it is run
        :end-before: # End


      The same module also provides random generators as object, so
      that you can have independent local random generators in case you
      need them.

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

*proportions*
    Gives the proportions of learning examples at which the tests are
    to be made, where applicable. The default is ``[0.1, 0.2, ..., 1.0]``.

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

Sampling and Testing Functions
==============================

.. autoclass:: Evaluation
   :members:

Classes
=======

Knowing classes :obj:`TestedExample` that stores results of testing
for a single test example and :obj:`ExperimentResults` that stores a list of
TestedExamples along with some other data on experimental procedures
and classifiers used, is important if you would like to write your own
measures of quality of models, compatible the sampling infrastructure
provided by Orange. If not, you can skip the remainder of this page.

.. autoclass:: TestedExample
    :members:

.. autoclass:: ExperimentResults
    :members:

References
==========

Salzberg, S. L. (1997). On comparing classifiers: Pitfalls to avoid
and a recommended approach. Data Mining and Knowledge Discovery 1,
pages 317-328.

