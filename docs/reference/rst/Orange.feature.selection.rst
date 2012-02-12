.. :py:currentmodule:: Orange.feature.selection

#########################
Selection (``selection``)
#########################

.. index:: feature selection

.. index::
   single: feature; feature selection

Feature selection module contains several functions for selecting features based on they scores. A typical example is the function :obj:`select_best_n` that returns the best n features:

    .. literalinclude:: code/selection-best3.py
        :lines: 7-

    The script outputs::

        Best 3 features:
        physician-fee-freeze
        el-salvador-aid
        synfuels-corporation-cutback

The module also includes a learner that incorporates feature subset
selection.

--------------------------------------
Functions for feature subset selection
--------------------------------------

.. automethod:: Orange.feature.selection.best_n

.. automethod:: Orange.feature.selection.above_threshold

.. automethod:: Orange.feature.selection.select_best_n

.. automethod:: Orange.feature.selection.select_above_threshold

.. automethod:: Orange.feature.selection.select_relief(data, measure=Orange.feature.scoring.Relief(k=20, m=10), margin=0)

--------------------------------------
Learning with feature subset selection
--------------------------------------

.. autoclass:: Orange.feature.selection.FilteredLearner(base_learner, filter=FilterAboveThreshold(), name=filtered)
   :members:

.. autoclass:: Orange.feature.selection.FilteredClassifier
   :members:


--------------------------------------
Class wrappers for selection functions
--------------------------------------

.. autoclass:: Orange.feature.selection.FilterAboveThreshold(data=None, measure=Orange.feature.scoring.Relief(k=20, m=50), threshold=0.0)
   :members:

.. autoclass:: Orange.feature.selection.FilterBestN(data=None, measure=Orange.feature.scoring.Relief(k=20, m=50), n=5)
   :members:

.. autoclass:: Orange.feature.selection.FilterRelief(data=None, measure=Orange.feature.scoring.Relief(k=20, m=50), margin=0)
   :members:



.. rubric:: Examples

The following script defines a new Naive Bayes classifier, that
selects five best features from the data set before learning.
The new classifier is wrapped-up in a special class (see
<a href="../ofb/c_pythonlearner.htm">Building your own learner</a>
lesson in <a href="../ofb/default.htm">Orange for Beginners</a>). The
script compares this filtered learner with one that uses a complete
set of features.

:download:`selection-bayes.py<code/selection-bayes.py>`

.. literalinclude:: code/selection-bayes.py
    :lines: 7-

Interestingly, and somehow expected, feature subset selection
helps. This is the output that we get::

    Learner      CA
    Naive Bayes  0.903
    with FSS     0.940

We can do all of  he above by wrapping the learner using
<code>FilteredLearner</code>, thus
creating an object that is assembled from data filter and a base learner. When
given a data table, this learner uses attribute filter to construct a new
data set and base learner to construct a corresponding
classifier. Attribute filters should be of the type like
<code>orngFSS.FilterAboveThresh</code> or
<code>orngFSS.FilterBestN</code> that can be initialized with the
arguments and later presented with a data, returning new reduced data
set.

The following code fragment replaces the bulk of code
from previous example, and compares naive Bayesian classifier to the
same classifier when only a single most important attribute is
used.

:download:`selection-filtered-learner.py<code/selection-filtered-learner.py>`

.. literalinclude:: code/selection-filtered-learner.py
    :lines: 13-16

Now, let's decide to retain three features (change the code in <a
href="fss4.py">fss4.py</a> accordingly!), but observe how many times
an attribute was used. Remember, 10-fold cross validation constructs
ten instances for each classifier, and each time we run
FilteredLearner a different set of features may be
selected. <code>orngEval.CrossValidation</code> stores classifiers in
<code>results</code> variable, and <code>FilteredLearner</code>
returns a classifier that can tell which features it used (how
convenient!), so the code to do all this is quite short.

.. literalinclude:: code/selection-filtered-learner.py
    :lines: 25-

Running :download:`selection-filtered-learner.py <code/selection-filtered-learner.py>` with three features selected each
time a learner is run gives the following result::

    Learner      CA
    bayes        0.903
    filtered     0.956

    Number of times features were used in cross-validation:
     3 x el-salvador-aid
     6 x synfuels-corporation-cutback
     7 x adoption-of-the-budget-resolution
    10 x physician-fee-freeze
     4 x crime

Experiment yourself to see, if only one attribute is retained for
classifier, which attribute was the one most frequently selected over
all the ten cross-validation tests!

==========
References
==========

* K. Kira and L. Rendell. A practical approach to feature selection. In
  D. Sleeman and P. Edwards, editors, Proc. 9th Int'l Conf. on Machine
  Learning, pages 249{256, Aberdeen, 1992. Morgan Kaufmann Publishers.

* I. Kononenko. Estimating attributes: Analysis and extensions of RELIEF.
  In F. Bergadano and L. De Raedt, editors, Proc. European Conf. on Machine
  Learning (ECML-94), pages  171-182. Springer-Verlag, 1994.

* R. Kohavi, G. John: Wrappers for Feature Subset Selection, Artificial
  Intelligence, 97 (1-2), pages 273-324, 1997
