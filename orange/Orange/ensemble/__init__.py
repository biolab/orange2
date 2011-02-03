"""

.. index:: ensemble

Module orngEnsemble implements Breiman's bagging and Random Forest, 
and Freund and Schapire's boosting algorithms.


==================
Boosting
==================
.. index:: ensembleboosting
.. autoclass:: Orange.ensemble.bagging.BaggedLearner
   :members:

==================
Bagging
==================
.. index:: ensemblebagging
.. autoclass:: Orange.ensemble.boosting.BoostedLearner
  :members:

Example
========
Let us try boosting and bagging on Lymphography data set and use TreeLearner
with post-pruning as a base learner. For testing, we use 10-fold cross
validation and observe classification accuracy.

`ensemble.py`_ (uses `lymphography.tab`_)

.. literalinclude:: code/ensemble.py
  :lines: 7-

.. _lymphography.tab: code/lymphography.tab
.. _ensemble.py: code/ensemble.py

Running this script, we may get something like::

    Classification Accuracy:
               tree: 0.764
       boosted tree: 0.770
        bagged tree: 0.790


==================
Forest
==================

.. index:: randomforest
.. autoclass:: Orange.ensemble.forest.RandomForestLearner
  :members:


Example
========

The following script assembles a random forest learner and compares it
to a tree learner on a liver disorder (bupa) data set.

`ensemble-forest.py`_ (uses `buba.tab`_)

.. literalinclude:: code/ensemble-forest.py
  :lines: 7-

.. _buba.tab: code/buba.tab
.. _ensemble-forest.py: code/ensemble-forest.py

Notice that our forest contains 50 trees. Learners are compared through 
10-fold cross validation, and results reported on classification accuracy,
brier score and area under ROC curve::

    Learner  CA     Brier  AUC
    tree     0.588  0.823  0.578
    forest   0.713  0.383  0.763

Perhaps the sole purpose of the following example is to show how to access
the individual classifiers once they are assembled into the forest, and to 
show how we can assemble a tree learner to be used in random forests. The 
tree induction uses an feature subset split constructor, which we have 
borrowed from :class:`Orange.ensemble` and from which we have requested the
best feature for decision nodes to be selected from three randomly 
chosen features.

`ensemble-forest2.py`_ (uses `buba.tab`_)

.. literalinclude:: code/ensemble-forest2.py
  :lines: 7-

.. _ensemble-forest2.py: code/ensemble-forest2.py

Running the above code would report on sizes (number of nodes) of the tree
in a constructed random forest.

================
MeasureAttribute
================

L. Breiman (2001) suggested the possibility of using random forests as a
non-myopic measure of attribute importance.

Assessing relevance of features with random forests is based on the
idea that randomly changing the value of an important feature greatly
affects example's classification while changing the value of an
unimportant feature doen't affect it much. Implemented algorithm
accumulates feature scores over given number of trees. Importances of
all features for a single tree are computed as: correctly classified OOB
examples minus correctly classified OOB examples when an feature is
randomly shuffled. The accumulated feature scores are divided by the
number of used trees and multiplied by 100 before they are returned.

.. autoclass:: Orange.ensemble.forest.MeasureAttribute_randomForests
  :members:

Computation of feature importance with random forests is rather slow. Also, 
importances for all features need to be considered simultaneous. Since we
normally compute feature importance with random forests for all features in
the dataset, MeasureAttribute_randomForests caches the results. When it 
is called to compute a quality of certain feature, it computes qualities
for all features in the dataset. When called again, it uses the stored 
results if the domain is still the same and the example table has not
changed (this is done by checking the example tables version and is
not foolproof; it won't detect if you change values of existing examples,
but will notice adding and removing examples; see the page on 
:class:`Orange.data.Table` for details).

Caching will only have an effect if you use the same instance for all
features in the domain.

`ensemble-forest-measure.py`_ (uses `iris.tab`_)

.. literalinclude:: code/ensemble-forest-measure.py
  :lines: 7-

.. _ensemble-forest-measure.py: code/ensemble-forest-measure.py
.. _iris.tab: code/iris.tab

Corresponding output::

    first: 3.30, second: 0.57

    different random seed
    first: 3.52, second: 0.64

    All importances:
       sepal length:   3.52
        sepal width:   0.64
       petal length:  26.99
        petal width:  34.42


References
============
* L Breiman. Bagging Predictors. `Technical report No. 421 \
    <http://www.stat.berkeley.edu/tech-reports/421.ps.Z>`_. University of \
    California, Berkeley, 1994.
* Y Freund, RE Schapire. `Experiments with a New Boosting Algorithm \
    <http://citeseer.ist.psu.edu/freund96experiments.html>`_. Machine \
    Learning: Proceedings of the Thirteenth International Conference (ICML'96), 1996.
* JR Quinlan. `Boosting, bagging, and C4.5 \
    <http://www.rulequest.com/Personal/q.aaai96.ps>`_ . In Proc. of 13th \
    National Conference on Artificial Intelligence (AAAI'96). pp. 725-730, 1996. 
* L Brieman. `Random Forests \
    <http://www.springerlink.com/content/u0p06167n6173512/>`_.\
    Machine Learning, 45, 5-32, 2001. 
* M Robnik-Sikonja. `Improving Random Forests \
    <http://lkm.fri.uni-lj.si/rmarko/papers/robnik04-ecml.pdf>`_. In \
    Proc. of European Conference on Machine Learning (ECML 2004),\
    pp. 359-370, 2004.
"""

__all__ = ["bagging", "boosting", "forest"]
__docformat__ = 'restructuredtext'
import Orange.core as orange