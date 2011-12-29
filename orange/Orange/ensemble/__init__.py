"""

.. index:: ensemble

Module Orange.ensemble implements Breiman's bagging and Random Forest, 
and Freund and Schapire's boosting algorithms.


*******
Bagging
*******

.. index:: bagging
.. index::
   single: ensemble; ensemble

.. autoclass:: Orange.ensemble.bagging.BaggedLearner
   :members:
   :show-inheritance:

.. autoclass:: Orange.ensemble.bagging.BaggedClassifier
   :members:
   :show-inheritance:

********
Boosting
********

.. index:: boosting
.. index::
   single: ensemble; boosting


.. autoclass:: Orange.ensemble.boosting.BoostedLearner
  :members:
  :show-inheritance:

.. autoclass:: Orange.ensemble.boosting.BoostedClassifier
   :members:
   :show-inheritance:

Example
=======
Let us try boosting and bagging on Lymphography data set and use TreeLearner
with post-pruning as a base learner. For testing, we use 10-fold cross
validation and observe classification accuracy.

:download:`ensemble.py <code/ensemble.py>` (uses :download:`lymphography.tab <code/lymphography.tab>`)

.. literalinclude:: code/ensemble.py
  :lines: 7-

Running this script, we may get something like::

    Classification Accuracy:
               tree: 0.764
       boosted tree: 0.770
        bagged tree: 0.790


*************
Random Forest
*************

.. index:: random forest
.. index::
   single: ensemble; random forest
   
.. autoclass:: Orange.ensemble.forest.RandomForestLearner
  :members:
  :show-inheritance:

.. autoclass:: Orange.ensemble.forest.RandomForestClassifier
  :members:
  :show-inheritance:


Example
========

The following script assembles a random forest learner and compares it
to a tree learner on a liver disorder (bupa) and housing data sets.

:download:`ensemble-forest.py <code/ensemble-forest.py>` (uses :download:`bupa.tab <code/bupa.tab>`, :download:`housing.tab <code/housing.tab>`)

.. literalinclude:: code/ensemble-forest.py
  :lines: 7-

Notice that our forest contains 50 trees. Learners are compared through 
3-fold cross validation::

    Classification: bupa.tab
    Learner  CA     Brier  AUC
    tree     0.586  0.829  0.575
    forest   0.710  0.392  0.752
    Regression: housing.tab
    Learner  MSE    RSE    R2
    tree     23.708  0.281  0.719
    forest   11.988  0.142  0.858

Perhaps the sole purpose of the following example is to show how to
access the individual classifiers once they are assembled into the
forest, and to show how we can assemble a tree learner to be used in
random forests. In the following example the best feature for decision
nodes is selected among three randomly chosen features, and maxDepth
and minExamples are both set to 5.

:download:`ensemble-forest2.py <code/ensemble-forest2.py>` (uses :download:`bupa.tab <code/bupa.tab>`)

.. literalinclude:: code/ensemble-forest2.py
  :lines: 7-

Running the above code would report on sizes (number of nodes) of the tree
in a constructed random forest.

    
Score Feature
=============

L. Breiman (2001) suggested the possibility of using random forests as a
non-myopic measure of feature importance.

The assessment of feature relevance with random forests is based on the
idea that randomly changing the value of an important feature greatly
affects instance's classification, while changing the value of an
unimportant feature does not affect it much. Implemented algorithm
accumulates feature scores over given number of trees. Importance of
all features for a single tree are computed as: correctly classified 
OOB instances minus correctly classified OOB instances when the feature is
randomly shuffled. The accumulated feature scores are divided by the
number of used trees and multiplied by 100 before they are returned.

.. autoclass:: Orange.ensemble.forest.ScoreFeature
  :members:

Computation of feature importance with random forests is rather slow and
importances for all features need to be computes simultaneously. When it 
is called to compute a quality of certain feature, it computes qualities
for all features in the dataset. When called again, it uses the stored 
results if the domain is still the same and the data table has not
changed (this is done by checking the data table's version and is
not foolproof; it will not detect if you change values of existing instances,
but will notice adding and removing instances; see the page on 
:class:`Orange.data.Table` for details).

:download:`ensemble-forest-measure.py <code/ensemble-forest-measure.py>` (uses :download:`iris.tab <code/iris.tab>`)

.. literalinclude:: code/ensemble-forest-measure.py
  :lines: 7-

Corresponding output::

    DATA:iris.tab

    first: 3.91, second: 0.38

    different random seed
    first: 3.39, second: 0.46

    All importances:
       sepal length:   3.39
        sepal width:   0.46
       petal length:  30.15
        petal width:  31.98

References
-----------
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
