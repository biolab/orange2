"""

.. index:: ensemble

Module orngEnsemble implements Breiman's bagging and Random Forest, 
and Freund and Schapire's boosting algorithms.


==================
Boosting
==================
.. index ensemble boosting
.. autoclass:: Orange.ensemble.bagging.BaggedLearner
   :members:

==================
Bagging
==================
.. index ensemble bagging
.. autoclass:: Orange.ensemble.boosting.BoostedLearner
  :members:

==================
Forest
==================
.. index ensemble forest
.. autoclass:: Orange.ensemble.forest.RandomForestLearner
  :members:

References
----------
* L Breiman. Bagging Predictors. `Technical report No. 421 \
    <http://www.stat.berkeley.edu/tech-reports/421.ps.Z>`_. University of \
    California, Berkeley, 1994.
* Y Freund, RE Schapire. `Experiments with a New Boosting Algorithm \
    <http://citeseer.ist.psu.edu/freund96experiments.html>`_. Machine\
    Learning: Proceedings of the Thirteenth International Conference (ICML'96), 1996.
* JR Quinlan. `Boosting, bagging, and C4.5 \
    <http://www.rulequest.com/Personal/q.aaai96.ps>`_ . In Proc. of 13th\
    National Conference on Artificial Intelligence (AAAI'96). pp. 725-730, 1996. 
* L Brieman. `Random Forests \
    <http://www.springerlink.com/content/u0p06167n6173512/>`_. \
    Machine Learning, 45, 5-32, 2001. 
* M Robnik-Sikonja. `Improving Random Forests \
    <http://lkm.fri.uni-lj.si/rmarko/papers/robnik04-ecml.pdf>`_. In \
    Proc. of European Conference on Machine Learning (ECML 2004),\
    pp. 359-370, 2004. [PDF]

Examples
========

.. literalinclude:: code/ensemble.py


"""

__all__ = ["bagging", "boosting", "forest"]
__docformat__ = 'restructuredtext'