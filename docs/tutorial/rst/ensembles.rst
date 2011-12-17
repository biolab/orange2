.. index:: ensembles
.. index:: 
   single: ensembles; bagging
.. index:: 
   single: ensembles; boosting

Ensemble learners
=================

Building ensemble classifiers in Orange is simple and easy. Starting
from learners/classifiers that can predict probabilities and, if
needed, use example weights, ensembles are actually wrappers that can
aggregate predictions from a list of constructed classifiers. These
wrappers behave exactly like other Orange learners/classifiers. We
will here first show how to use a module for bagging and boosting that
is included in Orange distribution (:py:mod:`Orange.ensemble` module), and
then, for a somehow more advanced example build our own ensemble
learner. Using this module, using it is very easy: you have to define
a learner, give it to bagger or booster, which in turn returns a new
(boosted or bagged) learner. Here goes an example (:download:`ensemble3.py <code/ensemble3.py>`,
uses :download:`promoters.tab <code/promoters.tab>`)::

   import orange, orngTest, orngStat, orngEnsemble
   data = orange.ExampleTable("promoters")
   
   majority = orange.MajorityLearner()
   majority.name = "default"
   knn = orange.kNNLearner(k=11)
   knn.name = "k-NN (k=11)"
   
   bagged_knn = orngEnsemble.BaggedLearner(knn, t=10)
   bagged_knn.name = "bagged k-NN"
   boosted_knn = orngEnsemble.BoostedLearner(knn, t=10)
   boosted_knn.name = "boosted k-NN"
   
   learners = [majority, knn, bagged_knn, boosted_knn]
   results = orngTest.crossValidation(learners, data, folds=10)
   print "        Learner   CA     Brier Score"
   for i in range(len(learners)):
       print ("%15s:  %5.3f  %5.3f") % (learners[i].name,
           orngStat.CA(results)[i], orngStat.BrierScore(results)[i])

Most of the code is used for defining and naming objects that learn,
and the last piece of code is to report evaluation results. Notice
that to bag or boost a learner, it takes only a single line of code
(like, ``bagged_knn = orngEnsemble.BaggedLearner(knn, t=10)``)!
Parameter ``t`` in bagging and boosting refers to number of
classifiers that will be used for voting (or, if you like better,
number of iterations by boosting/bagging). Depending on your random
generator, you may get something like::

           Learner   CA     Brier Score
           default:  0.473  0.501
       k-NN (k=11):  0.859  0.240
       bagged k-NN:  0.813  0.257
      boosted k-NN:  0.830  0.244


