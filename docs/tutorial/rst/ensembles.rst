.. index:: ensembles

Ensembles
=========

`Learning of ensembles <http://en.wikipedia.org/wiki/Ensemble_learning>`_ combines the predictions of separate models to gain in accuracy. The models may come from different training data samples, or may use different learners on the same data sets. Learners may also be diversified by changing their parameter sets.

In Orange, ensembles are simply wrappers around learners. They behave just like any other learner. Given the data, they return models that can predict the outcome for any data instance::

   >>> import Orange
   >>> data = Orange.data.Table("housing")
   >>> tree = Orange.classification.tree.TreeLearner()
   >>> btree = Orange.ensemble.bagging.BaggedLearner(tree)
   >>> btree
   BaggedLearner 'Bagging'
   >>> btree(data)
   BaggedClassifier 'Bagging'
   >>> btree(data)(data[0])
   <orange.Value 'MEDV'='24.6'>

The last line builds a predictor (``btree(data)``) and then uses it on a first data instance.

Most ensemble methods can wrap either classification or regression learners. Exceptions are task-specialized techniques such as boosting.

Bagging and Boosting
--------------------

.. index:: 
   single: ensembles; bagging

`Bootstrap aggregating <http://en.wikipedia.org/wiki/Bootstrap_aggregating>`_, or bagging, samples the training data uniformly and with replacement to train different predictors. Majority vote (classification) or mean (regression) across predictions then combines independent predictions into a single prediction. 

.. index:: 
   single: ensembles; boosting

In general, boosting is a technique that combines weak learners into a single strong learner. Orange implements `AdaBoost <http://en.wikipedia.org/wiki/AdaBoost>`_, which assigns weights to data instances according to performance of the learner. AdaBoost uses these weights to iteratively sample the instances to focus on those that are harder to classify. In the aggregation AdaBoost emphases individual classifiers with better performance on their training sets.

The following script wraps a classification tree in boosted and bagged learner, and tests the three learner through cross-validation:

.. literalinclude:: code/ensemble-bagging.py

The benefit of the two ensembling techniques, assessed in terms of area under ROC curve, is obvious::

    tree: 0.83
   boost: 0.90
    bagg: 0.91

Stacking
--------

.. index:: 
   single: ensembles; stacking

Consider we partition a training set into held-in and held-out set. Assume that our taks is prediction of y, either probability of the target class in classification or a real value in regression. We are given a set of learners. We train them on held-in set, and obtain a vector of prediction on held-out set. Each element of the vector corresponds to prediction of individual predictor. We can now learn how to combine these predictions to form a target prediction, by training a new predictor on a data set of predictions and true value of y in held-out set. The technique is called `stacked generalization <http://en.wikipedia.org/wiki/Ensemble_learning#Stacking>`_, or in short stacking. Instead of a single split to held-in and held-out data set, the vectors of predictions are obtained through cross-validation.

Orange provides a wrapper for stacking that is given a set of base learners and a meta learner:

.. literalinclude:: code/ensemble-stacking.py
   :lines: 3-

By default, the meta classifier is naive Bayesian classifier. Changing this to logistic regression may be a good idea as well::

    stack = Orange.ensemble.stacking.StackedClassificationLearner(base_learners, \
               meta_learner=Orange.classification.logreg.LogRegLearner)

Stacking is often better than each of the base learners alone, as also demonstrated by running our script::

   stacking: 0.967
      bayes: 0.933
       tree: 0.836
        knn: 0.947

Random Forests
--------------

.. index:: 
   single: ensembles; random forests

`Random forest <http://en.wikipedia.org/wiki/Random_forest>`_ ensembles tree predictors. The diversity of trees is achieved in randomization of feature selection for node split criteria, where instead of the best feature one is picked arbitrary from a set of best features. Another source of randomization is a bootstrap sample of data from which the threes are developed. Predictions from usually several hundred trees are aggregated by voting. Constructing so many trees may be computationally demanding. Orange uses a special tree inducer (Orange.classification.tree.SimpleTreeLearner, considered by default) optimized for speed in random forest construction: 

.. literalinclude:: code/ensemble-forest.py
   :lines: 3-

Random forests are often superior when compared to other base classification or regression learners::

   forest: 0.976
    bayes: 0.935
      knn: 0.952
