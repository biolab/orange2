Learners in Python
==================

.. index::
   single: classifiers; in Python

Orange comes with plenty classification and regression algorithms. But its also fun to make the new ones. You can build them anew, or wrap existing learners and add some preprocessing to construct new variants. Notice that learners in Orange have to adhere to certain rules. Let us observe them on a classification algorithm::

   >>> import Orange
   >>> data = Orange.data.Table("titanic")
   >>> learner = Orange.classification.logreg.LogRegLearner()
   >>> classifier = learner(data)
   >>> classifier(data[0])
   <orange.Value 'survived'='no'>

When learner is given the data it returns a predictor. In our case, classifier. Classifiers are passed data instances and return a value of a class. They can also return probability distribution, or this together with a class value::

   >>> classifier(data[0], Orange.classification.Classifier.GetProbabilities)
   Out[26]: <0.593, 0.407>
   >>> classifier(data[0], Orange.classification.Classifier.GetBoth)
   Out[27]: (<orange.Value 'survived'='no'>, <0.593, 0.407>)

Regression is similar, just that the regression model would return only the predicted continuous value.

Notice also that the constructor for the learner can be given the data, and in that case it will construct a classifier (what else could it do?)::

   >>> classifier = Orange.classification.logreg.LogRegLearner(data)
   >>> classifier(data[42])
   <orange.Value 'survived'='no'>

Now we are ready to build our own learner. We will do this for a classification problem.

Classifier with Feature Selection
---------------------------------

Consider a naive Bayesian classifiers. They do perform well, but could loose accuracy when there are many features, especially when these are correlated. Feature selection can help. We may want to wrap naive Bayesian classifier with feature subset selection, such that it would learn only from the few most informative features. We will assume the data contains only discrete features and will score them with information gain. Here is an example that sets the scorer (``gain``) and uses it to find best five features from the classification data set:

.. literalinclude:: code/py-score-features.py
   :lines: 3-

We need to incorporate the feature selection within the learner, at the point where it gets the data. Learners for classification tasks inherit from ``Orange.classification.PyLearner``:

.. literalinclude:: code/py-small.py
   :lines: 3-17

The initialization part of the learner (``__init__``) simply stores the based learner (in our case a naive Bayesian classifier), the name of the learner and a number of features we would like to use. Invocation of the learner (``__call__``) scores the features, stores the best one in the list (``best``), construct a data domain and then uses the one to transform the data (``Orange.data.Table(domain, data)``) by including only the set of the best features. Besides the most informative features we needed to include also the class. The learner then returns the classifier by using a generic classifier ``Orange.classification.PyClassifier``, where the actual prediction model is passed through ``classifier`` argument.

Note that classifiers in Orange also use the weight vector, which records the importance of training data items. This is useful for several algorithms, like boosting.

Let's check if this works::

   >>> data = Orange.data.Table("promoters")
   >>> s_learner = SmallLearner(m=3)
   >>> classifier = s_learner(data)
   >>> classifier(data[20])
   <orange.Value 'y'='mm'>
   >>> classifier(data[20], Orange.classification.Classifier.GetProbabilities)
   <0.439, 0.561>

It does! We constructed the naive Bayesian classifier with only three features. But how do we know what is the best number of features we could use? It's time to construct one more learner.

Estimation of Feature Set Size
------------------------------

Given a training data, what is the best number of features we could use with a training algorithm? We can estimate that through cross-validation, by checking possible feature set sizes and estimating how well does the classifier on such reduced feature set behave. When we are done, we use the feature sets size with best performance, and build a classifier on the entire training set. This procedure is often referred to as internal cross validation. We wrap it into a new learner:

.. literalinclude:: code/py-small.py
   :lines: 19-31

Again, our code stores the arguments at initialization (``__init__``). The learner invocation part selects the best value of parameter ``m``, the size of the feature set, and uses it to construct the final classifier.

We can now compare the three classification algorithms. That is, the base classifier (naive Bayesian), the classifier with a fixed number of selected features, and the classifier that estimates the optimal number of features from the training set:

.. literalinclude:: code/py-small.py
   :lines: 39-45

And the result? The classifier with feature set size wins (but not substantially. The results would be more pronounced if we would run this on the datasets with larger number of features)::

   opt_small: 0.942, small: 0.937, nbc: 0.933

