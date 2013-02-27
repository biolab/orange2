Classification
==============

.. index:: classification
.. index:: 
   single: data mining; supervised

Much of Orange is devoted to machine learning methods for classification, or supervised data mining. These methods rely on
the data with class-labeled instances, like that of senate voting. Here is a code that loads this data set, displays the first data instance and shows its predicted class (``republican``)::

   >>> data = Orange.data.Table("voting")
   >>> data[0]
   ['n', 'y', 'n', 'y', 'y', 'y', 'n', 'n', 'n', 'y', '?', 'y', 'y', 'y', 'n', 'y', 'republican']
   >>> data[0].get_class()
   <orange.Value 'party'='republican'>

Learners and Classifiers
------------------------

.. index::
   single: classification; learner
.. index::
   single: classification; classifier
.. index::
   single: classification; naive Bayesian classifier

Classification uses two types of objects: learners and classifiers. Learners consider class-labeled data and return a classifier. Given a data instance (a vector of feature values), classifiers return a predicted class::

    >>> import Orange
    >>> data = Orange.data.Table("voting")
    >>> learner = Orange.classification.bayes.NaiveLearner()
    >>> classifier = learner(data)
    >>> classifier(data[0])
    <orange.Value 'party'='republican'>

Above, we read the data, constructed a `naive Bayesian learner <http://en.wikipedia.org/wiki/Naive_Bayes_classifier>`_, gave it the data set to construct a classifier, and used it to predict the class of the first data item. We also use these concepts in the following code that predicts the classes of the first five instances in the data set:

.. literalinclude:: code/classification-classifier1.py
   :lines: 4-

The script outputs::

    republican; originally republican
    republican; originally republican
    republican; originally democrat
      democrat; originally democrat
      democrat; originally democrat

Naive Bayesian classifier has made a mistake in the third instance, but otherwise predicted correctly. No wonder, since this was also the data it trained from.

Probabilistic Classification
----------------------------

To find out what is the probability that the classifier assigns
to, say, democrat class, we need to call the classifier with
additional parameter that specifies the output type. If this is ``Orange.classification.Classifier.GetProbabilities``, the classifier will output class probabilities:

.. literalinclude:: code/classification-classifier2.py
   :lines: 4-

The output of the script also shows how badly the naive Bayesian classifier missed the class for the thrid data item::

   Probabilities for democrat:
   0.000; originally republican
   0.000; originally republican
   0.005; originally democrat
   0.998; originally democrat
   0.957; originally democrat

Cross-Validation
----------------

.. index:: cross-validation

Validating the accuracy of classifiers on the training data, as we did above, serves demonstration purposes only. Any performance measure that assess accuracy should be estimated on the independent test set. Such is also a procedure called `cross-validation <http://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_, which averages performance estimates across several runs, each time considering a different training and test subsets as sampled from the original data set:

.. literalinclude:: code/classification-cv.py
   :lines: 3-

.. index::
   single: classification; scoring
.. index::
   single: classification; area under ROC
.. index::
   single: classification; accuracy

Cross-validation is expecting a list of learners. The performance estimators also return a list of scores, one for every learner. There was just one learner in the script above, hence the list of size one was used. The script estimates classification accuracy and area under ROC curve. The later score is very high, indicating a very good performance of naive Bayesian learner on senate voting data set::

   Accuracy: 0.90
   AUC:      0.97


Handful of Classifiers
----------------------

Orange includes wide range of classification algorithms, including:

- logistic regression (``Orange.classification.logreg``)
- k-nearest neighbors (``Orange.classification.knn``)
- support vector machines (``Orange.classification.svm``)
- classification trees (``Orange.classification.tree``)
- classification rules (``Orange.classification.rules``)

Some of these are included in the code that estimates the probability of a target class on a testing data. This time, training and test data sets are disjoint:

.. index::
   single: classification; logistic regression
.. index::
   single: classification; trees
.. index::
   single: classification; k-nearest neighbors

.. literalinclude:: code/classification-other.py

For these five data items, there are no major differences between predictions of observed classification algorithms::

   Probabilities for republican:
   original class  tree      k-NN      lr       
   republican      0.949     1.000     1.000
   republican      0.972     1.000     1.000
   democrat        0.011     0.078     0.000
   democrat        0.015     0.001     0.000
   democrat        0.015     0.032     0.000

The following code cross-validates several learners. Notice the difference between this and the code above. Cross-validation requires learners, while in the script above, learners were immediately given the data and the calls returned classifiers.

.. literalinclude:: code/classification-cv2.py

Logistic regression wins in area under ROC curve::

            nbc  tree lr  
   Accuracy 0.90 0.95 0.94
   AUC      0.97 0.94 0.99

Reporting on Classification Models
----------------------------------

Classification models are objects, exposing every component of its structure. For instance, one can traverse classification tree in code and observe the associated data instances, probabilities and conditions. It is often, however, sufficient, to provide textual output of the model. For logistic regression and trees, this is illustrated in the script below:

.. literalinclude:: code/classification-models.py

The logistic regression part of the output is::
   
   class attribute = survived
   class values = <no, yes>

         Feature       beta  st. error     wald Z          P OR=exp(beta)
   
       Intercept      -1.23       0.08     -15.15      -0.00
    status=first       0.86       0.16       5.39       0.00       2.36
   status=second      -0.16       0.18      -0.91       0.36       0.85
    status=third      -0.92       0.15      -6.12       0.00       0.40
       age=child       1.06       0.25       4.30       0.00       2.89
      sex=female       2.42       0.14      17.04       0.00      11.25

Trees can also be rendered in `dot <http://en.wikipedia.org/wiki/DOT_language>`_::

   tree.dot(file_name="0.dot", node_shape="ellipse", leaf_shape="box")

Following figure shows an example of such rendering.

.. image:: files/tree.png
   :alt: A graphical presentation of a classification tree
