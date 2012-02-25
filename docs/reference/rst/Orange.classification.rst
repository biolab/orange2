.. automodule:: Orange.classification

###################################
Classification (``classification``)
###################################

Induction of models in Orange is implemented through a two-class
schema. A learning algorithm is represented by an instance of a class
derived from :obj:`Orange.classification.Learner`. The learner stores
all parameters of the learning algorithm. Induced models are
represented by instances of classes derived from
:obj:`Orange.classification.Classifier`.

Therefore, to induce models from data, one first needs to construct
the instance representing a learning algorithm
(e.g. :obj:`~Orange.classification.tree.TreeLearner`) and set its
parameters. Calling the learner with some training data returns a
classifier (e.g. :obj:`~Orange.classification.tree.TreeClassifier`). The
learner does not "learn" to classify but constructs classifiers.

.. literalinclude:: code/bayes-run.py
   :lines: 7-

To simplify the procedure, the learner's constructor can also be given
training data, in which case it fits and returns a model (an instance
of :obj:`~Orange.classification.Classifier`) instead of a learner::

    classifier = Orange.classification.bayes.NaiveLearner(titanic)


Orange contains a number of learning algorithms described in detail on
separate pages.

.. toctree::
   :maxdepth: 1

   Orange.classification.bayes
   Orange.classification.knn
   Orange.classification.rules
   Orange.classification.svm
   Orange.classification.tree
   Orange.classification.logreg
   Orange.classification.majority
   Orange.classification.lookup
   Orange.classification.classfromvar
   Orange.classification.constant
   

All learning algorithms and prediction models are derived from the following two clases.

.. class:: Learner()

    Abstract base class for learning algorithms.

    .. method:: __call__(data[, weightID])

        An abstract method that fits a model and returns it as an
        instance of :class:`Classifier`. The first argument gives the
        data (as :obj:`Orange.data.Table` and the optional second
        argument gives the id of the meta attribute with instance
        weights.


.. class:: Classifier()

    Abstract base class for prediction models (both classifiers and regressors).

    .. method:: __call__(instance, return_type=GetValue)

        Classify a new instance using this model. Results depends upon
        the second parameter that must be one of the following.

	:obj:`Orange.classification.Classifier.GetValue`

	    Return value of the target class when performing prediction.

	:obj:`Orange.classification.Classifier.GetProbabilities`

	    Return probability of each target class when performing prediction.

	:obj:`Orange.classification.Classifier.GetBoth`

	    Return a tuple of target class value and probabilities for each class.

        
        :param instance: data instance to be classified.
        :type instance: :class:`~Orange.data.Instance`

        :param return_type: what needs to be predicted
        :type return_type: :obj:`GetBoth`,
                           :obj:`GetValue`,
                           :obj:`GetProbabilities`

        :rtype: :class:`~Orange.data.Value`,
              :class:`~Orange.statistics.distribution.Distribution` or a
              tuple with both
