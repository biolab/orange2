.. automodule:: Orange.classification

###################################
Classification (``classification``)
###################################

Induction of models in Orange is implemented through a two-class
schema. A learning algorithm is represented as an instance of a class
derived from :obj:`Orange.classification.Learner`. The learner stores
all parameters of the learning algorithm. When a learner is called
with some data, it fits a model of the kind specific to the learning
algorithm and returns it as a (new) instance of a class derived
:obj:`Orange.classification.Classifier` that holds parameters of the model.

.. literalinclude:: code/bayes-run.py
   :lines: 7-

Orange implements various classifiers that are described in detail on
separate pages.

.. toctree::
   :maxdepth: 1

   Orange.classification.bayes
   Orange.classification.knn
   Orange.classification.logreg
   Orange.classification.lookup
   Orange.classification.majority
   Orange.classification.rules
   Orange.classification.svm
   Orange.classification.tree
   Orange.classification.classfromvar
   
Base classes
------------

All learning algorithms and prediction models are derived from the following two clases.

.. class:: Learner()

    Abstract base class for learning algorithms.

    .. method:: __call__(data)

        An abstract method that fits a model and returns it as an
        instance of :class:`Classifier`.


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


Constant Classifier
-------------------

The classification module also contains a classifier that always
predicts a constant value regardless of given data instances. This
classifier is constructed by different learners such as
:obj:`~Orange.classification.majority.MajorityLearner`, and by some other
methods.

.. class:: ConstantClassifier

    Predict the specified ``default_val`` or ``default_distribution``
    for any instance.

    .. attribute:: class_var

        Class variable that the classifier predicts.

    .. attribute:: default_val

        The value returned by the classifier.

    .. attribute:: default_distribution

        Class probabilities returned by the classifier.
    
    .. method:: __init__(variable, value, distribution)

        Constructor can be called without arguments, with a
        variable, value or both. If the value is given and is of type
        :obj:`Orange.data.Value`, its attribute
        :obj:`Orange.data.Value.variable` will either be used for
        initializing
        :obj:`~Orange.classification.ConstantClassifier.variable` or
        checked against it, if :obj:`variable` is given as an
        argument.
        
        :param variable: Class variable that the classifier predicts.
        :type variable: :obj:`Orange.feature.Descriptor`
        :param value: Value returned by the classifier.
        :type value: :obj:`Orange.data.Value` or int (index) or float
        :param distribution: Class probabilities returned by the classifier.
        :type dstribution: :obj:`Orange.statistics.distribution.Distribution`
       
    .. method:: __call__(instance, return_type)
        
        Return :obj:`default_val` and/or :obj:`default_distribution`
        (depending upon :obj:`return_type`) disregarding the
        :obj:`instance`.



