.. automodule:: Orange.classification

###################################
Classification (``classification``)
###################################

Induction of models in Orange is implemented through a two-class schema:
"learners" are classes that induce models, and classifiers represent
trained models. The learner holds the parameters that
are used for fitting the model. When learner is called with a data table,
it fits a model and returns an instance of classifier. Classifiers can be subsequently used to predict dependent values for new data instances.

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

All learners and classifiers, including regressors, are derived from the following two clases.

.. class:: Learner()

    Base class for all orange learners.

    .. method:: __call__(data)

        Fit a model and return it as an instance of :class:`Classifier`.

        This method is abstract and needs to be implemented on each learner.

.. class:: Classifier()

    Base class for all orange classifiers.

    .. method:: __call__(instance, return_type=GetValue)

        Classify a new instance using this model. Results depends upon
        the second parameter that must be one of the following.

	:obj:`Orange.classification.Classifier.GetValue`

	    Return value of the target class when performing prediction.

	:obj:`Orange.classification.Classifier.GetProbabilities`

	    Return probability of each target class when performing prediction.

	:obj:`Orange.classification.Classifier.GetBoth`

	    Return a tuple of target class value and probabilities for each class.

        This method is abstract and needs to be implemented on each
        classifier.

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

The classification module also contains a classifier that always predicts a
constant value regardless of given data instances. It is usually not used
directly but through other other learners and methods, such as
:obj:`~Orange.classification.majority.MajorityLearner`.

.. class:: ConstantClassifier

    ConstantClassifier always classifies to the same class and reports the
    same class probabilities.

    .. attribute:: class_var

        Class variable that the classifier predicts.

    .. attribute:: default_val

        Value returned by the classifier.

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
       
    .. method:: __call__(data, return_type)
        
        ConstantClassifier always returns the same prediction
        (:obj:`default_val` and/or :obj:`default_distribution`), regardless
        of the given data instance.



