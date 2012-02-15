.. automodule:: Orange.classification

###################################
Classification (``classification``)
###################################

All Orange prediction models for classification consist of two parts,
a learner and a classifier. A learner is constructed with all parameters that
will be used for learning. When learner is called with a data table,
a model is fitted to the data and returned in the form of a
Classifier, which is then used for predicting the dependent variable(s) of
new instances.

.. literalinclude:: code/bayes-run.py
   :lines: 7-

Orange implements various classifiers that are described in detail on
separate pages.

.. toctree::
   :maxdepth: 2

   Orange.classification.bayes
   Orange.classification.knn
   Orange.classification.logreg
   Orange.classification.lookup
   Orange.classification.majority
   Orange.classification.rules
   Orange.classification.svm
   Orange.classification.tree
   Orange.classification.classfromvar
   
Constant Classifier
-------------------

The classification module also contains a classifier that always predicts
constant values regardless of given data instances. It is usually not used
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
       
    .. method:: __call__(instances, return_type)
        
        ConstantClassifier always returns the same prediction
        (:obj:`default_val` and/or :obj:`default_distribution`), regardless
        of the given data instance.



Writing custom Classifiers
--------------------------

When developing new prediction models, one should extend :obj:`Learner` and
:obj:`Classifier`\. Code that infers the model from the data should be placed
in learner's :obj:`~Learner.__call__` method. This method should
return a :obj:`Classifier`. Classifiers' :obj:`~Classifier.__call__` method
should  return the prediction; :class:`~Orange.data.Value`,
:class:`~Orange.statistics.distribution.Distribution` or a tuple with both
based on the value of the parameter :obj:`return_type`.

.. class:: Learner()

    Base class for all orange learners.

    .. method:: __call__(instances)

        Fit a model and return it as an instance of :class:`Classifier`.

        This method is abstract and needs to be implemented on each learner.

.. class:: Classifier()

    Base class for all orange classifiers.

    .. attribute:: GetValue

        Return value of the target class when performing prediction.

    .. attribute:: GetProbabilities

        Return probability of each target class when performing prediction.

    .. attribute:: GetBoth

        Return a tuple of target class value and probabilities for each class.


    .. method:: __call__(instance, return_type)

        Classify a new instance using this model.

        This method is abstract and needs to be implemented on each classifier.

        :param instance: data instance to be classified.
        :type instance: :class:`~Orange.data.Instance`

        :param return_type: what needs to be predicted
        :type return_type: :obj:`GetBoth`,
                           :obj:`GetValue`,
                           :obj:`GetProbabilities`

        :rtype: :class:`~Orange.data.Value`,
              :class:`~Orange.statistics.distribution.Distribution` or a
              tuple with both