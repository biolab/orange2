.. automodule:: Orange.classification

###################################
Classification (``classification``)
###################################

To facilitate correct evaluation, all classifiers in Orange consist of two
parts, a Learner and a Classifier. A learner is constructed with all
parameters that will be used for learning. When a data table is passed to its
__call__ method, a model is fitted to the data and returned in the form of a
Classifier, which is then used for predicting the dependent variable(s) of
new instances.

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

When developing new prediction models, one should extend :obj:`Learner` and
:obj:`Classifier`\. Code that infers the model from the data should be placed
in Learners's :obj:`~Learner.__call__` method. This method should
return a :obj:`Classifier`. Classifiers' :obj:`~Classifier.__call__` method
should  return the predicition; :class:`~Orange.data.Value`,
:class:`~Orange.statistics.distribution.Distribution` or a tuple with both
based on the value of the parameter :obj:`return_type`.
    

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

    Its constructor can be called without arguments, with a variable (for
    :obj:`class_var`), value (for :obj:`default_val`) or both. If the value
    is given and is of type :obj:`Orange.data.Value` (alternatives are an
    integer index of a discrete value or a continuous value), its attribute
    :obj:`Orange.data.Value.variable` will either be used for initializing
    :obj:`class_var` or checked against it, if :obj:`class_var` is given
    as an argument. 
    
    .. method:: __init__(class_var, default_val, default_distribution)

        The constructor can be called without arguments, with a variable
        (for :obj:`class_var`), value (for :obj:`default_val`) or both.
        If the value is given and is of type :obj:`Orange.data.Value`
        (alternatives are an integer index of a discrete value or a continuous
        value), its attribute :obj:`Orange.data.Value.variable` will either
        be used for initializing :obj:`class_var` or checked against it,
        if :obj:`class_var` is given as an argument. 
        
        :param class_var: Class variable that the classifier predicts.
        :param default_val: Value that is returned by the classifier.
        :param default_distribution: Class probabilities returned by the classifier.
       
    .. method:: __call__(instances, return_type)
        
        ConstantClassifier always returns the same prediction
        (:obj:`default_val` and/or :obj:`default_distribution`), regardless
        of the given data instance.

    :obj:`ConstantClassifier` also has the following attributes, which
    correspond to the constructor arguments described above.

    .. attribute:: class_var
    .. attribute:: default_val
    .. attribute:: default_distribution

