.. py:currentmodule:: Orange.classification

Constant Classifier
-------------------

Constant classifier always predicts the same value. It is constructed
by various learners such as
:obj:`~Orange.classification.majority.MajorityLearner`, and also used
in other places.

.. class:: ConstantClassifier

    Always predict the specified ``default_val`` or
    ``default_distribution``, disregarding the instance.

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



