.. index: classification from variable
.. index:
   single: classification; classification from variable

************************
Classifier from variable 
************************

:obj:`~Orange.classification.ClassifierFromVar` and
:obj:`~Orange.classification.ClassifierFromVarFD` are helper
classifiers used to compute variable's values from another variables. They are used, for instance, in discretization of continuous variables.

:obj:`~Orange.classification.ClassifierFromVarFD` retrieves the
feature value based on its position in the domain and
:obj:`~Orange.classification.ClassifierFromVar` retrieves the feature
with the given descriptor.

Both classifiers can be given a function to transform the value. In
discretization, for instance, the transformer computes the
corresponding discrete interval for a continuous value of the original
variable.


.. class:: ClassifierFromVar(which_var[, transformer])
    
    Return the value of variable :obj:`~ClassifierFromVar.which_var`;
    transform it by the :obj:`~ClassifierFromVar.transformer`, if it
    is given.
 
    .. attribute:: which_var

        The descriptor of the feature whose value is returned.

    .. attribute:: transformer        

        The transformer for the value. It should be a class derived
        from :obj:`~Orange.data.utils.TransformValue` or a function
        written in Python.

    .. attribute:: transform_unknowns

        Defines the treatment of missing values.

    .. attribute:: distribution_for_unknown

        The distribution that is returned when the
        :obj:`~ClassifierFromVar.which_var`'s value is undefined and
        :obj:`~ClassifierFromVar.transform_unknowns` is ``False``.

    .. method:: __call__(inst[, result_type])

        Return ``transformer(instance[which_var])``. The value of
        :obj:`~ClassifierFromVar.which_var` can be either an ordinary
        variable, a meta variable or a variable which is not defined
        for the instance but its descriptor has a
        :obj:`~Orange.feature.Descriptor.get_value_from` that can be
        used to compute the value.

        If the feature is not found or its value is missing, the
        missing value is passed to the transformer if
        :obj:`~ClassifierFromVar.transform_unknowns` is
        ``True``. Otherwise,
        :obj:`~ClassifierFromVar.distribution_for_unknown` is
        returned.

The following example demonstrates the use of the class on the Monk 1
dataset. It construct a new variable `e1` that has a value of `1`, when
`e` is `1`, and `not 1` otherwise.

.. literalinclude:: code/classifier-from-var-example.py
    :lines: 1-19



.. class:: ClassifierFromVarFD


    A class similar to
    :obj:`~Orange.classification.ClassifierFromVar` except that the
    variable is given by its index in the domain. The index can also
    be negative to denote a meta attribute.

    The only practical difference between the two classes is that this
    does not compute the value of the variable from other variables
    through the descriptor's
    :obj:`Orange.feature.Descriptor.get_value_from`.

    .. attribute:: domain (inherited from :obj:`ClassifierFromVarFD`)
    
        The domain to which the :obj:`position` applies.

    .. attribute:: position

        The position of the attribute in the domain or its meta-id.

    .. attribute:: transformer        

        The transformer for the value. It should be a class derived
        from :obj:`Orange.data.utils.TransformValue` or a function
        written in Python.

    .. attribute:: transform_unknowns

        Defines the treatment of missing values.

    .. attribute:: distribution_for_unknown

        The distribution that is returned when the `which_var`'s value
        is undefined and :obj:`transform_unknowns` is ``False``.

    The use of this class is similar to that of 
    :obj:`~Orange.classification.ClassifierFromVar`.

    .. literalinclude:: code/classifier-from-var-example.py
        :lines: 21-25
