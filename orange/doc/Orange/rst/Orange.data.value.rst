.. py:currentmodule:: Orange.data

==================
Values (``Value``)
==================

.. class:: Value

    Contains a value of a variable. Value can be discrete,
    continuous or of some other type, like discrete or continuous
    distribution, or a string.

    Values are not references. For instance, when taking a value from
    a data instance (e.g. ``value = data[2]``), the result is a copy of
    the value; changing it does not affect the original data instance.

    .. attribute:: value

        The actual value. Values of discrete and continuous variables
        are internally stored as integers or floating point numbers,
        respectively. This attribute gives a number for continuous
        variables and strings for discrete. If variable descriptor
        (:obj:`variable`) is not :obj:`None`, the string is a symbolic value
        for the attribute, otherwise it contains a number enclosed in
        <> brackets. If value unknown, the result is a string '?', '~'
        or '.'  for don't know, don't care and other, respectively.

    .. attribute:: svalue

        Instance of :obj:`Orange.core.SomeValue`, such as
        :obj:`Orange.data.StringValue`,
        :obj:`Orange.statistics.distribution.Discrete` or
        :obj:`Orange.statistics.distribution.Continuous`, which is
        used for storing values of non-numeric and non-discrete types.

    .. attribute:: variable 

        Descriptor related to the value. Can be :obj:`None`.

    .. attribute:: var_type

        Read-only descriptor that gives the variable type. Can be
        ``Orange.data.Type.Discrete``, ``Orange.data.Type.Continuous``,
        ``Orange.data.Type.String`` or ``Orange.data.Type.Other``.

    .. attribute:: value_type

        Tells whether the value is known or not. Can be
       ``Orange.data.Value.Regular``, ``Orange.data.Value.DC``,
       ``Orange.data.Value.DK``.

    .. method:: __init__(variable[, value])

	 Construct a value with the given descriptor. Value can be any
	 object, that the descriptor understands. For discrete
	 variables, this can be a string or an index, for continuous
	 it can be a string or a number. If the value is omitted, it
	 is set to unknown (DK). ::

             import Orange
             v = Orange.data.variable.Discrete("fruit", values=["plum", "orange", "apple"])
             an_apple = Orange.data.Value(v, "apple")
             another_apple = Orange.data.Value(v, 2)
             unknown_fruit = Orange.data.Value(v)

             v2 = Orange.data.variable.Continuous("iq")
             genius = Orange.data.Value(v2, 180)
             troll = Orange.data.Value(v2, "42")
             stranger = Orange.data.value(v2)

        :param variable: variable descriptor
        :type variables: Orange.data.variable.Variable
	:param value: A value
	:type value: int, float or string, or another type accepted by the given descriptor

    .. method:: __init__(value)

        Construct either a discrete value, if the argument is an
        integer, or a continuous one, if the argument is a
        floating-point number.

	:param value: A value
	:type value: int or float

    .. method:: native()

        Return the value in a "native" Python form: strings for
        discrete and undefined values and floating-point numbers for
        continuous values.

    .. method:: is_DC()

        Return :obj:`True` if value is "don't care".

    .. method:: is_DK()

        Return :obj:`True` if value is "don't know".

    .. method:: is_special()

        Return :obj:`True` if value is either "don't know" or "don't
        care".

    Discrete and continuous values can be cast to Python types :obj:`int`,
    :obj:`float`, :obj:`long`, to strings and to boolean values. A value is
    considered true if it is not undefined. Continuous values support arithmetic
    operations.

    Values can be compared between themselves or with ordinary
    numbers. All discrete variables are treated as ordinal; values are
    compared by their respective indices and not in alphabetical order
    of their symbolic representations. When comparing values
    corresponding to different descriptors, Orange checks whether the
    order is unambiguous. Here are two such values::

        deg3 = Orange.data.variable.Discrete("deg3",
                                       values=["little", "medium", "big"])
        deg4 = orange.data.variable.Discrete("deg4",
                                       values=["tiny", "little", "big", "huge"])
        val3 = orange.Value(deg3)
        val4 = orange.Value(deg4)
        val3.value = "medium"
        val4.value = "little"

    Given this order, "medium" and "little" can be compared, since it is known,
    from ``deg3``, that "little" is less than "medium". ::

        val3.value = "medium"
        val4.value = "huge"

    These two values cannot be compared since they do not appear in the same
    variable. (Orange is not smart enough to discover that medium is less than
    big and huge is more than big.)

    Two values also cannot be compared when they have different order in the two variables. 
