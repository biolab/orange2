.. py:currentmodule:: Orange.data

==================
Values (``Value``)
==================

.. class:: Value

    Contains a value of a variable. Value can be discrete,
    continuous or of some other type, like discrete or continuous
    distribution, or a string.

    Values are not assigned as references. For instance,
    when taking a value from a data instance (e.g. ``value = data[2]``),
    the result is a copy of the value; changing it does not affect the
    original data instance.

    .. attribute:: value

        The actual value.

        * If the `Value` is numeric, `value` is a number.

        * If it is discrete and the variable descriptor is present
          (:obj:`variable` is not :obj:`None`), `value` is a string with
          the symbolic value (as retrieved from
          :obj:`Orange.feature.Discrete.values`).

        * If it is discrete and the variable descriptor is unknown,
          `value` is a string with the number representing the value
          index, enclosed in brackets.

        * If the value is missing, `value` is a string ``'?'``, ``'~'``
          or ``'.'``  for don't know, don't care and other, respectively.

    .. attribute:: svalue

        Stores a value that corresponds to a variable that is neither
        :obj:`Orange.feature.Discrete` nor
        :obj:`Orange.feature.Continuous` or a distribution of a
        discrete or continuous value.

        This attribute is most often used for values of
        :obj:`Orange.feature.StringVariable`; in that case `svalue`
        is an instance of :obj:`Orange.data.StringValue`. Distributions
        are seldom used; when `svalue` contains a distribution, it is
        represented with an instance of
        :obj:`Orange.statistics.distribution.Discrete` or
        :obj:`Orange.statistics.distribution.Continuous`.

    .. attribute:: variable 

        Descriptor related to the value. Can be ``None``.

    .. attribute:: var_type

        Read-only descriptor that gives the variable type. Can be
        :obj:`Orange.feature.Type.Discrete`, :obj:`Orange.feature.Type.Continuous`,
        :obj:`Orange.feature.Type.String` or :obj:`Orange.feature.Type.Other`.

    .. attribute:: value_type

        Tells whether the value is known
        (:obj:`Orange.data.Value.Regular`) or not
        (:obj:`Orange.data.Value.DC`. :obj:`Orange.data.Value.DK`).

    .. method:: __init__(variable[, value])

         Construct a value with the given descriptor. Value can be any
         object that the descriptor understands. For discrete
         variables, this can be a string or an index (an integer number),
         for continuous it can be a string or a number. If the value is
         omitted, it is set to unknown (:obj:`Orange.data.Value.DK`). ::

             import Orange
             v = Orange.feature.Discrete("fruit", values=["plum", "orange", "apple"])
             an_apple = Orange.data.Value(v, "apple")
             another_apple = Orange.data.Value(v, 2)
             unknown_fruit = Orange.data.Value(v)

             v2 = Orange.feature.Continuous("iq")
             genius = Orange.data.Value(v2, 180)
             troll = Orange.data.Value(v2, "42")
             stranger = Orange.data.value(v2)

        :param variable: variable descriptor
        :type variables: Orange.feature.Descriptor
        :param value: A value
        :type value: int, float or string, or another type accepted by descriptor

    .. method:: __init__(value)

        Construct either a discrete value, if the argument is an
        integer, or a continuous one, if the argument is a
        floating-point number. Descriptor is set to ``None``.

	:param value: A value
	:type value: int or float

    .. method:: native()

        Return the value in a "native" Python form: strings for
        discrete and undefined values and floating-point numbers for
        continuous values.

    .. method:: is_DC()

        Return ``True`` if value is "don't care".

    .. method:: is_DK()

        Return ``True`` if value is "don't know".

    .. method:: is_special()

        Return ``True`` if value is either "don't know" or "don't care".

Casting and comparison of values
--------------------------------

Discrete and continuous values can be cast to Python types :obj:`int`,
:obj:`float`, :obj:`long`, to strings and to boolean values. A value is
considered true if it is not undefined. Continuous values support
arithmetic operations.

Values can be compared with each other or with ordinary numbers.
All discrete variables are treated as ordinal; values are
compared by their respective indices and not in alphabetical order
of their symbolic representations. When comparing values
corresponding to different descriptors, Orange checks whether the
order is unambiguous. Here are two such values::

    deg3 = Orange.feature.Discrete(
        "deg3", values=["little", "medium", "big"])
    deg4 = orange.feature.Discrete(
        "deg4", values=["tiny", "little", "big", "huge"])
    val3 = orange.Value(deg3)
    val4 = orange.Value(deg4)
    val3.value = "medium"
    val4.value = "little"

Given this order, "medium" and "little" can be compared since it is known,
from ``deg3``, that "little" is less than "medium". ::

    val3.value = "medium"
    val4.value = "huge"

These two values cannot be compared since they do not appear in the same
variable. (Orange cannot use transitivity to conclude that medium is
less than huge since medium is less than big and big is less than
huge.)

Two values also cannot be compared when they have different order in
the two variables.
