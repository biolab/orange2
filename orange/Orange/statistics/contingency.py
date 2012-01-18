"""
.. index:: Contingency table

=================
Contingency table
=================

Contingency table contains conditional distributions. Unless explicitly
'normalized', they contain absolute frequencies, that is, the number of
instances with a particular combination of two variables' values. If they are
normalized by dividing each cell by the row sum, the represent conditional
probabilities of the column variable (here denoted as ``innerVariable``)
conditioned by the row variable (``outerVariable``).

Contingency tables are usually constructed for discrete variables. Tables
for continuous variables have certain limitations described in a :ref:`separate
section <contcont>`.

The example below loads the monks-1 data set and prints out the conditional
class distribution given the value of `e`.

.. literalinclude:: code/statistics-contingency.py
    :lines: 1-7

This code prints out::

    1 <0.000, 108.000>
    2 <72.000, 36.000>
    3 <72.000, 36.000>
    4 <72.000, 36.000> 

Contingencies behave like lists of distributions (in this case, class
distributions) indexed by values (of `e`, in this
example). Distributions are, in turn indexed by values (class values,
here). The variable `e` from the above example is called the outer
variable, and the class is the inner. This can also be reversed. It is
also possible to use features for both, outer and inner variable, so
the table shows distributions of one variable's values given the
value of another.  There is a corresponding hierarchy of classes:
:obj:`Table` is a base class for :obj:`VarVar` (both
variables are attributes) and :obj:`Class` (one variable is
the class).  The latter is the base class for
:obj:`VarClass` and :obj:`ClassVar`.

The most commonly used of the above classes is :obj:`VarClass` which
can compute and store conditional probabilities of classes given the feature value.

Contingency tables
==================

.. class:: Table

    Provides a base class for storing and manipulating contingency
    tables. Although it is not abstract, it is seldom used directly but rather
    through more convenient derived classes described below.

    .. attribute:: outerVariable

       Outer variable (:class:`Orange.data.variable.Variable`) whose values are
       used as the first, outer index.

    .. attribute:: innerVariable

       Inner variable(:class:`Orange.data.variable.Variable`), whose values are
       used as the second, inner index.
 
    .. attribute:: outerDistribution

        The marginal distribution (:class:`Distribution`) of the outer variable.

    .. attribute:: innerDistribution

        The marginal distribution (:class:`Distribution`) of the inner variable.
        
    .. attribute:: innerDistributionUnknown

        The distribution (:class:`distribution.Distribution`) of the inner variable for
        instances for which the outer variable was undefined. This is the
        difference between the ``innerDistribution`` and (unconditional)
        distribution of inner variable.
      
    .. attribute:: varType

        The type of the outer variable (:obj:`Orange.data.Type`, usually
        :obj:`Orange.data.variable.Discrete` or
        :obj:`Orange.data.variable.Continuous`); equals
        ``outerVariable.varType`` and ``outerDistribution.varType``.

    .. method:: __init__(outer_variable, inner_variable)
     
        Construct an instance of contingency table for the given pair of
        variables.
     
        :param outer_variable: Descriptor of the outer variable
        :type outer_variable: Orange.data.variable.Variable
        :param outer_variable: Descriptor of the inner variable
        :type inner_variable: Orange.data.variable.Variable
        
    .. method:: add(outer_value, inner_value[, weight=1])
    
        Add an element to the contingency table by adding ``weight`` to the
        corresponding cell.

        :param outer_value: The value for the outer variable
        :type outer_value: int, float, string or :obj:`Orange.data.Value`
        :param inner_value: The value for the inner variable
        :type inner_value: int, float, string or :obj:`Orange.data.Value`
        :param weight: Instance weight
        :type weight: float

    .. method:: normalize()

        Normalize all distributions (rows) in the table to sum to ``1``::
        
            >>> cont.normalize()
            >>> for val, dist in cont.items():
                   print val, dist

        Output: ::

            1 <0.000, 1.000>
            2 <0.667, 0.333>
            3 <0.667, 0.333>
            4 <0.667, 0.333>

        .. note::
       
            This method does not change the ``innerDistribution`` or
            ``outerDistribution``.
        
    With respect to indexing, contingency table is a cross between dictionary
    and a list. It supports standard dictionary methods ``keys``, ``values`` and
    ``items``. ::

        >> print cont.keys()
        ['1', '2', '3', '4']
        >>> print cont.values()
        [<0.000, 108.000>, <72.000, 36.000>, <72.000, 36.000>, <72.000, 36.000>]
        >>> print cont.items()
        [('1', <0.000, 108.000>), ('2', <72.000, 36.000>),
        ('3', <72.000, 36.000>), ('4', <72.000, 36.000>)] 

    Although keys returned by the above functions are strings, contingency can
    be indexed by anything that can be converted into values of the outer
    variable: strings, numbers or instances of ``Orange.data.Value``. ::

        >>> print cont[0]
        <0.000, 108.000>
        >>> print cont["1"]
        <0.000, 108.000>
        >>> print cont[orange.Value(data.domain["e"], "1")] 

    The length of the table equals the number of values of the outer
    variable. However, iterating through contingency
    does not return keys, as with dictionaries, but distributions. ::

        >>> for i in cont:
            ... print i
        <0.000, 108.000>
        <72.000, 36.000>
        <72.000, 36.000>
        <72.000, 36.000>
        <72.000, 36.000> 


.. class:: Class

    An abstract base class for contingency tables that contain the class,
    either as the inner or the outer variable.

    .. attribute:: classVar (read only)
    
        The class attribute descriptor; always equal to either
        :obj:`Table.innerVariable` or :obj:``Table.outerVariable``.

    .. attribute:: variable
    
        Variable; always equal either to either ``innerVariable`` or ``outerVariable``

    .. method:: add_var_class(variable_value, class_value[, weight=1])

        Add an element to contingency by increasing the corresponding count. The
        difference between this and :obj:`Table.add` is that the variable
        value is always the first argument and class value the second,
        regardless of which one is inner and which one is outer.

        :param variable_value: Variable value
        :type variable_value: int, float, string or :obj:`Orange.data.Value`
        :param class_value: Class value
        :type class_value: int, float, string or :obj:`Orange.data.Value`
        :param weight: Instance weight
        :type weight: float


.. class:: VarClass

    A class derived from :obj:`Class` in which the variable is
    used as :obj:`Table.outerVariable` and class as the
    :obj:`Table.innerVariable`. This form is a form suitable for
    computation of conditional class probabilities given the variable value.
    
    Calling :obj:`VarClass.add_var_class(v, c)` is equivalent to
    :obj:`Table.add(v, c)`. Similar as :obj:`Table`,
    :obj:`VarClass` can compute contingency from instances.

    .. method:: __init__(feature, class_variable)

        Construct an instance of :obj:`VarClass` for the given pair of
        variables. Inherited from :obj:`Table`.

        :param feature: Outer variable
        :type feature: Orange.data.variable.Variable
        :param class_attribute: Class variable; used as ``innerVariable``
        :type class_attribute: Orange.data.variable.Variable
        
    .. method:: __init__(feature, data[, weightId])

        Compute the contingency table from data.

        :param feature: Outer variable
        :type feature: Orange.data.variable.Variable
        :param data: A set of instances
        :type data: Orange.data.Table
        :param weightId: meta attribute with weights of instances
        :type weightId: int

    .. method:: p_class(value)

        Return the probability distribution of classes given the value of the
        variable.

        :param value: The value of the variable
        :type value: int, float, string or :obj:`Orange.data.Value`
        :rtype: Orange.statistics.distribution.Distribution


    .. method:: p_class(value, class_value)

        Returns the conditional probability of the class_value given the
        feature value, p(class_value|value) (note the order of arguments!)
        
        :param value: The value of the variable
        :type value: int, float, string or :obj:`Orange.data.Value`
        :param class_value: The class value
        :type value: int, float, string or :obj:`Orange.data.Value`
        :rtype: float

    .. literalinclude:: code/statistics-contingency3.py
        :lines: 1-23

    The inner and the outer variable and their relations to the class are
    as follows::

        Inner variable:  y
        Outer variable:  e
    
        Class variable:  y
        Feature:         e

    Distributions are normalized, and probabilities are elements from the
    normalized distributions. Knowing that the target concept is
    y := (e=1) or (a=b), distributions are as expected: when e equals 1, class 1
    has a 100% probability, while for the rest, probability is one third, which
    agrees with a probability that two three-valued independent features
    have the same value. ::

        Distributions:
          p(.|1) = <0.000, 1.000>
          p(.|2) = <0.662, 0.338>
          p(.|3) = <0.659, 0.341>
          p(.|4) = <0.669, 0.331>
    
        Probabilities of class '1'
          p(1|1) = 1.000
          p(1|2) = 0.338
          p(1|3) = 0.341
          p(1|4) = 0.331
    
        Distributions from a matrix computed manually:
          p(.|1) = <0.000, 1.000>
          p(.|2) = <0.662, 0.338>
          p(.|3) = <0.659, 0.341>
          p(.|4) = <0.669, 0.331>


.. class:: ClassVar

    :obj:`ClassVar` is similar to :obj:`VarClass` except
    that the class is outside and the variable is inside. This form of
    contingency table is suitable for computing conditional probabilities of
    variable given the class. All methods get the two arguments in the same
    order as :obj:`VarClass`.

    .. method:: __init__(feature, class_variable)

        Construct an instance of :obj:`VarClass` for the given pair of
        variables. Inherited from :obj:`Table`, except for the reversed
        order of arguments.

        :param feature: Outer variable
        :type feature: Orange.data.variable.Variable
        :param class_variable: Class variable
        :type class_variable: Orange.data.variable.Variable
        
    .. method:: __init__(feature, data[, weightId])

        Compute contingency table from the data.

        :param feature: Descriptor of the outer variable
        :type feature: Orange.data.variable.Variable
        :param data: A set of instances
        :type data: Orange.data.Table
        :param weightId: meta attribute with weights of instances
        :type weightId: int

    .. method:: p_attr(class_value)

        Return the probability distribution of variable given the class.

        :param class_value: The value of the variable
        :type class_value: int, float, string or :obj:`Orange.data.Value`
        :rtype: Orange.statistics.distribution.Distribution

    .. method:: p_attr(value, class_value)

        Returns the conditional probability of the value given the
        class, p(value|class_value).

        :param value: Value of the variable
        :type value: int, float, string or :obj:`Orange.data.Value`
        :param class_value: Class value
        :type value: int, float, string or :obj:`Orange.data.Value`
        :rtype: float

    .. literalinclude:: code/statistics-contingency4.py
        :lines: 1-27

    The role of the feature and the class are reversed compared to
    :obj:`ClassVar`::
    
        Inner variable:  e
        Outer variable:  y
    
        Class variable:  y
        Feature:         e
    
    Distributions given the class can be printed out by calling :meth:`p_attr`.
    
    .. literalinclude:: code/statistics-contingency4.py
        :lines: 30-31
    
    will print::
        p(.|0) = <0.000, 0.333, 0.333, 0.333>
        p(.|1) = <0.500, 0.167, 0.167, 0.167>
    
    If the class value is '0', the attribute `e` cannot be `1` (the first
    value), while distribution across other values is uniform.  If the class
    value is `1`, `e` is `1` for exactly half of instances, and distribution of
    other values is again uniform.

.. class:: VarVar

    Contingency table in which none of the variables is the class.  The class
    is derived from :obj:`Table`, and adds an additional constructor and
    method for getting conditional probabilities.

    .. method:: VarVar(outer_variable, inner_variable)

        Inherited from :obj:`Table`.

    .. method:: __init__(outer_variable, inner_variable, data[, weightId])

        Compute the contingency from the given instances.

        :param outer_variable: Outer variable
        :type outer_variable: Orange.data.variable.Variable
        :param inner_variable: Inner variable
        :type inner_variable: Orange.data.variable.Variable
        :param data: A set of instances
        :type data: Orange.data.Table
        :param weightId: meta attribute with weights of instances
        :type weightId: int

    .. method:: p_attr(outer_value)

        Return the probability distribution of the inner variable given the
        outer variable value.

        :param outer_value: The value of the outer variable
        :type outer_value: int, float, string or :obj:`Orange.data.Value`
        :rtype: Orange.statistics.distribution.Distribution
 
    .. method:: p_attr(outer_value, inner_value)

        Return the conditional probability of the inner_value
        given the outer_value.

        :param outer_value: The value of the outer variable
        :type outer_value: int, float, string or :obj:`Orange.data.Value`
        :param inner_value: The value of the inner variable
        :type inner_value: int, float, string or :obj:`Orange.data.Value`
        :rtype: float

    The following example investigates which material is used for
    bridges of different lengths.
    
    .. literalinclude:: code/statistics-contingency5.py
        :lines: 1-17

    Short bridges are mostly wooden or iron, and the longer (and most of the
    middle sized) are made from steel::
    
        SHORT:
           WOOD (56%)
           IRON (44%)
    
        MEDIUM:
           WOOD (9%)
           IRON (11%)
           STEEL (79%)
    
        LONG:
           STEEL (100%)
    
    As all other contingency tables, this one can also be computed "manually".
    
    .. literalinclude:: code/statistics-contingency5.py
        :lines: 18-


Contingencies for entire domain
===============================

A list of contingency tables, either :obj:`VarClass` or
:obj:`ClassVar`.

.. class:: Domain

    .. method:: __init__(data[, weightId=0, classOuter=0|1])

        Compute a list of contingency tables.

        :param data: A set of instances
        :type data: Orange.data.Table
        :param weightId: meta attribute with weights of instances
        :type weightId: int
        :param classOuter: `True`, if class is the outer variable
        :type classOuter: bool

        .. note::
        
            ``classIsOuter`` cannot be given as positional argument,
            but needs to be passed by keyword.

    .. attribute:: classIsOuter (read only)

        Tells whether the class is the outer or the inner variable.

    .. attribute:: classes

        Contains the distribution of class values on the entire dataset.

    .. method:: normalize()

        Call normalize for all contingencies.

    The following script prints the contingency tables for features
    "a", "b" and "e" for the dataset Monk 1.
        
    .. literalinclude:: code/statistics-contingency8.py
        :lines: 9

    Contingency tables of type :obj:`VarClass` give
    the conditional distributions of classes, given the value of the variable.
    
    .. literalinclude:: code/statistics-contingency8.py
        :lines: 12- 

.. _contcont:

Contingency tables for continuous variables
===========================================

If the outer variable is continuous, the index must be one of the
values that do exist in the contingency table; other values raise an
exception:

.. literalinclude:: code/statistics-contingency6.py
    :lines: 1-4,17-

Since even rounding can be a problem, the only safe way to get the key
is to take it from from the contingencies' ``keys``.

Contingency tables with discrete outer variable and continuous inner variables
are more useful, since methods :obj:`ContingencyClassVar.p_class`
and :obj:`ContingencyVarClass.p_attr` use the primitive density estimation
provided by :obj:`Orange.statistics.distribution.Distribution`.

For example, :obj:`ClassVar` on the iris dataset can return the
probability of the sepal length 5.5 for different classes:

.. literalinclude:: code/statistics-contingency7.py

The script outputs::

    Estimated frequencies for e=5.5
      f(5.5|Iris-setosa) = 2.000
      f(5.5|Iris-versicolor) = 5.000
      f(5.5|Iris-virginica) = 1.000

"""

from Orange.core import Contingency as Table
from Orange.core import ContingencyAttrAttr as VarVar
from Orange.core import ContingencyClass as Class
from Orange.core import ContingencyAttrClass as VarClass
from Orange.core import ContingencyClassAttr as ClassVar

from Orange.core import DomainContingency as Domain
