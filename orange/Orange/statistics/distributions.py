"""

Orange has several classes for computing and storing basic statistics about
features, distributions and contingencies.

    
========================================
Basic Statistics for Continuous Features
========================================

The are two simple classes for computing basic statistics
for continuous features, such as their minimal and maximal value
or average: :class:`BasicStatistics` holds the statistics for a single feature
and :class:`DomainBasicStatistics` is a container storing a list of instances of
the above class for all features in the domain.

.. class:: BasicStatistics

    `DomainBasicStatistics` computes on-the fly statistics. 

    .. attribute:: variable
    
        Descriptor for the variable to which the data applies.

    .. attribute:: min, max

        Minimal and maximal variable value encountered.

    .. attribute:: avg, dev

        Average value and standard deviation.

    .. attribute:: n

        Number of instances for which the value was defined
        (and used in the statistics). If instances were weighted,
        ``n`` is the sum of weights of those instances.

    .. attribute:: sum, sum2

        Weighted sum of values and weighted sum of
        squared values of this feature.

    ..
        .. attribute:: holdRecomputation
    
            Holds recomputation of the average and standard deviation.

    .. method:: add(value[, weight=1])
    
        Add a value to the statistics.

        :param value: Value to be added to the statistics
        :type value: float
        :param weight: Weight assigned to the value
        :type weight: float

    ..
        .. method:: recompute()

            Recompute the average and deviation.

    The class works as follows. Values are added by :obj:`add`, for each value
    it checks and, if necessary, adjusts :obj:`min` and :obj:`max`, adds the value to
    :obj:`sum` and its square to :obj:`sum2`. The weight is added to :obj:`n`.


    The statistics does not include the median or any other statistics that can be computed on the fly, without remembering the data. Quantiles can be computed
    by :obj:`ContDistribution`. !!!TODO

    Instances of this class are seldom constructed manually; they are more often
    returned by :obj:`DomainBasicStatistics` described below.

.. class:: DomainBasicStatistics

    ``DomainBasicStatistics`` behaves like a ordinary list, except that its
    elements can also be indexed by feature descriptors or feature names.

    .. method:: __init__(data[, weight=None])

        Compute the statistics for all continuous features in the
        give data, and put `None` to the places corresponding to features of other types.

        :param data: A table of instances
        :type data: Orange.data.Table
        :param weight: The id of the meta-attribute with weights
        :type weight: `int` or none
        
    .. method:: purge()
    
        Remove the ``None``'s corresponding to non-continuous features.
    
    part of `distributions-basic-stat.py`_ (uses monks-1.tab)
    
    .. literalinclude:: code/distributions-basic-stat.py
        :lines: 1-10

    Output::

             feature   min   max   avg
        sepal length 4.300 7.900 5.843
         sepal width 2.000 4.400 3.054
        petal length 1.000 6.900 3.759
         petal width 0.100 2.500 1.199


    part of `distributions-basic-stat`_ (uses iris.tab)
    
    .. literalinclude:: code/distributions-basic-stat.py
        :lines: 11-

    Output::

        5.84333467484 

.. _distributions-basic-stat: code/distributions-basic-stat.py
.. _distributions-basic-stat.py: code/distributions-basic-stat.py


Contingency Matrix
==================

Contingency matrix contains conditional distributions. When initialized, they
will typically contain absolute frequencies, that is, the number of instances
with a particular combination of two variables' values. If they are normalized
by dividing each cell by the row sum, the represent conditional probabilities
of the column variable (here denoted as ``innerVariable``) conditioned by the
row variable (``outerVariable``). 

Contingencies work with both, discrete and continuous variables.

.. _distributions-contingency: code/distributions-contingency.py

part of `distributions-contingency`_ (uses monks-1.tab)

.. literalinclude:: code/distributions-contingency.py
    :lines: 1-8

This code prints out::

    1 <0.000, 108.000>
    2 <72.000, 36.000>
    3 <72.000, 36.000>
    4 <72.000, 36.000> 

Contingencies behave like lists of distributions (in this case, class distributions) indexed by values (of `e`, in this example). Distributions are, in turn indexed
by values (class values, here). The variable `e` from the above example is called
the outer variable, and the class is the inner. This can also be reversed, and it
is also possible to use features for both, outer and inner variable, so the
matrix shows distributions of one variable's values given the value of another.
There is a corresponding hierarchy of classes for handling hierarchies: :obj:`Contingency` is a base class for :obj:`ContingencyVarVar` (both variables
are attribtes) and :obj:`ContingencyClass` (one variable is the class).
The latter is the base class for :obj:`ContingencyVarClass` and :obj:`ContingencyClassVar`.

The most commonly used of the above classes is :obj:`ContingencyVarClass` which
can compute and store conditional probabilities of classes given the feature value.

.. class:: Orange.statistics.distribution.Contingency

    .. attribute:: outerVariable

       Descriptor (:class:`Orange.data.feature.Feature`) of the outer variable.

    .. attribute:: innerVariable

        Descriptor (:class:`Orange.data.feature.Feature`) of the inner variable.
 
    .. attribute:: outerDistribution

        The marginal distribution (:class:`Distribution`) of the outer variable.

    .. attribute:: innerDistribution

        The marginal distribution (:class:`Distribution`) of the inner variable.
        
    .. attribute:: innerDistributionUnknown

        The distribution (:class:`Distribution`) of the inner variable for 
        instances for which the outer variable was undefined.
        This is the difference between the ``innerDistribution``
        and unconditional distribution of inner variable.
      
    .. attribute:: varType

        The type of the outer feature (:obj:`Orange.data.Type`, usually
        :obj:`Orange.data.feature.Discrete` or 
        :obj:`Orange.data.feature.Continuous`). ``varType`` equals ``outerVariable.varType`` and ``outerDistribution.varType``.

    .. method:: __init__(outerVariable, innerVariable)
     
        Construct an instance of ``Contingency`` for the given pair of
        variables.
     
        :param outerVariable: Descriptor of the outer variable
        :type outerVariable: Orange.data.feature.Feature
        :param outerVariable: Descriptor of the inner variable
        :type innerVariable: Orange.data.feature.Feature
        
    .. method:: add(outer_value, inner_value[, weight=1])
    
        Add an element to the contingency matrix by adding
        ``weight`` to the corresponding cell.

        :param outer_value: The value for the outer variable
        :type outer_value: int, float, string or :obj:`Orange.data.Value`
        :param inner_value: The value for the inner variable
        :type inner_value: int, float, string or :obj:`Orange.data.Value`
        :param weight: Instance weight
        :type weight: float

    .. method:: normalize()

        Normalize all distributions (rows) in the contingency to sum to ``1``::
        
            >>> cont.normalize()
            >>> for val, dist in cont.items():
                   print val, dist

        Output: ::

            1 <0.000, 1.000>
            2 <0.667, 0.333>
            3 <0.667, 0.333>
            4 <0.667, 0.333>

        .. note::
       
            This method doesn't change the ``innerDistribution`` or
            ``outerDistribution``.
        
    With respect to indexing, contingency matrix is a cross between dictionary
    and a list. It supports standard dictionary methods ``keys``, ``values`` and
    ``items``.::

        >> print cont.keys()
        ['1', '2', '3', '4']
        >>> print cont.values()
        [<0.000, 108.000>, <72.000, 36.000>, <72.000, 36.000>, <72.000, 36.000>]
        >>> print cont.items()
        [('1', <0.000, 108.000>), ('2', <72.000, 36.000>),
        ('3', <72.000, 36.000>), ('4', <72.000, 36.000>)] 

    Although keys returned by the above functions are strings, contingency
    can be indexed with anything that converts into values
    of the outer variable: strings, numbers or instances of ``Orange.data.Value``.::

        >>> print cont[0]
        <0.000, 108.000>
        >>> print cont["1"]
        <0.000, 108.000>
        >>> print cont[orange.Value(data.domain["e"], "1")] 

    The length of ``Contingency`` equals the number of values of the outer
    variable. However, iterating through contingency
    doesn't return keys, as with dictionaries, but distributions.::

        >>> for i in cont:
            ... print i
        <0.000, 108.000>
        <72.000, 36.000>
        <72.000, 36.000>
        <72.000, 36.000>
        <72.000, 36.000> 


.. class:: Orange.statistics.distribution.ContingencyClass

    ``ContingencyClass`` is an abstract base class for contingency matrices
    that contain the class, either as the inner or the outer
    variable.

    .. attribute:: classVar (read only)
    
        The class attribute descriptor.
        This is always equal either to :obj:`Contingency.innerVariable` or
        ``outerVariable``.

    .. attribute:: variable
    
        The class attribute descriptor.
        This is always equal either to innerVariable or outerVariable

    .. method:: add_attrclass(variable_value, class_value[, weight])

        Adds an element to contingency. The difference between this and
        Contigency.add is that the variable value is always the first
        argument and class value the second, regardless of what is inner and
        outer. 

        :param attribute_value: Variable value
        :type attribute_value: int, float, string or :obj:`Orange.data.Value`
        :param class_value: Class value
        :type class_value: int, float, string or :obj:`Orange.data.Value`
        :param weight: Instance weight
        :type weight: float



.. class:: Orange.statistics.distribution.ContingencyVarClass

    A class derived from :obj:`ContingencyVarClass`, which uses a given feature
    as the :obj:`Contingency.outerVariable` and class as the
    :obj:`Contingency.innerVariable` to provide a form suitable for computation
    of conditional class probabilities given the variable value.
    
    Calling :obj:`ContingencyVarClass.add_attrclass(v, c)` is equivalent
    to calling :obj:`Contingency.add(v, c)`. Similar as :obj:`Contingency`,
    :obj:`ContingencyVarClass` can compute contingency from instances.

    .. method:: __init__(feature, class_attribute)

        Construct an instance of :obj:`ContingencyVarClass` for the given pair of
        variables. Inherited from :obj:`Contingency`.

        :param outerVariable: Descriptor of the outer variable
        :type outerVariable: Orange.data.feature.Feature
        :param outerVariable: Descriptor of the inner variable
        :type innerVariable: Orange.data.feature.Feature
        
    .. method:: __init__(feature, data[, weightId])

        Compute the contingency from the given instances.     

        :param feature: Descriptor of the outer variable
        :type feature: Orange.data.feature.Feature
        :param data: A set of instances
        :type data: Orange.data.Table
        :param weightId: meta attribute with weights of instances
        :type weightId: int

    .. method:: p_class(value)

        Return the probability distribution of classes given the value of the
        variable. Equivalent to `self[value]`, except for normalization.

        :param value: The value of the variable
        :type value: int, float, string or :obj:`Orange.data.Value`

    .. method:: p_class(value, class_value)

        Returns the conditional probability of the class_value given the
        feature value, p(class_value|value) (note the order of arguments!)
        Equivalent to `self[values][class_value]`, except for normalization.
        
        :param value: The value of the variable
        :type value: int, float, string or :obj:`Orange.data.Value`
        :param class_value: The class value
        :type value: int, float, string or :obj:`Orange.data.Value`

    .. _distributions-contingency3.py: code/distributions-contingency3.py

    part of `distributions-contingency3.py`_ (uses monks-1.tab)

    .. literalinclude:: code/distributions-contingency3.py
        :lines: 1-25

    The inner and the outer variable and their relations to the class are
    as follows::

        Inner variable:  y
        Outer variable:  e
    
        Class variable:  y
        Feature:         e

    Distributions are normalized and probabilities are elements from the
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


.. class:: Orange.statistics.distribution.ContingencyClassVar

    :obj:`ContingencyClassVar` is similar to :obj:`ContingencyVarClass` except
    that here the class is outside and the variable is inside. This form of
    contingency matrix is suitable for computing conditional probabilities of
    variable given the class. All methods get the two arguments in the same
    order as in :obj:`ContingencyVarClass`.

    .. method:: __init__(feature, class_attribute)

        Construct an instance of :obj:`ContingencyVarClass` for the given pair of
        variables. Inherited from :obj:`Contingency`, except for the reversed
        order.

        :param outerVariable: Descriptor of the outer variable
        :type outerVariable: Orange.data.feature.Feature
        :param outerVariable: Descriptor of the inner variable
        :type innerVariable: Orange.data.feature.Feature
        
    .. method:: __init__(feature, instances[, weightId])

        Compute the contingency from the given instances.     

        :param feature: Descriptor of the outer variable
        :type feature: Orange.data.feature.Feature
        :param data: A set of instances
        :type data: Orange.data.Table
        :param weightId: meta attribute with weights of instances
        :type weightId: int

    .. method:: p_attr(class_value)

        Return the probability distribution of variable given the class.
        Equivalent to `self[class_value]`, except for normalization.

        :param class_value: The value of the variable
        :type class_value: int, float, string or :obj:`Orange.data.Value`

    .. method:: p_attr(value, class_value)

        Returns the conditional probability of the value given the
        class, p(value|class_value).
        Equivalent to `self[class][value]`, except for normalization.

        :param value: The value of the variable
        :type value: int, float, string or :obj:`Orange.data.Value`
        :param class_value: The class value
        :type value: int, float, string or :obj:`Orange.data.Value`

    .. _distributions-contingency4.py: code/distributions-contingency4.py
    
    part of the output from `distributions-contingency4.py`_ (uses monk1.tab)
    
    The inner and the outer variable and their relations to the class
    and the features are exactly the reverse from :obj:`ContingencyClassVar`::
    
        Inner variable:  e
        Outer variable:  y
    
        Class variable:  y
        Feature:         e
    
    Distributions given the class can be printed out by calling :meth:`p_attr`.
    
    part of `distributions-contingency4.py`_ (uses monks-1.tab)
    
    .. literalinclude:: code/distributions-contingency4.py
        :lines: 31-
    
    will print::
        p(.|0) = <0.000, 0.333, 0.333, 0.333>
        p(.|1) = <0.500, 0.167, 0.167, 0.167>
    
    If the class value is '0', the attribute e cannot be '1' (the first value),
    while distribution across other values is uniform.
    If the class value is '1', e is '1' in exactly half of examples, and
    distribution of other values is again uniform.

.. class:: Orange.statistics.distribution.ContingencyVarVar

    Contingency matrices in which none of the variables is the class. 
    The class is similar to the parent class :obj:`Contingency`, except for
    an additional constructor and method for getting conditional probabilities.

    .. method:: ContingencyVarVar(outer_variable, inner_variable)

        Inherited from :obj:`Contingency`.

    .. method:: __init__(outer_variable, inner_variable, data[, weightId])

        Compute the contingency from the given instances.

        :param outer_variable: Descriptor of the outer variable
        :type outer_variable: Orange.data.feature.Feature
        :param inner_variable: Descriptor of the inner variable
        :type inner_variable: Orange.data.feature.Feature
        :param data: A set of instances
        :type data: Orange.data.Table
        :param weightId: meta attribute with weights of instances
        :type weightId: int

    .. method:: p_attr(outer_value)

        Return the probability distribution of the inner
        variable given the outer variable value.

        :param outer_value: The value of the outer variable
        :type outer_value: int, float, string or :obj:`Orange.data.Value`
 
    .. method:: p_attr(outer_value, inner_value)

        Return the conditional probability of the inner_value
        given the outer_value.

        :param outer_value: The value of the outer variable
        :type outer_value: int, float, string or :obj:`Orange.data.Value`
        :param inner_value: The value of the inner variable
        :type inner_value: int, float, string or :obj:`Orange.data.Value`

    The following example investigates which material is used for
    bridges of different lengths.
    
    .. _distributions-contingency5: code/distributions-contingency5.py
    
    part of `distributions-contingency5`_ (uses bridges.tab)
    
    .. literalinclude:: code/distributions-contingency5.py
        :lines: 1-19

    Short bridges are mostly wooden or iron,
    and the longer (and the most of middle sized) are made from steel::
    
        SHORT:
           WOOD (56%)
           IRON (44%)
    
        MEDIUM:
           WOOD (9%)
           IRON (11%)
           STEEL (79%)
    
        LONG:
           STEEL (100%)
    
    As all other contingency matrices, this one can also be computed "manually".
    
    .. literalinclude:: code/distributions-contingency5.py
        :lines: 20-


Contingency matrices for continuous variables
---------------------------------------------

The described classes can also be used for continuous values.

If the outer feature is continuous, the index must be one
of the values that do exist in the contingency matrix. Using other values
triggers an exception::

    .. _distributions-contingency6: code/distributions-contingency6.py
    
    part of `distributions-contingency6`_ (uses monks-1.tab)
    
    .. literalinclude:: code/distributions-contingency6.py
        :lines: 1-5,18,19

Since even rounding is a problem, the keys should generally come from the
contingencies `keys`.

Contingencies with discrete outer variable continuous inner variables are
more useful, since methods :obj:`ContingencyClassVar.p_class` and 
:obj:`ContingencyVarClass.p_attr` use the primitive density estimation
provided by :obj:`Distribution`.

For example, :obj:`ContingencyClassVar` on the iris dataset,
you can enquire about the probability of the sepal length 5.5::

    .. _distributions-contingency7: code/distributions-contingency7.py
    
    part of `distributions-contingency7`_ (uses iris.tab)
    
    .. literalinclude:: code/distributions-contingency7.py

The script outputs::

    Estimated frequencies for e=5.5
      f(5.5|Iris-setosa) = 2.000
      f(5.5|Iris-versicolor) = 5.000
      f(5.5|Iris-virginica) = 1.000


Contingency matrices for the entire domain
------------------------------------------

DomainContingency is basically a list of contingencies,
either :obj:`ContingencyVarClass` or :obj:`ContingencyClassVar`.

.. class:: DomainContingency

    .. method:: __init__(data[, weightId=0, classOuter=0|1])

        Compute a list of contingencies.

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

    .. method:: normalize

        Call normalize for all contingencies.

    The following script prints the contingencies for features
    "a", "b" and "e" for the dataset Monk 1.
    
    .. _distributions-contingency8: code/distributions-contingency8.py
    
    part of `distributions-contingency8`_ (uses monks-1.tab)
    
    .. literalinclude:: code/distributions-contingency8.py
        :lines: 1-11

    Contingencies are of type :obj:`ContingencyVarClass` give
    the conditional distributions of classes, given the value of the variable.
    
    .. _distributions-contingency8: code/distributions-contingency8.py
    
    part of `distributions-contingency8`_ (uses monks-1.tab)
    
    .. literalinclude:: code/distributions-contingency8.py
        :lines: 13- 

"""



from Orange.core import \
     DomainContingency, \
     DomainDistributions, \
     DistributionList, \
     ComputeDomainContingency, \
     Contingency

from Orange.core import BasicAttrStat as BasicStatistics
from Orange.core import DomainBasicAttrStat as DomainBasicStatistics
from Orange.core import ContingencyAttrAttr as ContingencyVarVar
from Orange.core import ContingencyAttrAttr as ContingencyClass
from Orange.core import ContingencyAttrAttr as ContingencyVarClass
from Orange.core import ContingencyAttrAttr as ContingencyClassVar
