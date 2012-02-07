"""
.. index:: Basic Statistics for Continuous Features

========================================
Basic Statistics for Continuous Features
========================================

The are two simple classes for computing basic statistics
for continuous features, such as their minimal and maximal value
or average: :class:`Orange.statistics.basic.Variable` holds the statistics for a single variable
and :class:`Orange.statistics.basic.Domain` behaves like a list of instances of
the above class for all variables in the domain.

.. class:: Variable

    Computes and stores minimal, maximal, average and
    standard deviation of a variable. It does not include the median or any
    other statistics that can be computed on the fly, without remembering the
    data; such statistics can be obtained classes from module :obj:`Orange.statistics.distribution`.

    Instances of this class are seldom constructed manually; they are more often
    returned by :obj:`Domain` described below.

    .. attribute:: variable
    
        The variable to which the data applies.

    .. attribute:: min

        Minimal value encountered

    .. attribute:: max

        Maximal value encountered

    .. attribute:: avg

        Average value

    .. attribute:: dev

        Standard deviation

    .. attribute:: n

        Number of instances for which the value was defined.
        If instances were weighted, :obj:`n` holds the sum of weights
        
    .. attribute:: sum

        Weighted sum of values

    .. attribute:: sum2

        Weighted sum of squared values

    ..
        .. attribute:: holdRecomputation
    
            Holds recomputation of the average and standard deviation.

    .. method:: add(value[, weight=1])
    
        Add a value to the statistics: adjust :obj:`min` and :obj:`max` if
        necessary, increase :obj:`n` and recompute :obj:`sum`, :obj:`sum2`,
        :obj:`avg` and :obj:`dev`.

        :param value: Value to be added to the statistics
        :type value: float
        :param weight: Weight assigned to the value
        :type weight: float

    ..
        .. method:: recompute()

            Recompute the average and deviation.

.. class:: Domain

    ``statistics.basic.Domain`` behaves like an ordinary list, except that its
    elements can also be indexed by variable names or descriptors.

    .. method:: __init__(data[, weight=None])

        Compute the statistics for all continuous variables in the data, and put
        :obj:`None` to the places corresponding to variables of other types.

        :param data: A table of instances
        :type data: Orange.data.Table
        :param weight: The id of the meta-attribute with weights
        :type weight: `int` or none
        
    .. method:: purge()
    
        Remove the :obj:`None`'s corresponding to non-continuous features; this
        truncates the list, so the indices do not respond to indices of
        variables in the domain.
    
    part of :download:`distributions-basic-stat.py <code/distributions-basic-stat.py>`
    
    .. literalinclude:: code/distributions-basic-stat.py
        :lines: 1-10

    Output::

             feature   min   max   avg
        sepal length 4.300 7.900 5.843
         sepal width 2.000 4.400 3.054
        petal length 1.000 6.900 3.759
         petal width 0.100 2.500 1.199


    part of :download:`distributions-basic-stat.py <code/distributions-basic-stat.py>`
    
    .. literalinclude:: code/distributions-basic-stat.py
        :lines: 11-

    Output::

        5.84333467484 

"""

from Orange.core import BasicAttrStat as Variable
from Orange.core import DomainBasicAttrStat as Domain
