"""

Orange has several classes for computing and storing basic statistics about
features, distributions and contingencies.
 
========================================
Basic Statistics for Continuous Features
========================================

The are two simple classes for computing basic statistics
for continuous features, such as their minimal and maximal value
or average: :class:`BasicAttrStat` holds the statistics for a single feature
and :class:`DomainBasicAttrStat` is a container storing a list of instances of
the above class for all features in the domain.

.. class:: BasicAttrStat

    `BasicAttrStat` computes on-the fly statistics. 

    .. attribute:: variable
    
        The descriptor for the feature to which the data applies.

    .. attribute:: min, max

        Minimal and maximal feature value encountered
        in the data table.

    .. attribute:: avg, dev

        Average value and standard deviation.

    .. attribute:: n

        Number of instances for which the value was defined
        (and used in the statistics). If instances were weighted,
        `n` is the sum of weights of those instances.

    .. attribute:: sum, sum2

        Weighted sum of values and weighted sum of
        squared values of this feature.

    ..
        .. attribute:: holdRecomputation
    
            Holds recomputation of the average and standard deviation.

    .. method:: add(value[, weight=1.0])

        Adds a value to the statistics. Both arguments should be numbers.

    ..
        .. method:: recompute()

            Recomputes the average and deviation.

    The class works as follows. Values are added by :obj:`add`, for each value
    it checks and, if necessary, adjusts :obj:`min` and :obj:`max`, adds the value to
    :obj:`sum` and its square to :obj:`sum2`. The weight is added to :obj:`n`.

    The statistics does not include the median or any other statistics that can be computed on the fly, without remembering the data. Quantiles can be computed
    by :obj:`ContDistribution`. !!!TODO

    Instances of this class are seldom constructed manually; they are more often
    returned by :obj:`DomainBasicAttrStat` described below.

.. class:: DomainBasicAttrStat

    :param data: A table of instances
    :type data: Orange.data.Table
    :param weight: The id of the meta-attribute with weights
    :type weight: `int` or none
    
    Constructor computes the statistics for all continuous features in the
    give data, and puts `None` to the places corresponding to other types of
    features.

    .. method:: purge()
    
        Removes the `None`'s corresponding to non-continuous features.
    
    `DomainBasicAttrStat` behaves like a ordinary list, except that its
    elements can also be indexed by feature descriptors or feature names.

    .. _distributions-basic-stat: code/distributions-basic-stat.py
    part of `distributions-basic-stat`_ (uses monks-1.tab)

    .. literalinclude:: code/distributions-basic-stat.py
        :lines: 1-10

    This code prints out::

                 feature   min   max   avg
            sepal length 4.300 7.900 5.843
             sepal width 2.000 4.400 3.054
            petal length 1.000 6.900 3.759
             petal width 0.100 2.500 1.199


    .. _distributions-basic-stat: code/distributions-basic-stat.py
    part of `distributions-basic-stat`_ (uses iris.tab)

    .. literalinclude:: code/distributions-basic-stat.py
        :lines: 11-

    This code prints out::

        5.84333467484 



==================
Contingency Matrix
==================

Contingency matrix contains conditional distributions. They can work for both,
discrete and continuous variables; although examples on this page will mostly
use discrete ones, similar code could be run for continuous variables.

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
There is a corresponding hierarchy of classes for handling hierarchies: :obj:`Contingency` is a base class for :obj:`ContingencyAttrAttr` (and
:obj:`ContingencyClass`; the latter is 



There is a hierarchy of classes with contingencies::

    Contingency::
        ContingencyClass
        ContingencyClassAttr
        ContingencyAttrClass
    ContingencyAttrAttr

The base object is Contingency. Derived from it is ContingencyClass
in which one of the feaure is class; ContingencyClass
is a base for two classes, ContingencyAttrClass and ContingencyClassAttr,
the former having class as the inner and the latter as the outer feature.
Class ContingencyAttrAttr is derived directly from Contingency and represents
contingency matrices in which none of the feature is the class.

The most common used of the above classes is ContingencyAttrClass which
resembles conditional probabilities of classes given the feature value.

Here's what all contingency matrices share in common.

.. class:: Orange.probability.distribution.Contingency

    The base class is, once for a change, not abstract. Its constructor expects
    two feature descriptors, the first one for the outer and the second for
    the inner feature. It initializes empty distributions and it's up to you
    to fill them. This is, for instance, how to manually reproduce results of
    the script at the top of the page.

    .. attribute:: outerVariable

       The outer feature descriptor. In the above case, it is e. 

    .. attribute:: innerVariable

        The inner feature descriptor. In the above case, it is the class

    .. attribute:: outerDistribution

        The distribution of the outer featue's values - sums of rows.
        In the above case, distribution of e is
        <108.000, 108.000, 108.000, 108.000>

    .. attribute:: innerDistribution

        The distribution of the inner feature.
        In the above case, it is the class distribution
        which is <216.000, 216.000<. 

    .. attribute:: innerDistributionUnknown

        The distribution of the inner feature for the
        instances where the outer feature was unknown.
        This is the difference between the innerDistribution
        and the sum of all distributions in the matrix.
      
    .. attribute:: varType

        The varType for the outer feature (discrete, continuous...);
        varType equals outerVariable.varType and outerDistribution.varType.

Contingency matrix is a cross between dictionary and a list.
It supports standard dictionary methods keys, values and items.::

    >> print cont.keys()
    ['1', '2', '3', '4']
    >>> print cont.values()
    [<0.000, 108.000>, <72.000, 36.000>, <72.000, 36.000>, <72.000, 36.000>]
    >>> print cont.items()
    [('1', <0.000, 108.000>), ('2', <72.000, 36.000>),
    ('3', <72.000, 36.000>), ('4', <72.000, 36.000>)] 

Although keys returned by the above functions are strings,
you can index the contingency with anything that converts into values
of the outer feature - strings, numbers or instances of Value.::

    >>> print cont[0]
    <0.000, 108.000>
    >>> print cont["1"]
    <0.000, 108.000>
    >>> print cont[orange.Value(data.domain["e"], "1")] 

Naturally, the length of Contingency equals the number of values of the outer
feature. The only weird thing is that iterating through contingency
(by using a for loop, for instance) doesn't return keys, as with dictionaries,
but dictionary values.::

    >>> for i in cont:
        ... print i
    <0.000, 108.000>
    <72.000, 36.000>
    <72.000, 36.000>
    <72.000, 36.000>
    <72.000, 36.000> 

If cont behaved like a normal dictionary, the above script would print out strings from '0' to '3'.


Other methods

.. class:: Orange.probability.distributions.Contingency

    .. method:: add(outer_value, inner_value[, weight])

       Adds an element to the contingency matrix.

    .. method:: normalize()

Normalizes all distributions (rows) in the contingency to sum to 1.
It doesn't change the innerDistribution or outerDistribution.::

    >>> cont.normalize()
    >>> for val, dist in cont.items():
           print val, dist

This outputs: ::

    1 <0.000, 1.000>
    2 <0.667, 0.333>
    3 <0.667, 0.333>
    4 <0.667, 0.333>

.. _distributions-contingency2: code/distributions-contingency2.py
part of `distributions-contingency2`_ (uses monks-1.tab)

.. literalinclude:: code/distributions-contingency2.py

The "reproduction" is not perfect. We didn't care about unknown values
and haven't computed innerDistribution and outerDistribution.
The better way to do it is by using the method add, so that the loop becomes: ::

    for ins in table:
        cont.add(ins["e"], ins.getclass()) 

It's not only simpler, but also correctly handles unknown values
and updates innerDistribution and outerDistribution. 

.. class:: Orange.probability.distribution.ContingencyClass

    ContingencyClass is an abstract base class for contingency matrices
    that contain the class, either as the inner or the outer
    feature. If offers a function for making filing the contingency clearer.

    After reading through the rest of this page you might ask yourself
    why do we need to separate the classes ContingencyAttrClass,
    ContingencyClassAttr and ContingencyAttrAttr,
    given that the underlying matrix is the same. This is to avoid confusion
    about what is in the inner and the outer variable.
    Contingency matrices are most often used to compute probabilities of conditional
    classes or features. By separating the classes and giving them specialized
    methods for computing the probabilities that are most suitable to compute
    from a particular class, the user (ie, you or the method that gets passed
    the matrix) is relieved from checking what kind of matrix it got, that is,
    where is the class and where's the feature.



    .. attribute:: classVar (read only)
    
        The class attribute descriptor.
        This is always equal either to innerVariable or outerVariable

    .. attribute:: variable
    
        The class attribute descriptor.
        This is always equal either to innerVariable or outerVariable

    .. method:: add_attrclass(attribute_value, class_value[, weight])

        Adds an element to contingency. The difference between this and
        Contigency.add is that the feature value is always the first
        argument and class value the second, regardless whether the feature
        is actually the outer variable or the inner. 



.. class:: Orange.probability.distribution.ContingencyClass

    ContingencyAttrClass is derived from ContingencyClass.
    Here, feature is the outer variable (hence variable=outerVariable)
    and class is the inner (classVar=innerVariable), so this form of
    contingency matrix is suitable for computing the conditional probabilities
    of classes given a value of a feature.

    Calling add_attrclass(v, c) is here equivalent to calling add(v, c).
    In addition to this, the class supports computation of contingency from instances,
    as you have already seen in the example at the top of this page.


    .. method:: ContingencyAttrClass(feature, class_attribute)

        The inherited constructor, which does exactly the same
        as Contingency's constructor.

    .. method:: ContingencyAttrClass(feaure, class_attribute)

        The inherited constructor, which does exactly the same
        as Contingency's constructor.

    .. method::  ContingencyAttrClass(feature, instances[, weightID])

        Constructor that constructs the contingency and computes the
        data from the given instances. If these are weighted, the meta
        attribute with instance weights can be specified.     

    .. method:: p_class(attribute_value)

        Returns the distribution of classes given the attribute_value.
        If the matrix is normalized, this is equivalent to returning
        self[attribute_value].
        Result is returned as a normalized Distribution.

    .. method:: p_class(attribute_value, class_value)

        Returns the conditional probability of class_value given the
        attribute_value. If the matrix is normalized, this is equivalent
        to returning self[attribute_value][class_value].

Don't confuse the order of arguments: feature value is the first,
class value is the second, just as in add_attrclass. Although in this
instance counterintuitive (since the returned value represents the conditional
probability P(class_value|attribute_value), this order is uniform for all
(applicable) methods of classes derived from ContingencyClass.

You have seen this form of matrix used already at the top of the page.
We shall only explore the new stuff we've learned about it.


.. _distributions-contingency3.py: code/distributions-contingency3.py
part of `distributions-contingency3.py`_ (uses monks-1.tab)

.. literalinclude:: code/distributions-contingency3.py
    :lines: 1-25

The inner and the outer variable and their relations to the class
and the features are as expected.::

    Inner variable:  y
    Outer variable:  e

    Class variable:  y
    Feature:         e

Distributions are normalized and probabilities are elements from the
normalized distributions. Knowing that the target concept is
y := (e=1) or (a=b), distributions are as expected: when e equals 1, class 1
has a 100% probability, while for the rest, probability is one third, which
agrees with a probability that two three-valued independent features
have the same value.::

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


Manual computation using add_attrclass is similar
(to be precise: exactly the same) as computation using add.

.. _distributions-contingency3.py: code/distributions-contingency3.py

part of `distributions-contingency3.py`_ (uses monks-1.tab)

.. literalinclude:: code/distributions-contingency3.py
    :lines: 27-


.. class:: Orange.probability.distribution.ContingencyClassAttr

    ContingencyClassAttr is similar to ContingencyAttrClass except that here
    the class is the outer and the feature the inner variable.
    As a consequence, this form of contingency matrix is suitable
    for computing conditional probabilities of feature values given class.
    Constructor and add_attrclass nevertheless get the arguments
    in the same order as for ContingencyAttrClass, that is,
    feaure first, class second.


    ..method:: ContingencyClassAttr(attribute, class_attribute)

        The inherited constructor is exactly the same as Contingency's
        constructor, except that the argument order is reversed
        (in Contingency, the outer attribute is given first,
        while here the first argument, attribute, is the inner attribute).
    
    .. method:: ContingencyAttrClass(attribute, examples[, weightID])

        Constructs the contingency and computes the data from the given
        examples. If these are weighted, the meta attribute with example
        weights can be specified. 
    
    .. method:: p_attr(class_value)

        Returns the distribution of attribute values given the class_value.
        If the matrix is normalized, this is equivalent to returning
        self[class_value]. Result is returned as a normalized Distribution.

    .. method:: p_attr(attribute_value, class_value)
    
        Returns the conditional probability of attribute_value given the
        class_value. If the matrix is normalized, this is equivalent to
        returning self[class_value][attribute_value].
  
As you can see, the class is rather similar to ContingencyAttrClass,
except that it has p_attr instead of p_class.
If you, for instance, take the above script and replace the class name,
the first bunch of prints print out


.. _distributions-contingency4.py: code/distributions-contingency4.py

part of the output from `distributions-contingency4.py`_ (uses monk1.tab)

The inner and the outer variable and their relations to the class
and the features are as expected.::

    Inner variable:  e
    Outer variable:  y

    Class variable:  y
    Feature:         e


This is exactly the reverse of the printout from ContingencyAttrClass.
To print out the distributions, the only difference now is that you need
to iterate through values of the class attribute and call p_attr. For instance,

part of `distributions-contingency4.py`_ (uses monks-1.tab)

.. literalinclude:: code/distributions-contingency4.py
    :lines: 31-

will print::
    p(.|0) = <0.000, 0.333, 0.333, 0.333>
    p(.|1) = <0.500, 0.167, 0.167, 0.167>


If the class value is '0', than attribute e cannot be '1' (the first value),
but can be anything else, with equal probabilities of 0.333.
If the class value is '1', e is '1' in exactly half of examples
(work-out why this is so); in the remaining cases,
e is again distributed uniformly.
    

.. class:: Orange.probability.distribution.ContingencyAttrAttr

    ContingencyAttrAttr stores contingency matrices in which none
    of the features is the class. This is rather similar to Contingency,
    except that it has an additional constructor and method for getting
    the conditional probabilities.

    .. method:: ContingencyAttrAttr(outer_variable, inner_variable)

        This constructor is exactly the same as that of Contingency.

    .. method:: ContingencyAttrAttr(outer_variable, inner_variable, 
    instances[, weightID])

        Computes the contingency from the given instances.

    .. method:: p_attr(outer_value)

        Returns the probability distribution of the inner
        variable given the outer variable.

    .. method:: p_attr(outer_value, inner_value)

        Returns the conditional probability of the inner_value
        given the outer_value.


In the following example, we shall use the ContingencyAttrAttr
on dataset "bridges" to determine which material is used for
bridges of different lengths.


.. _distributions-contingency5: code/distributions-contingency5.py
part of `distributions-contingency5`_ (uses bridges.tab)

.. literalinclude:: code/distributions-contingency5.py
    :lines: 1-19

The output tells us that short bridges are mostly wooden or iron,
and the longer (and the most of middle sized) are made from steel.::

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


=================
Contingencies with Continuous Values
=================

What happens if one or both features are continuous?
As first, contingencies can be built for such features as well.
Just imagine a contingency as a dictionary with features values
as keys and objects of type Distribution as values.

If the outer feature is continuous, you can use either its values
or ordinary floating point number for indexing. The index must be one
of the values that do exist in the contingency matrix.

The following script will query for a distribution in between
the first two keys, which triggers an exception.


.. _distributions-contingency6: code/distributions-contingency6.py
part of `distributions-contingency6`_ (uses monks-1.tab)

.. literalinclude:: code/distributions-contingency6.py
    :lines: 1-5,18,19

If you still find such contingencies useful, you need to take care
about what you pass for indices. Always use the values from keys()
directly, instead of manually entering the keys' values you see printed.
If, for instance, you print out the first key, see it's 4.500 and then
request cont[4.500] this can give an index error due to rounding.

Contingencies with continuous inner features are more useful.
As first, indexing by discrete values is easier than with continuous.
Secondly, class Distribution covers both, discrete and continuous
distributions, so even the methods p_class and p_attr will work,
except they won't return is not the probability but the density
(interpolated, if necessary). See the page about Distribution
for more information.

For instance, if you build a ContingencyClassAttr on the iris dataset,
you can enquire about the probability of the sepal length 5.5.

.. _distributions-contingency7: code/distributions-contingency7.py
part of `distributions-contingency7`_ (uses iris.tab)

.. literalinclude:: code/distributions-contingency7.py

    
The script's output is::

    Estimated frequencies for e=5.5
      f(5.5|Iris-setosa) = 2.000
      f(5.5|Iris-versicolor) = 5.000
      f(5.5|Iris-virginica) = 1.000


=================
Computing Contingencies for All Features
=================

Computing contingency matrices requires iteration through instances.
We often need to compute ContingencyAttrClass or ContingencyClassAttr
for all features in the dataset and it is obvious that this will be faster
if we do it for all features at once. That's taken care of
by class DomainContingency.

DomainContingency is basically a list of contingencies,
either of type ContingencyAttrClass or ContingencyClassAttr, with two
additional fields and a constructor that computes the contingencies.

.. class:: DomainContingency(instances[, weightID][, classIsOuter=0|1])

    Constructor needs to be given a list of instances.
    It constructs a list of contingencies; if classIsOuter is 0 (default),
    these will be ContingencyAttrClass, if 1, ContingencyClassAttr are used.
    It then iterates through instances and computes the contingencies.

    .. attribute:: classIsOuter (read only)

        Tells whether the class is the outer or the inner featue.
        Effectively, this tells whether the elements of the list
        are ContingencyAttrClass or ContingencyClassAttr.

    .. attribute:: classes

        Contains the distribution of class values on the entire dataset.

    .. method:: normalize

        Calls normalize for each contingency.

The following script will print the contingencies for features
"a", "b" and "e" for the dataset Monk 1.

.. _distributions-contingency8: code/distributions-contingency8.py
part of `distributions-contingency8`_ (uses monks-1.tab)

.. literalinclude:: code/distributions-contingency8.py
    :lines: 1-11


The contingencies in the DomainContingency dc are of type ContingencyAttrClass
and tell us conditional distributions of classes, given the value of the feature.
To compute the distribution of feature values given the class,
one needs to get a list of ContingencyClassAttr.

Note that classIsOuter cannot be given as positional argument,
but needs to be passed by keyword.

.. _distributions-contingency8: code/distributions-contingency8.py
part of `distributions-contingency8`_ (uses monks-1.tab)

.. literalinclude:: code/distributions-contingency8.py
    :lines: 13- 

"""



from orange import \
     DomainContingency, \
     DomainDistributions, \
     DistributionList, \
     ComputeDomainContingency, \
     Contingency

from orange import BasicAttrStat as BasicStatistics
from orange import DomainBasicAttrStat as DomainBasicStatistics
from orange import ContingencyAttrAttr as ContingencyVarVar
from orange import ContingencyAttrAttr as ContingencyClass
from orange import ContingencyAttrAttr as ContingencyVarClass
from orange import ContingencyAttrAttr as ContingencyClassVar
