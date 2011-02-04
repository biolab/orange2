"""

=================
Basic Statistics for Continuous Features
=================


.. class:: Orange.probability.distribution.BasicAttrStat

    Orange contains two simple classes for computing basic statistics
    for continuous features, such as their minimal and maximal value
    or average: BasicAttrStat holds the statistics for a single feature
    and DomainBasicAttrStat holds the statistics for all features in the domain.


    .. attribute:: variable
    
        The descriptor for the feature to which the data applies.

    .. attribute:: min, max

        Minimal and maximal feature value that was encountered
        in the data table.

    .. attribute:: avg, dev

        Average value and standard deviation.

    .. attribute:: n

        Number of instances for which the value was defined
        (and used in the statistics). If instances were weighted,
        n is the sum of weights of those instances.

    .. attribute:: sum, sum2

        Weighted sum of values and weighted sum of
        squared values of this feature.

    .. attribute:: holdRecomputation

        Holds recomputation of the average and standard deviation.

    .. method:: add(value[, weight])

        Adds a value to the statistics. Both arguments should be numbers;
        weight is optional, default is 1.0.
        
    .. method:: recompute()

        Recomputes the average and deviation.


You most probably won't construct the class yourself, but instead call
DomainBasicAttrStat to compute statistics for all continuous
features in the dataset.

Nevertheless, here's how the class works. Values are fed into add;
this is usually done by DomainBasicAttrStat, but you can traverse the
instances and feed the values in Python, if you want to. For each value
it checks and, if necessary, adjusts min and max, adds the value to
sum and its square to sum2. The weight is added to n. If holdRecomputation
is false, it also computes the average and the deviation.
If true, this gets postponed until recompute is called.
It makes sense to postpone recomputation when using the class from C++,
while when using it from Python, the recomputation will take much much
less time than the Python interpreter, so you can leave it on.

You can see that the statistics does not include the median or,
more generally, any quantiles. That's because it only collects
statistics that can be computed on the fly, without remembering the data.
If you need quantiles, you will need to construct a ContDistribution.


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

Methods

.. class:: Orange.probability.distribution.DomainBasicAttrStat

    DomainBasicAttrStat behaves as a list of BasicAttrStat except
    for a few details.

    Constructor expects an instance generator;
    if instances are weighted, the second (otherwise optional)
    arguments should give the id of the meta-attribute with weights.

    DomainBasicAttrStat behaves like a ordinary list, except that its
    elements can also be indexed by feature descriptors or feaure names.    

    .. method:: purge()
  
    Noticed the "if a" in the script? It's needed because of discrete
    features for which this statistics cannot be measured and are thus
    represented by a None. Method purge gets rid of them by removing
    the None's from the list.


.. _distributions-basic-stat: code/distributions-basic-stat.py
part of `distributions-basic-stat`_ (uses iris.tab)

.. literalinclude:: code/distributions-basic-stat.py
    :lines: 11-

This code prints out::

    5.84333467484 



=================
Contingency Matrix
=================

Contingency matrix contains conditional distributions. They can work for both,
discrete and continuous features; although the examples on this page will be
mostly limited to discrete features, the analogous could be done with
continuous values.

.. _distributions-contingency: code/distributions-contingency.py
part of `distributions-contingency`_ (uses monks-1.tab)

.. literalinclude:: code/distributions-contingency.py
    :lines: 1-8


This code prints out::

    1 <0.000, 108.000>
    2 <72.000, 36.000>
    3 <72.000, 36.000>
    4 <72.000, 36.000> 


As this simple example shows, contingency is similar to a dictionary
(or a list, it is a bit ambiguous), where feature values serve as
keys and class distributions are the dictionary values.
The feature e is here called the outer feature, and the class
is the inner. That's not the only possible configuration of contingency
matrix; class can also be outside or there can be no class at all and the
matrix shows distributions of one feature values given the value of another.

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


=================
General Contingency Matrix
=================

Here's what all contingency matrices share in common.

Attributes

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

    .. method:: add(outer_value, inner_value[, weight])

       Adds an element to the contingency matrix.

    .. method:: normalize()

Normalizes all distributions (rows) in the contingency to sum to 1.
It doesn't change the innerDistribution or outerDistribution.::

    >>> cont.normalize()
    >>> for val, dist in cont.items():
           print val, dist

The ouput: ::

    1 <0.000, 1.000>
    2 <0.667, 0.333>
    3 <0.667, 0.333>
    4 <0.667, 0.333>


=================
Contingency
=================

The base class is, once for a change, not abstract. Its constructor expects
two feature descriptors, the first one for the outer and the second for
the inner feature. It initializes empty distributions and it's up to you
to fill them. This is, for instance, how to manually reproduce results of
the script at the top of the page.

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


=================
ContingencyClass
=================

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


.. class:: Orange.probability.distribution.ContingencyClass

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


=================
ContingencyAttrClass
=================

ContingencyAttrClass is derived from ContingencyClass.
Here, feature is the outer variable (hence variable=outerVariable)
and class is the inner (classVar=innerVariable), so this form of
contingency matrix is suitable for computing the conditional probabilities
of classes given a value of a feature.

Calling add_attrclass(v, c) is here equivalent to calling add(v, c).
In addition to this, the class supports computation of contingency from instances,
as you have already seen in the example at the top of this page.

.. class:: Orange.probability.distribution.ContingencyClass

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


=================
ContingencyClassAttr
=================

ContingencyClassAttr is similar to ContingencyAttrClass except that here
the class is the outer and the feature the inner variable.
As a consequence, this form of contingency matrix is suitable
for computing conditional probabilities of feature values given class.
Constructor and add_attrclass nevertheless get the arguments
in the same order as for ContingencyAttrClass, that is,
feaure first, class second.


.. class:: Orange.probability.distribution.ContingencyClassAttr

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
    

=================
ContingencyAttrAttr
=================

ContingencyAttrAttr stores contingency matrices in which none
of the features is the class. This is rather similar to Contingency,
except that it has an additional constructor and method for getting
the conditional probabilities.

.. class:: Orange.probability.distribution.ContingencyAttrAttr

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


.. attribute:: classIsOuter (read only)

    Tells whether the class is the outer or the inner featue.
    Effectively, this tells whether the elements of the list
    are ContingencyAttrClass or ContingencyClassAttr.

.. attribute:: classes

    Contains the distribution of class values on the entire dataset.

..method:: DomainContingency(instances[, weightID][, classIsOuter=0|1])

    Constructor needs to be given a list of instances.
    It constructs a list of contingencies; if classIsOuter is 0 (default),
    these will be ContingencyAttrClass, if 1, ContingencyClassAttr are used.
    It then iterates through instances and computes the contingencies.

..method:: normalize

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

part of **distributions-contingency8.py** (uses monk1.tab)::

Note that classIsOuter cannot be given as positional argument,
but needs to be passed by keyword.

.. _distributions-contingency8: code/distributions-contingency8.py
part of `distributions-contingency8`_ (uses monks-1.tab)

.. literalinclude:: code/distributions-contingency8.py
    :lines: 13- 

"""


from orange import \
     BasicAttrStat, \
     DomainBasicAttrStat, \
     DomainContingency, \
     DomainDistributions, \
     DistributionList, \
     ComputeDomainContingency, \
     ConditionalProbabilityEstimator, \
     ConditionalProbabilityEstimator_ByRows, \
     ConditionalProbabilityEstimator_FromDistribution, \
     ConditionalProbabilityEstimatorConstructor, \
     ConditionalProbabilityEstimatorConstructor_ByRows, \
     ConditionalProbabilityEstimatorConstructor_loess, \
     ConditionalProbabilityEstimatorList, \
     Contingency, \
     ContingencyAttrAttr, \
     ContingencyClass, \
     ContingencyAttrClass, \
     ContingencyClassAttr
