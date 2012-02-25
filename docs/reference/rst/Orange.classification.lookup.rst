.. py:currentmodule:: Orange.classification.lookup

.. index:: classification; lookup

*******************************
Lookup classifiers (``lookup``)
*******************************

Lookup classifiers predict classes by looking into stored lists of
cases. There are two kinds of such classifiers in Orange. The simpler
and faster :obj:`ClassifierByLookupTable` uses up to three discrete
features and has a stored mapping from values of those features to the
class value. The more complex classifiers store an
:obj:`Orange.data.Table` and predict the class by matching the
instance to instances in the table.

.. index::
   single: feature construction; lookup classifiers

A natural habitat for these classifiers is feature construction: they
usually reside in :obj:`~Orange.feature.Descriptor.get_value_from`
fields of constructed features to facilitate their automatic
computation. For instance, the following script shows how to translate
the ``monks-1.tab`` data set features into a more useful subset that
will only include the features ``a``, ``b``, ``e``, and features that
will tell whether ``a`` and ``b`` are equal and whether ``e`` is 1
(part of :download:`lookup-lookup.py <code/lookup-lookup.py>`):

..
    .. literalinclude:: code/lookup-lookup.py
        :lines: 7-21

.. testcode::

    import Orange

    monks = Orange.data.Table("monks-1")

    a, b, e = monks.domain["a"], monks.domain["b"], monks.domain["e"]

    ab = Orange.feature.Discrete("a==b", values = ["no", "yes"])
    ab.get_value_from = Orange.classification.lookup.ClassifierByLookupTable(ab, a, b,
                        ["yes", "no", "no",  "no", "yes", "no",  "no", "no", "yes"])

    e1 = Orange.feature.Discrete("e==1", values = ["no", "yes"])
    e1.get_value_from = Orange.classification.lookup.ClassifierByLookupTable(e1, e,
                        ["yes", "no", "no", "no", "?"])

    monks2 = monks.select([a, b, ab, e, e1, monks.domain.class_var])
    
We can check the correctness of the script by printing out several
random examples from table ``monks2``.

    >>> for i in range(5):
    ...     print monks2.randomexample()
    ['3', '2', 'no', '2', 'no', '0']
    ['2', '2', 'yes', '2', 'no', '1']
    ['1', '2', 'no', '2', 'no', '0']
    ['2', '3', 'no', '1', 'yes', '1']
    ['1', '3', 'no', '1', 'yes', '1']

The first :obj:`ClassifierByLookupTable` takes values of features ``a``
and ``b`` and computes the value of ``ab`` according to the rule given in the
given table. The first three values correspond to ``a=1`` and ``b=1,2,3``;
for the first combination, value of ``ab`` should be "yes", for the other
two ``a`` and ``b`` are different. The next triplet corresponds to ``a=2``;
here, the middle value is "yes"...

The second lookup is simpler: since it involves only a single feature,
the list is a simple one-to-one mapping from the four-valued ``e`` to the
two-valued ``e1``. The last value in the list is returned when ``e`` is unknown
and tells that ``e1`` should be unknown then as well.

Note that :obj:`ClassifierByLookupTable` is not needed for this.
The new feature ``e1`` could be computed with a callback to Python,
for instance::

    e2.get_value_from = lambda ex, rw: orange.Value(e2, ex["e"] == "1")


Classifiers by lookup table
===========================

.. index::
   single: classification; lookup table

Although the above example used :obj:`ClassifierByLookupTable` as if
it was a concrete class, :obj:`ClassifierByLookupTable` is actually
abstract. Calling its constructor does not return an instance of
:obj:`ClassifierByLookupTable`, but either
:obj:`ClassifierByLookupTable1`, :obj:`ClassifierByLookupTable2` or
:obj:`ClassifierByLookupTable3`, that take one (``e``, above), two
(like ``a`` and ``b``) or three features, respectively. Class
predictions for each combination of feature values are stored in a
(one dimensional) table. To classify an instance, the classifier
computes an index of the element of the table that corresponds to the
combination of feature values.

These classifiers are built to be fast, not safe. For instance, if the
number of values for one of the features is changed, Orange will most
probably crash.  To alleviate this, many of these classes' attributes
are read-only and can only be set when the object is constructed.


.. py:class:: ClassifierByLookupTable(class_var, variable1[, variable2[, variable3]] [, lookup_table[, distributions]])
    
    A general constructor that, based on the number of feature
    descriptors, constructs one of the three classes discussed. If
    :obj:`lookup_table` and :obj:`distributions` are omitted, the
    constructor also initializes them to two lists of the right sizes,
    but their elements are missing values and empty distributions. If
    they are given, they must be of correct size.
    
    .. attribute:: variable1[, variable2[, variable3]](read only)
        
        The feature(s) that the classifier uses for classification.
        :obj:`ClassifierByLookupTable1` only has :obj:`variable1`,
        :obj:`ClassifierByLookupTable2` also has :obj:`variable2` and
        :obj:`ClassifierByLookupTable3` has all three.

    .. attribute:: variables (read only)
        
        The above variables, returned as a tuple.

    .. attribute:: no_of_values1[, no_of_values2[, no_of_values3]] (read only)
        
        The number of values for :obj:`variable1`, :obj:`variable2`
        and :obj:`variable3`. This is stored here to make the
        classifier faster. These attributes are defined only for
        :obj:`ClassifierByLookupTable2` (the first two) and
        :obj:`ClassifierByLookupTable3` (all three).

    .. attribute:: lookup_table (read only)
        
        A list of values, one for each possible combination of
        features. For :obj:`ClassifierByLookupTable1`, there is an
        additional element that is returned when the feature's value
        is unknown. Values are ordered by values of features, with
        :obj:`variable1` being the most important. For instance, for
        two three-valued features, the elements of :obj:`lookup_table`
        correspond to combinations (1, 1), (1, 2), (1, 3), (2, 1), (2,
        2), (2, 3), (3, 1), (3, 2), (3, 3).
        
        The attribute is read-only; it cannot be assigned a new list,
        but the existing list can be changed. Changing its size will
        most likely crash Orange.

    .. attribute:: distributions (read only)
        
        Similar to :obj:`lookup_table`, but storing a distribution for
        each combination of values. 

    .. attribute:: data_description
        
        An object of type :obj:`EFMDataDescription`, defined only for
        :obj:`ClassifierByLookupTable2` and
        :obj:`ClassifierByLookupTable3`. They use it to make
        predictions when one or more feature values are missing.
        :obj:`ClassifierByLookupTable1` does not need it since this
        case is covered by an additional element in
        :obj:`lookup_table` and :obj:`distributions`, as described
        above.
        
    .. method:: get_index(inst)
    
        Returns an index of in :obj:`lookup_table` and
        :obj:`distributions` that corresponds to the given data
        instance ``inst`` . The formula depends upon the type of the
        classifier. If value\ *i* is int(example[variable\ *i*]), then
        the corresponding formulae are

        ``ClassifierByLookupTable1``:
            index = value1, or len(lookup_table) - 1 if value of :obj:`variable1` is missing

        ``ClassifierByLookupTable2``:
            index = value1 * no_of_values1 + value2, or -1 if ``value1`` or ``value2`` is missing

        ClassifierByLookupTable3:
            index = (value1 * no_of_values1 + value2) * no_of_values2 + value3, or -1 if any value is missing

.. py:class:: ClassifierByLookupTable1(class_var, variable1 [, lookup_table, distributions])
    
    Uses a single feature for lookup. See
    :obj:`ClassifierByLookupTable` for more details.

.. py:class:: ClassifierByLookupTable2(class_var, variable1, variable2, [, lookup_table[, distributions]])
    
    Uses two features for lookup. See
    :obj:`ClassifierByLookupTable` for more details.
        
.. py:class:: ClassifierByLookupTable3(class_var, variable1, variable2, variable3, [, lookup_table[, distributions]])
    
    Uses three features for lookup. See
    :obj:`ClassifierByLookupTable` for more details.


Classifier by data table
========================

.. index::
   single: classification; data table

:obj:`ClassifierByDataTable` is used in similar contexts as
:obj:`ClassifierByLookupTable`. The class is much slower so it is recommended to use :obj:`ClassifierByLookupTable` if the number of features is less than four.

.. py:class:: ClassifierByDataTable

    :obj:`ClassifierByDataTable` is the alternative to
    :obj:`ClassifierByLookupTable` for more than three features.
    Instead of having a lookup table, it stores the data in
    :obj:`Orange.data.Table` that is optimized for faster access.
    
    .. attribute:: sorted_examples
        
        A :obj:`Orange.data.Table` with sorted data instances for
        lookup.  If there were multiple instances with the same
        feature values (but possibly different classes) in the
        original data, they can be merged into a single
        instance. Regardless of merging, class values in this table
        are distributed: their ``svalue`` contains a
        :obj:`~Orange.statistics.distribution.Distribution`.

    .. attribute:: classifier_for_unknown
        
        The classifier for instances that are not found in the
        table. If not set, :obj:`ClassifierByDataTable` returns
        missing value for such instances.

    .. attribute:: variables (read only)
        
        A tuple with features in the domain. Equal to
        :obj:`domain.features`, but here for similarity with
        :obj:`ClassifierByLookupTable`.



.. py:class:: LookupLearner
    
    A learner that constructs a table for
    :obj:`ClassifierByDataTable.sorted_examples`. It sorts the data
    instances and merges those with the same feature values.
    
    The constructor returns an instance of :obj:`LookupLearners`,
    unless the data is provided, in which case it return
    :obj:`ClassifierByDataTable`.

    :obj:`LookupLearner` also supports a different call signature than
    other learners. Besides instances, it accepts a new class
    variable and the features that should be used for
    classification. 

part of :download:`lookup-table.py <code/lookup-table.py>`:

..
    .. literalinclude:: code/lookup-table.py
        :lines: 7-13

.. testcode::
        
    import Orange

    table = Orange.data.Table("monks-1")
    a, b, e = table.domain["a"], table.domain["b"], table.domain["e"]

    table_s = table.select([a, b, e, table.domain.class_var])
    abe = Orange.classification.lookup.LookupLearner(table_s)


In ``table_s``, we have prepared a table in which instances are described
only by ``a``, ``b``, ``e`` and the class. The learner constructs a
:obj:`ClassifierByDataTable` and stores instances from ``table_s`` into its
:obj:`~ClassifierByDataTable.sorted_examples`. Instances are merged so that
there are no duplicates.

    >>> print len(table_s)
    556
    >>> print len(abe.sorted_examples)
    36
    >>> for i in abe.sorted_examples[:10]:  # doctest: +SKIP
    ...     print i
    ['1', '1', '1', '1']
    ['1', '1', '2', '1']
    ['1', '1', '3', '1']
    ['1', '1', '4', '1']
    ['1', '2', '1', '1']
    ['1', '2', '2', '0']
    ['1', '2', '3', '0']
    ['1', '2', '4', '0']
    ['1', '3', '1', '1']
    ['1', '3', '2', '0']

Each instance's class value also stores the distribution of classes
for all instances that were merged into it. In our case, the three
features suffice to unambiguously determine the classes and, since
instances cover the entire space, all distributions have 12
instances in one of the class and none in the other.

    >>> for i in abe.sorted_examples[:10]:  # doctest: +SKIP
    ...     print i, i.get_class().svalue
    ['1', '1', '1', '1'] <0.000, 12.000>
    ['1', '1', '2', '1'] <0.000, 12.000>
    ['1', '1', '3', '1'] <0.000, 12.000>
    ['1', '1', '4', '1'] <0.000, 12.000>
    ['1', '2', '1', '1'] <0.000, 12.000>
    ['1', '2', '2', '0'] <12.000, 0.000>
    ['1', '2', '3', '0'] <12.000, 0.000>
    ['1', '2', '4', '0'] <12.000, 0.000>
    ['1', '3', '1', '1'] <0.000, 12.000>
    ['1', '3', '2', '0'] <12.000, 0.000>

A typical use of :obj:`ClassifierByDataTable` is to construct a new
feature and put the classifier into its
:obj:`~Orange.feature.Descriptor.get_value_from`.

    >>> y2 = Orange.feature.Discrete("y2", values = ["0", "1"])
    >>> y2.get_value_from = abe

Although ``abe`` determines the value of ``y2``, ``abe.class_var`` is
still ``y``.  Orange does not complain about the mismatch.

Using the specific :obj:`LookupLearner`'s call signature can save us
from constructing `table_s` and reassigning the
:obj:`~Orange.data.Domain.class_var`, but it still does not set the
:obj:`~Orange.feature.Descriptor.get_value_from`.

part of :download:`lookup-table.py <code/lookup-table.py>`::

    import Orange

    table = Orange.data.Table("monks-1")
    a, b, e = table.domain["a"], table.domain["b"], table.domain["e"]

    y2 = Orange.feature.Discrete("y2", values = ["0", "1"])
    abe2 = Orange.classification.lookup.LookupLearner(y2, [a, b, e], table)

For the final example, :obj:`LookupLearner`'s alternative call
arguments offers an easy way to observe feature interactions. For this
purpose, we shall omit ``e``, and construct a
:obj:`ClassifierByDataTable` from ``a`` and ``b`` only (part of
:download:`lookup-table.py <code/lookup-table.py>`):

.. literalinclude:: code/lookup-table.py
    :lines: 32-35

The script's output show how the classes are distributed for different
values of ``a`` and ``b``::

    ['1', '1', '1'] <0.000, 48.000>
    ['1', '2', '0'] <36.000, 12.000>
    ['1', '3', '0'] <36.000, 12.000>
    ['2', '1', '0'] <36.000, 12.000>
    ['2', '2', '1'] <0.000, 48.000>
    ['2', '3', '0'] <36.000, 12.000>
    ['3', '1', '0'] <36.000, 12.000>
    ['3', '2', '0'] <36.000, 12.000>
    ['3', '3', '1'] <0.000, 48.000>

For instance, when ``a`` is '1' and ``b`` is '3', the majority class is '0',
and the class distribution is 36:12 in favor of '0'.


Utility functions
=================


There are several functions related to the above classes.

.. function:: lookup_from_function(class_var, bound, function)

    Construct a :obj:`ClassifierByLookupTable` or
    :obj:`ClassifierByDataTable` with the given bound variables and
    then use the function to initialize the lookup table.

    The function is given the values of features as integer indices and
    must return an integer index of the `class_var`'s value.

    The following example constructs a new feature called ``a=b``
    whose value will be "yes" when ``a`` and ``b`` are equal and "no"
    when they are not. We will then add the feature to the data set.
    
        >>> bound = [table.domain[name] for name in ["a", "b"]]
        >>> new_var = Orange.feature.Discrete("a=b", values=["no", "yes"])
        >>> lookup = Orange.classification.lookup.lookup_from_function(new_var, bound, lambda x: x[0] == x[1])
        >>> new_var.get_value_from = lookup
        >>> import orngCI
        >>> table2 = orngCI.addAnAttribute(new_var, table)
        >>> for i in table2[:30]:
        ...     print i
        ['1', '1', '1', '1', '3', '1', 'yes', '1']
        ['1', '1', '1', '1', '3', '2', 'yes', '1']
        ['1', '1', '1', '3', '2', '1', 'yes', '1']
        ...
        ['1', '2', '1', '1', '1', '2', 'no', '1']
        ['1', '2', '1', '1', '2', '1', 'no', '0']
        ['1', '2', '1', '1', '3', '1', 'no', '0']
        ...

    The feature was inserted with use of ``orngCI.addAnAttribute``. By setting
    ``new_var.get_value_from`` to ``lookup`` we state that when converting domains
    (either when needed by ``addAnAttribute`` or at some other place), ``lookup``
    should be used to compute ``new_var``'s value.

.. function:: lookup_from_data(examples [, weight])

    Take a set of data instances (e.g. :obj:`Orange.data.Table`) and
    turn it into a classifier. If there are one, two or three features
    and no ambiguous data instances (i.e. no instances with same
    feature values and different classes), it will construct an
    appropriate :obj:`ClassifierByLookupTable`. Otherwise, it will
    return an :obj:`ClassifierByDataTable`.
    
        >>> lookup = Orange.classification.lookup.lookup_from_data(table)
        >>> test_instance = Orange.data.Instance(table.domain, ['3', '2', '2', '3', '4', '1', '?'])
        >>> lookup(test_instance)
        <orange.Value 'y'='0'>
    
.. function:: dump_lookup_function(func)

    Returns a string with a lookup function. Argument ``func`` can be
    any of the above-mentioned classifiers or a feature whose
    :obj:`~Orange.feature.Descriptor.get_value_from` contains one of
    such classifiers.

    For instance, if ``lookup`` is such as constructed in the example for
    ``lookup_from_function``, it can be printed by::
    
        >>> print dump_lookup_function(lookup)
        a      b      a=b
        ------ ------ ------
        1      1      yes
        1      2      no
        1      3      no
        2      1      no
        2      2      yes
        2      3      no
        3      1      no
        3      2      no
        3      3      yes

