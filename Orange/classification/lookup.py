"""

.. index:: classification; lookup

*******************************
Lookup classifiers (``lookup``)
*******************************

Lookup classifiers predict classes by looking into stored lists of
cases. There are two kinds of such classifiers in Orange. The simpler
and faster :obj:`ClassifierByLookupTable` uses up to three discrete
features and has a stored mapping from values of those features to the
class value. The more complex classifiers store an
:obj:`Orange.data.Table` and predict the class by matching the instance
to instances in the table.

.. index::
   single: feature construction; lookup classifiers

A natural habitat for these classifiers is feature construction:
they usually reside in :obj:`~Orange.feature.Descriptor.get_value_from` fields of constructed
features to facilitate their automatic computation. For instance,
the following script shows how to translate the `monks-1.tab` data set
features into a more useful subset that will only include the features
``a``, ``b``, ``e``, and features that will tell whether ``a`` and ``b`` are equal and
whether ``e`` is 1 (don't bother about the details, they follow later; 
:download:`lookup-lookup.py <code/lookup-lookup.py>`):

.. literalinclude:: code/lookup-lookup.py
    :lines: 7-21
    
We can check the correctness of the script by printing out several
random examples from ``data2``.

    >>> for i in range(5):
    ...     print table2.randomexample()
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

Note that you don't need :obj:`ClassifierByLookupTable` for this.
The new feature ``e1`` could be computed with a callback to Python,
for instance::

    e2.get_value_from = lambda ex, rw: orange.Value(e2, ex["e"] == "1")


Classifiers by lookup table
===========================

.. index::
   single: classification; lookup table

Although the above example used :obj:`ClassifierByLookupTable` as if it
was a concrete class, ClassifierByLookupTable is actually
abstract. Calling its constructor is a typical Orange trick: what you
get, is never ClassifierByLookupTable, but either
:obj:`ClassifierByLookupTable1`, :obj:`ClassifierByLookupTable2` or
:obj:`ClassifierByLookupTable3`. As their names tell, the first
classifies using a single feature (so that's what we had for ``e1``),
the second uses a pair of features (and has been constructed for ``ab``
above), and the third uses three features. Class predictions for each
combination of feature values are stored in a (one dimensional) table.
To classify an instance, the classifier computes an index of the element
of the table that corresponds to the combination of feature values.

These classifiers are built to be fast, not safe. If you, for instance,
change the number of values for one of the features, Orange will
most probably crash. To protect you somewhat, many of these classes'
features are read-only and can only be set when the object is
constructed.


.. py:class:: ClassifierByLookupTable(class_var, variable1[, variable2[, variable3]] [, lookup_table[, distributions]])
    
    A general constructor that, based on the number of feature descriptors,
    constructs one of the three classes discussed. If :obj:`lookup_table`
    and :obj:`distributions` are omitted, the constructor also initializes
    them to two lists of the right sizes, but their elements are don't knows
    and empty distributions. If they are given, they must be of correct size.
    
    .. attribute:: variable1[, variable2[, variable3]](read only)
        
        The feature(s) that the classifier uses for classification.
        ClassifierByLookupTable1 only has variable1,
        ClassifierByLookupTable2 also has variable2 and
        ClassifierByLookupTable3 has all three.

    .. attribute:: variables (read only)
        
        The above variables, returned as a tuple.

    .. attribute:: no_of_values1[, no_of_values2[, no_of_values3]] (read only)
        
        The number of values for variable1, variable2 and variable3.
        This is stored here to make the classifier faster. Those features
        are defined only for ClassifierByLookupTable2 (the first two) and
        ClassifierByLookupTable3 (all three).

    .. attribute:: lookup_table (read only)
        
        A list of values (:obj:`Orange.core.ValueList`), one for each possible
        combination of features. For ClassifierByLookupTable1, there is an
        additional element that is returned when the feature's value is
        unknown. Values are ordered by values of features, with variable1
        being the most important. In case of two three valued features, the
        list order is therefore 1-1, 1-2, 1-3, 2-1, 2-2, 2-3, 3-1, 3-2, 3-3,
        where the first digit corresponds to variable1 and the second to
        variable2.
        
        The list is read-only in the sense that you cannot assign a new
        list to this field. You can, however, change its elements. Don't
        change its size, though. 

    .. attribute:: distributions (read only)
        
        Similar to :obj:`lookup_table`, but is of type
        :obj:`Orange.core.DistributionList` and stores a distribution
        for each combination of values. 

    .. attribute:: data_description
        
        An object of type :obj:`EFMDataDescription`, defined only for
        ClassifierByLookupTable2 and ClassifierByLookupTable3. They use it
        to make predictions when one or more feature values are unknown.
        ClassifierByLookupTable1 doesn't need it since this case is covered by
        an additional element in :obj:`lookup_table` and :obj:`distributions`,
        as told above.
        
    .. method:: get_index(example)
    
        Returns an index of ``example`` in :obj:`lookup_table` and
        :obj:`distributions`. The formula depends upon the type of
        the classifier. If value\ *i* is int(example[variable\ *i*]),
        then the corresponding formulae are

        ClassifierByLookupTable1:
            index = value1, or len(lookup_table) - 1 if value is unknown
        ClassifierByLookupTable2:
            index = value1 * no_of_values1 + value2, or -1 if any value is unknown
        ClassifierByLookupTable3:
            index = (value1 * no_of_values1 + value2) * no_of_values2 + value3, or -1 if any value is unknown

        Let's see some indices for randomly chosen examples from the original table.
        
        part of :download:`lookup-lookup.py <code/lookup-lookup.py>`:

        .. literalinclude:: code/lookup-lookup.py
            :lines: 26-29
        
        Output::
        
            ['3', '2', '1', '2', '2', '1', '0']: ab 7, e1 1 
            ['2', '2', '1', '2', '2', '1', '1']: ab 4, e1 1 
            ['1', '2', '1', '2', '2', '2', '0']: ab 1, e1 1 
            ['2', '3', '2', '3', '1', '1', '1']: ab 5, e1 0 
            ['1', '3', '2', '2', '1', '1', '1']: ab 2, e1 0 



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
:obj:`ClassifierByLookupTable`. If you write, for instance, a
constructive induction algorithm, it is recommended that the values
of the new feature are computed either by one of classifiers by lookup
table or by ClassifierByDataTable, depending on the number of bound
features.

.. py:class:: ClassifierByDataTable

    :obj:`ClassifierByDataTable` is the alternative to
    :obj:`ClassifierByLookupTable`. It is to be used when the
    classification is based on more than three features. Instead of having
    a lookup table, it stores an :obj:`Orange.data.Table`, which is
    optimized for a faster access.
    

    .. attribute:: sorted_examples
        
        A :obj:`Orange.data.Table` with sorted data instances for lookup.
        Instances in the table can be merged; if there were multiple
        instances with the same feature values (but possibly different
        classes), they are merged into a single instance. Regardless of
        merging, class values in this table are distributed: their svalue
        contains a :obj:`Distribution`.

    .. attribute:: classifierForUnknown
        
        This classifier is used to classify instances which were not found
        in the table. If classifierForUnknown is not set, don't know's are
        returned.

    .. attribute:: variables (read only)
        
        A tuple with features in the domain. This field is here so that
        :obj:`ClassifierByDataTable` appears more similar to
        :obj:`ClassifierByLookupTable`. If a constructive induction
        algorithm returns the result in one of these classifiers, and you
        would like to check which features are used, you can use variables
        regardless of the class you actually got.

    There are no specific methods for ClassifierByDataTable.
    Since this is a classifier, it can be called. When the instance to be
    classified includes unknown values, :obj:`classifierForUnknown` will be
    used if it is defined.



.. py:class:: LookupLearner
    
    Although :obj:`ClassifierByDataTable` is not really a classifier in
    the sense that you will use it to classify instances, but is rather a
    function for computation of intermediate values, it has an associated
    learner, :obj:`LookupLearner`. The learner's task is, basically, to
    construct a Table for :obj:`sorted_examples`. It sorts them, merges them
    and, of course, regards instance weights in the process as well.
    
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

part of :download:`lookup-table.py <code/lookup-table.py>`:

.. literalinclude:: code/lookup-table.py
    :lines: 7-13


In data_s, we have prepared a table in which instances are described
only by a, b, e and the class. Learner constructs a
ClassifierByDataTable and stores instances from data_s into its
sorted_examples. Instances are merged so that there are no duplicates.

    >>> print len(table_s)
    556
    >>> print len(abe.sorted_examples)
    36
    >>> for i in abe.sorted_examples[:10]:
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

Well, there's a bit more here than meets the eye: each instance's class
value also stores the distribution of classes for all instances that
were merged into it. In our case, the three features suffice to
unambiguously determine the classes and, since instances covered the
entire space, all distributions have 12 instances in one of the class
and none in the other.

    >>> for i in abe.sorted_examples[:10]:
    ...     print i, i.getclass().svalue
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

ClassifierByDataTable will usually be used by :obj:`Orange.feature.Descriptor.get_value_from`. So, we
would probably continue this by constructing a new feature and put the
classifier into its :obj:`Orange.feature.Descriptor.get_value_from`.

    >>> y2 = Orange.feature.Discrete("y2", values = ["0", "1"])
    >>> y2.get_value_from = abe

There's something disturbing here. Although abe determines the value of
y2, abe.class_var is still y. Orange doesn't bother (the whole example
is artificial - you will seldom pack the entire data set in an
ClassifierByDataTable...), so shouldn't you. But still, for the sake
of hygiene, you can conclude by

    >>> abe.class_var = y2

The whole story can be greatly simplified. LookupLearner can also be
called differently than other learners. Besides instances, you can pass
the new class variable and the features that should be used for
classification. This saves us from constructing data_s and reassigning
the class_var. It doesn't set the :obj:`Orange.feature.Descriptor.get_value_from`, though.

part of :download:`lookup-table.py <code/lookup-table.py>`::

    import Orange

    table = Orange.data.Table("monks-1")
    a, b, e = table.domain["a"], table.domain["b"], table.domain["e"]

    y2 = Orange.feature.Discrete("y2", values = ["0", "1"])
    abe2 = Orange.classification.lookup.LookupLearner(y2, [a, b, e], table)

Let us, for the end, show another use of LookupLearner. With the
alternative call arguments, it offers an easy way to observe feature
interactions. For this purpose, we shall omit e, and construct a
ClassifierByDataTable from a and b only (part of :download:`lookup-table.py <code/lookup-table.py>`):

.. literalinclude:: code/lookup-table.py
    :lines: 32-35

The script's output show how the classes are distributed for different
values of a and b::

    ['1', '1', '1'] <0.000, 48.000>
    ['1', '2', '0'] <36.000, 12.000>
    ['1', '3', '0'] <36.000, 12.000>
    ['2', '1', '0'] <36.000, 12.000>
    ['2', '2', '1'] <0.000, 48.000>
    ['2', '3', '0'] <36.000, 12.000>
    ['3', '1', '0'] <36.000, 12.000>
    ['3', '2', '0'] <36.000, 12.000>
    ['3', '3', '1'] <0.000, 48.000>

For instance, when a is '1' and b is '3', the majority class is '0',
and the class distribution is 36:12 in favor of '0'.


Utility functions
=================


There are several functions for working with classifiers that use a stored
data table for making predictions. There are four such classifiers; the most
general stores an :class:`Orange.data.Table` and the other three are
specialized and optimized for cases where the domain contains only one, two or
three features (besides the class variable).

.. function:: lookup_from_bound(classVar, bound)

    This function constructs an appropriate lookup classifier for one, two or
    three features. If there are more, it returns None. The resulting
    classifier is of type :obj:`ClassifierByLookupTable`,
    :obj:`ClassifierByLookupTable2` or :obj:`ClassifierByLookupTable3`, with
    classVar and bound set set as given.

    If, for instance, table contains a data set Monk 1 and you would like to
    construct a new feature from features a and b, you can call this function
    as follows.
    
        >>> newvar = Orange.feature.Discrete()
        >>> bound = [table.domain[name] for name in ["a", "b"]]
        >>> lookup = lookup_from_bound(newvar, bound)
        >>> print lookup.lookup_table
        <?, ?, ?, ?, ?, ?, ?, ?, ?>

    Function lookup_from_bound does not initialize neither newVar nor
    the lookup table...

.. function:: lookup_from_function(classVar, bound, function)

    ... and that's exactly where lookup_from_function differs from
    :obj:`lookup_from_bound`. lookup_from_function first calls
    lookup_from_bound and then uses the function to initialize the lookup
    table. The other difference between this and the previous function is that
    lookup_from_function also accepts bound sets with more than three
    features. In this case, it construct a :obj:`ClassifierByDataTable`.

    The function gets the values of features as integer indices and should
    return an integer index of the "class value". The class value must be
    properly initialized.

    For exercise, let us construct a new feature called a=b whose value will
    be "yes" when a and b are equal and "no" when they are not. We will then
    add the feature to the data set.
    
        >>> bound = [table.domain[name] for name in ["a", "b"]]
        >>> newVar = Orange.feature.Discrete("a=b", values=["no", "yes"])
        >>> lookup = lookup_from_function(newVar, bound, lambda x: x[0] == x[1])
        >>> newVar.get_value_from = lookup
        >>> import orngCI
        >>> table2 = orngCI.addAnAttribute(newVar, table)
        >>> for i in table2[:30]:
            ... print i
        ['1', '1', '1', '1', '1', '1', 'yes', '1']
        ['1', '1', '1', '1', '1', '2', 'yes', '1']
        ['1', '1', '1', '1', '2', '1', 'yes', '1']
        ['1', '1', '1', '1', '2', '2', 'yes', '1']
        ...
        ['2', '1', '2', '3', '4', '1', 'no', '0']
        ['2', '1', '2', '3', '4', '2', 'no', '0']
        ['2', '2', '1', '1', '1', '1', 'yes', '1']
        ['2', '2', '1', '1', '1', '2', 'yes', '1']
        ...

    The feature was inserted with use of orngCI.addAnAttribute. By setting
    newVar.get_value_from to lookup we state that when converting domains
    (either when needed by addAnAttribute or at some other place), lookup
    should be used to compute newVar's value. (A bit off topic, but
    important: you should never call :obj:`Orange.feature.Descriptor.get_value_from` directly, but always call
    it through computeValue.)

.. function:: lookup_from_data(examples [, weight])

    This function takes a set of examples (e.g. :obj:`Orange.data.Table`)
    and turns it into a classifier. If there are one, two or three features and
    no ambiguous examples (examples are ambiguous if they have same values of
    features but with different class values), it will construct an appropriate
    :obj:`ClassifierByLookupTable`. Otherwise, it will return an
    :obj:`ClassifierByDataTable`.
    
        >>> lookup = lookup_from_data(table)
        >>> test_instance = Orange.data.Instance(table.domain, ['3', '2', '2', '3', '4', '1', '?'])
        >>> lookup(test_instance)
        <orange.Value 'y'='0'>
    
.. function:: dump_lookup_function(func)

    dump_lookup_function returns a string with a lookup function in
    tab-delimited format. Argument func can be any of the above-mentioned
    classifiers or a feature whose :obj:`Orange.feature.Descriptor.get_value_from` points to one of such
    classifiers.

    For instance, if lookup is such as constructed in the example for
    lookup_from_function, you can print it out by::
    
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

"""

import Orange.data
from Orange.core import \
        LookupLearner, \
         ClassifierByLookupTable, \
              ClassifierByLookupTable1, \
              ClassifierByLookupTable2, \
              ClassifierByLookupTable3, \
              ClassifierByExampleTable as ClassifierByDataTable


def lookup_from_bound(attribute, bound):
    if not len(bound):
        raise TypeError, "no bound attributes"
    elif len(bound) <= 3:
        return [ClassifierByLookupTable, ClassifierByLookupTable2,
                ClassifierByLookupTable3][len(bound) - 1](attribute, *list(bound))
    else:
        return None

    
def lookup_from_function(attribute, bound, function):
    """Constructs ClassifierByDataTable or ClassifierByLookupTable
    mirroring the given function
    
    """
    lookup = lookup_from_bound(attribute, bound)
    if lookup:
        lookup.lookup_table = [Orange.data.Value(attribute, function(attributes))
                              for attributes in Orange.misc.counters.LimitedCounter(
                                  [len(attr.values) for attr in bound])]
        return lookup
    else:
        examples = Orange.data.Table(Orange.data.Domain(bound, attribute))
        for attributes in Orange.misc.counters.LimitedCounter([len(attr.values)
                                                   for attr in dom.attributes]):
            examples.append(Orange.data.Example(dom, attributes +
                                                [function(attributes)]))
        return LookupLearner(examples)
      

from Orange.misc import deprecated_keywords
@deprecated_keywords({"learnerForUnknown":"learner_for_unknown"})
def lookup_from_data(examples, weight=0, learner_for_unknown=None):
    if len(examples.domain.attributes) <= 3:
        lookup = lookup_from_bound(examples.domain.class_var,
                                 examples.domain.attributes)
        lookup_table = lookup.lookup_table
        for example in examples:
            ind = lookup.getindex(example)
            if not lookup_table[ind].isSpecial() and (lookup_table[ind] !=
                                                     example.getclass()):
                break
            lookup_table[ind] = example.getclass()
        else:
            return lookup

        # there are ambiguities; a backup plan is
        # ClassifierByDataTable, let it deal with them
        return LookupLearner(examples, weight,
                             learner_for_unknown=learner_for_unknown)

    else:
        return LookupLearner(examples, weight,
                             learner_for_unknown=learner_for_unknown)
        
        
def dump_lookup_function(func):
    if isinstance(func, Orange.feature.Descriptor):
        if not func.get_value_from:
            raise TypeError, "attribute '%s' does not have an associated function" % func.name
        else:
            func = func.get_value_from

    outp = ""
    if isinstance(func, ClassifierByDataTable):
    # XXX This needs some polishing :-)
        for i in func.sorted_examples:
            outp += "%s\n" % i
    else:
        boundset = func.boundset()
        for a in boundset:
            outp += "%s\t" % a.name
        outp += "%s\n" % func.class_var.name
        outp += "------\t" * (len(boundset)+1) + "\n"
        
        lc = 0
        if len(boundset)==1:
            cnt = Orange.misc.counters.LimitedCounter([len(x.values)+1 for x in boundset])
        else:
            cnt = Orange.misc.counters.LimitedCounter([len(x.values) for x in boundset])
        for ex in cnt:
            for i in range(len(ex)):
                if ex[i] < len(boundset[i].values):
                    outp += "%s\t" % boundset[i].values[ex[i]]
                else:
                    outp += "?\t",
            outp += "%s\n" % func.class_var.values[int(func.lookup_table[lc])]
            lc += 1
    return outp
