"""

.. index:: lookup

Lookup classifiers predict classes by looking into stored lists of
cases. There are two kinds of such classifiers in Orange. The simpler
and fastest :obj:`ClassifierByLookupTable` use up to three discrete
features and have a stored mapping from values of those features to
class value. The more complex classifiers store a
:obj:`Orange.data.Table` and predict the class by matching the instance
to instances in the table.

The natural habitat of these classifiers is feature construction:
they usually reside in :obj:`getValueFrom` fields of constructed
features to facilitate their automatic computation. For instance,
the following script shows how to translate the `monks-1.tab`_ dataset
features into a more useful subset that will only include the features
a, b, e, and features that will tell whether a and b are equal and
whether e is 1 (don't bother about the details, they follow later).

part of `lookup-lookup.py`_ (uses: `monks-1.tab`_):

.. literalinclude:: code/lookup-lookup.py
    :lines: 7-21
    
We can check the correctness of the script by printing out several
random examples from data2.

    >>> for i in range(5):
    ...     print data2.randomexample()
    ['1', '1', 'yes', '4', 'no', '1']
    ['3', '3', 'yes', '2', 'no', '1']
    ['2', '1', 'no', '4', 'no', '0']
    ['2', '1', 'no', '1', 'yes', '1']
    ['1', '1', 'yes', '3', 'no', '1']

The first :obj:`ClassifierByLookupTable` takes values of features a
and b and computes the value of ab according to the rule given in the
given table. The first three values correspond to a=1 and b=1, 2, 3;
for the first combination, value of ab should be "yes", for the other
two a and b are different. The next triplet correspond to a=2;
here, the middle value is "yes"...

The second lookup is simpler: since it involves only a single feature,
the list is a simple one-to-one mapping from the four-valued e to the
two-valued e1. The last value in the list is returned when e is unknown
and tells that e1 should be unknown then as well.

Note that you don't need :obj:`ClassifierByLookupTable` for this.
The new feature e1 could be computed with a callback to Python,
for instance::

    e2.getValueFrom = lambda ex, rw: orange.Value(e2, ex["e"]=="1")

===========================
Classifiers by Lookup Table
===========================

Although the above example used :obj:`ClassifierByLookupTable` as if it
was a concrete class, :obj:`ClassifierByLookupTable` is actually
abstract. Calling its constructor is a typical Orange trick: what you
get, is never :obj:`ClassifierByLookupTable`, but either
:obj:`ClassifierByLookupTable1`, :obj:`ClassifierByLookupTable2` or
:obj:`ClassifierByLookupTable3`. As their names tell, the first
classifies using a single feature (so that's what we had for e1),
the second uses a pair of features (and has been constructed for ab
above), and the third uses three features. Class predictions for each
combination of feature values are stored in a (one dimensional) table.
To classify an instance, the classifier computes an index of the element
of the table that corresponds to the combination of feature values.

These classifiers are built to be fast, not safe. If you, for instance,
change the number of values for one of the features, Orange will
most probably crash. To protect you somewhat, many of these classes'
features are read-only and can only be set when the object is
constructed.

**Attributes:**

.. attribute:: variable1[, variable2[, variable3]](read only)
    
    The feature(s) that the classifier uses for classification.
    ClassifierByLookupTable1 only has variable1,
    ClassifierByLookupTable2 also has variable2 and
    ClassifierByLookupTable3 has all three.

.. attribute:: variables (read only)
    
    The above variables, returned as a tuple.

.. attribute:: noOfValues1, noOfValues2[, noOfValues3] (read only)
    
    The number of values for variable1, variable2 and variable3.
    This is stored here to make the classifier faster. Those features
    are defined only for ClassifierByLookupTable2 (the first two) and
    ClassifierByLookupTable3 (all three).

.. attribute:: lookupTable (read only)
    
    A list of values (ValueList), one for each possible combination of
    features. For ClassifierByLookupTable1, there is an additional
    element that is returned when the feature's value is unknown.
    Values are ordered by values of features, with variable1 being the
    most important. In case of two three valued features, the list
    order is therefore 1-1, 1-2, 1-3, 2-1, 2-2, 2-3, 3-1, 3-2, 3-3,
    where the first digit corresponds to variable1 and the second to
    variable2.
    
    The list is read-only in the sense that you cannot assign a new
    list to this field. You can, however, change its elements. Don't
    change its size, though. 

.. attribute:: distributions (read only)
    
    Similar to :obj:`lookupTable`, but is of type DistributionList
    and stores a distribution for each combination of values. 

.. attribute:: dataDescription
    
    An object of type EFMDataDescription, defined only for
    ClassifierByLookupTable2 and ClassifierByLookupTable3. They use
    it to make predictions when one or more feature values are unknown.
    ClassifierByLookupTable1 doesn't need it since this case is covered
    by an additional element in lookupTable and distributions,
    as told above. 

**Methods:**

.. method:: ClassifierByLookupTable(classVar, variable1[, variable2[, variable3]] [, lookupTable[, distributions]])

    A general constructor that, based on the number of attribute
    descriptors, constructs one of the three classes discussed.
    If lookupTable and distributions are omitted, constructor also
    initializes lookupTable and distributions to two lists of the
    right sizes, but their elements are don't knows and empty
    distributions. If they are given, they must be of correct size.

.. method:: ClassifierByLookupTable1(classVar, variable1 [, lookupTable, distributions])
            ClassifierByLookupTable2(classVar, variable1, variable2, [, lookupTable[, distributions]])
            ClassifierByLookupTable3(classVar, variable1, variable2, variable3, [, lookupTable[, distributions]])
    
    Class-specific constructors that you can call instead of the general constructor. The number of attributes must match the constructor called.

.. method:: getindex(example)
    
    Returns an index into lookupTable or distributions. The formula
    depends upon the type of the classifier. If value *i* is
    int(example[variable*i*]), then the corresponding formulae are

    ClassifierByLookupTable1:
        index = value1, or len(lookupTable)-1 if value is unknown
    ClassifierByLookupTable2:
        index = value1*noOfValues1 + value2, or -1 if any value is unknown 
    ClassifierByLookupTable3:
        index = (value1*noOfValues1 + value2) * noOfValues2 + value3, or -1 if any value is unknown

    Let's see some indices for randomly chosen examples from the original table.
    
    part of `lookup-lookup.py`_ (uses: `monks-1.tab`_):

    .. literalinclude:: code/lookup-lookup.py
        :lines: 26-29
    
    Output::
    
        ['1', '1', '2', '2', '4', '1', '1']: ab 0, e1 3
        ['3', '3', '1', '2', '2', '1', '1']: ab 8, e1 1
        ['2', '1', '2', '3', '4', '2', '0']: ab 3, e1 3
        ['2', '1', '1', '2', '1', '1', '1']: ab 3, e1 0
        ['1', '1', '1', '2', '3', '1', '1']: ab 0, e1 2 


==========================
Classifier by ExampleTable
==========================

:obj:`ClassifierByExampleTable` is the alternative to
:obj:`ClassifierByLookupTable`. It is to be used when the
classification is based on more than three features. Instead of having
a lookup table, it stores an :obj:`Orange.data.Table`, which is
optimized for a faster access.

This class is used in similar contexts as
:obj:`ClassifierByLookupTable`. If you write, for instance, a
constructive induction algorithm, it is recommendable that the values
of the new feature are computed either by one of classifiers by lookup
table or by :obj:`ClassifierByExampleTable`, depending on the number
of bound features.

**Attributes:**

.. attribute:: sortedExamples
    
    A :obj:`Orange.data.Table` with sorted instances for lookup.
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
    :obj:`ClassifierByExampleTable` appears more similar to
    :obj:`ClassifierByLookupTable`. If a constructive induction
    algorithm returns the result in one of these classifiers, and you
    would like to check which features are used, you can use variables
    regardless of the class you actually got.

There are no specific methods for :obj:`ClassifierByExampleTable`.
Since this is a classifier, it can be called. When the instance to be
classified includes unknown values, :obj:`classifierForUnknown` will be
used if it is defined.

Although :obj:`ClassifierByExampleTable` is not really a classifier in
the sense that you will use it to classify instances, but is rather a
function for computation of intermediate values, it has an associated
learner, :obj:`LookupLearner`. The learner's task is, basically, to
construct a Table for :obj:`sortedExamples`. It sorts them, merges them
and, of course, regards instance weights in the process as well.

part of `lookup-table.py`_ (uses: `monks-1.tab`_):

.. literalinclude:: code/lookup-table.py
    :lines: 7-13


In data_s, we have prepared a table in which instances are described
only by a, b, e and the class. Learner constructs a
ClassifierByExampleTable and stores instances from data_s into its
sortedExamples. Instances are merged so that there are no duplicates.

    >>> print len(data_s)
    432
    >>> print len(abe2.sortedExamples)
    36
    >>> for i in abe2.sortedExamples[:5]:
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

    >>> for i in abe2.sortedExamples[:10]:
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

ClassifierByExampleTable will usually be used by getValueFrom. So, we
would probably continue this by constructing a new feature and put the
classifier into its getValueFrom.

    >>> y2 = orange.EnumVariable("y2", values = ["0", "1"])
    >>> y2.getValueFrom = abe

There's something disturbing here. Although abe determines the value of
y2, abe.classVar is still y. Orange doesn't bother (the whole example
is artificial - you will seldom pack the entire dataset in an
ClassifierByExampleTable...), so shouldn't you. But still, for the sake
of hygiene, you can conclude by

    >>> abe.classVar = y2

The whole story can be greatly simplified. LookupLearner can also be
called differently than other learners. Besides instances, you can pass
the new class variable and the features that should be used for
classification. This saves us from constructing data_s and reassigning
the classVar. It doesn't set the getValueFrom, though.

part of `lookup-table.py`_ (uses: `monks-1.tab`_)::

    import Orange

    table = Orange.data.Table("monks-1")
    a, b, e = table.domain["a"], table.domain["b"], table.domain["e"]

    y2 = Orange.data.feature.Discrete("y2", values = ["0", "1"])
    abe2 = Orange.classification.lookup.LookupLearner(y2, [a, b, e], table)

Let us, for the end, show another use of LookupLearner. With the
alternative call arguments, it offers an easy way to observe feature
interactions. For this purpose, we shall omit e, and construct a
ClassifierByExampleTable from a and b only.

part of `lookup-table.py`_ (uses: `monks-1.tab`_):

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


.. _lookup-lookup.py: code/lookup-lookup.py
.. _lookup-table.py: code/lookup-table.py
.. _monks-1.tab: code/monks-1.tab

"""

import Orange.data
from Orange.core import \
        LookupLearner, \
         ClassifierByLookupTable, \
              ClassifierByLookupTable1, \
              ClassifierByLookupTable2, \
              ClassifierByLookupTable3, \
              ClassifierByExampleTable as ClassifierByDataTable


import orngMisc


def lookupFromBound(attribute, bound):
    if not len(bound):
        raise TypeError, "no bound attributes"
    elif len(bound) <= 3:
        return apply([ClassifierByLookupTable, ClassifierByLookupTable2,
                      ClassifierByLookupTable3][len(bound) - 1],
                     [attribute] + list(bound))
    else:
        return None

    
def lookupFromFunction(attribute, bound, function):
    """Constructs ClassifierByExampleTable or ClassifierByLookupTable
    mirroring the given function
    
    """
    lookup = lookupFromBound(attribute, bound)
    if lookup:
        lookup.lookupTable = [Orange.data.Value(attribute, function(attributes))
                              for attributes in orngMisc.LimitedCounter(
                                  [len(attr.values) for attr in bound])]
        return lookup
    else:
        examples = Orange.data.Table(Orange.data.Domain(bound, attribute))
        for attributes in orngMisc.LimitedCounter([len(attr.values)
                                                   for attr in dom.attributes]):
            examples.append(Orange.data.Example(dom, attributes +
                                                [function(attributes)]))
        return LookupLearner(examples)
      

def lookupFromExamples(examples, weight = 0, learnerForUnknown = None):
    if len(examples.domain.attributes) <= 3:
        lookup = lookupFromBound(examples.domain.classVar,
                                 examples.domain.attributes)
        lookupTable = lookup.lookupTable
        for example in examples:
            ind = lookup.getindex(example)
            if not lookupTable[ind].isSpecial() and (lookupTable[ind] <>
                                                     example.getclass()):
                break
            lookupTable[ind] = example.getclass()
        else:
            return lookup

        # there are ambiguities; a backup plan is
        # ClassifierByExampleTable, let it deal with them
        return LookupLearner(examples, weight,
                             learnerForUnknown=learnerForUnknown)

    else:
        return LookupLearner(examples, weight,
                             learnerForUnknown=learnerForUnknown)
        
        
def printLookupFunction(func):
    if isinstance(func, Orange.data.feature.Feature):
        if not func.getValueFrom:
            raise TypeError, "attribute '%s' does not have an associated function" % func.name
        else:
            func = func.getValueFrom

    outp = ""
    if isinstance(func, ClassifierByExampleTable):
    # XXX This needs some polishing :-)
        for i in func.sortedExamples:
            outp += "%s\n" % i
    else:
        boundset = func.boundset()
        for a in boundset:
            outp += "%s\t" % a.name
        outp += "%s\n" % func.classVar.name
        outp += "------\t" * (len(boundset)+1) + "\n"
        
        lc = 0
        if len(boundset)==1:
            cnt = orngMisc.LimitedCounter([len(x.values)+1 for x in boundset])
        else:
            cnt = orngMisc.LimitedCounter([len(x.values) for x in boundset])
        for ex in cnt:
            for i in range(len(ex)):
                if ex[i]<len(boundset[i].values):
                    outp += "%s\t" % boundset[i].values[ex[i]]
                else:
                    outp += "?\t",
            outp += "%s\n" % func.classVar.values[int(func.lookupTable[lc])]
            lc += 1
    return outp
