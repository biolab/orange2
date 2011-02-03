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

part of `classifierByLookupTable.py`_ (uses: `monks-1.tab`_)::

    import Orange
    
    data = Orange.data.Table("monks-1")
    
    a, b, e = data.domain["a"], data.domain["b"], data.domain["e"]
    
    ab = Orange.data.feature.Discrete("a==b", values = ["no", "yes"])
    ab.getValueFrom = Orange.classification.lookup.ClassifierByLookupTable(ab, a, b,
                        ["yes", "no", "no",  "no", "yes", "no",  "no", "no", "yes"])
    
    e1 = Orange.data.feature.Discrete("e==1", values = ["no", "yes"])
    e1.getValueFrom = Orange.classification.lookup.ClassifierByLookupTable(e1, e,
                        ["yes", "no", "no", "no", "?"])
    
    data2 = data.select([a, b, ab, e, e1, data.domain.classVar])
    
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
    
    The attribute(s) that the classifier uses for classification. ClassifierByLookupTable1 only has variable1, ClassifierByLookupTable2 also has variable2 and ClassifierByLookupTable3 has all three.

.. attribute:: variables (read only)
    
    The above variables, returned as a tuple.

.. attribute:: noOfValues1, noOfValues2[, noOfValues3] (read only)
    
    The number of values for variable1, variable2 and variable3. This is stored here to make the classifier faster. Those attributes are defined only for ClassifierByLookupTable2 (the first two) and ClassifierByLookupTable3 (all three).

.. attribute:: lookupTable (read only)
    
    A list of values (ValueList), one for each possible combination of attributes. For ClassifierByLookupTable1, there is an additional element that is returned when the attribute's value is unknown. Values are ordered by values of attributes, with variable1 being the most important. In case of two three valued attributes, the list order is therefore 1-1, 1-2, 1-3, 2-1, 2-2, 2-3, 3-1, 3-2, 3-3, where the first digit corresponds to variable1 and the second to variable2.
    
    The list is read-only in the sense that you cannot assign a new list to this field. You can, however, change its elements. Don't change its size, though. 

.. attribute:: distributions (read only)
    
    Similar to lookupTable, but is of type DistributionList and stores a distribution for each combination of values. 

.. attribute:: dataDescription
    
    An object of type EFMDataDescription, defined only for ClassifierByLookupTable2 and ClassifierByLookupTable3. They use it to make predictions when one or more attribute values are unknown. ClassifierByLookupTable1 doesn't need it since this case is covered by an additional element in lookupTable and distributions, as told above. 

**Methods:**

.. method:: ClassifierByLookupTable(classVar, variable1[, variable2[, variable3]] [, lookupTable[, distributions]])

    A general constructor that, based on the number of attribute descriptors, constructs one of the three classes discussed. If lookupTable and distributions are omitted, constructor also initializes lookupTable and distributions to two lists of the right sizes, but their elements are don't knows and empty distributions. If they are given, they must be of correct size.


.. _classifierByLookupTable.py: code/classifierByLookupTable.py
.. _monks-1.tab: code/monks-1.tab

"""

import Orange.data
from Orange.core import \
        LookupLearner, \
         ClassifierByLookupTable, \
              ClassifierByLookupTable1, \
              ClassifierByLookupTable2, \
              ClassifierByLookupTable3, \
              ClassifierByExampleTable


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
