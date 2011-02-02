"""

.. index:: lookup

Lookup classifiers predict classes by looking into stored lists of
cases. There are two kinds of such classifiers in Orange. The simpler
and fastest ClassifierByLookupTable use up to three discrete attributes
and have a stored mapping from values of those attributes to class
value. The more complex classifiers stores an ExampleTable and predicts
the class by matching the example to examples in the table.

The natural habitat of these classifiers is feature construction:
they usually reside in getValueFrom fields of constructed attributes
to facilitate their automatic computation. For instance, the following
script shows how to translate the Monk 1 dataset features into a more
useful subset that will only include the attributes a, b, e, and
attributes that will tell whether a and b are equal and whether e is 1
(don't bother the details, they follow later).

`classifierByLookupTable.py`_ (uses: `monks-1.tab`_):

.. literalinclude:: code/classifierByLookupTable.py

We can check the correctness of the script by printing out several
random examples from data2.


.. _classifierByLookupTable.py: code/classifierByLookupTable.py
.. _monks-1.tab: code/monks-1.tab

"""

import Orange.data
import Orange.feature
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
    if isinstance(func, Orange.feature.Feature):
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
