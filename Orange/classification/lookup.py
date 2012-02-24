from Orange.misc import deprecated_keywords
import Orange.data
from Orange.core import \
        LookupLearner, \
         ClassifierByLookupTable, \
              ClassifierByLookupTable1, \
              ClassifierByLookupTable2, \
              ClassifierByLookupTable3, \
              ClassifierByExampleTable as ClassifierByDataTable


@deprecated_keywords({"attribute":"class_var"})
def lookup_from_bound(class_var, bound):
    if not len(bound):
        raise TypeError, "no bound attributes"
    elif len(bound) <= 3:
        return [ClassifierByLookupTable, ClassifierByLookupTable2,
                ClassifierByLookupTable3][len(bound) - 1](class_var, *list(bound))
    else:
        return None

    
@deprecated_keywords({"attribute":"class_var"})
def lookup_from_function(class_var, bound, function):
    """
    Construct ClassifierByDataTable or ClassifierByLookupTable
    mirroring the given function.
    
    """
    lookup = lookup_from_bound(class_var, bound)
    if lookup:
        for i, attrs in enumerate(Orange.misc.counters.LimitedCounter(
                    [len(var.values) for var in bound])):
            lookup.lookup_table[i] = Orange.data.Value(class_var, function(attrs))
        return lookup
    else:
        dom = Orange.data.Domain(bound, class_var)
        data = Orange.data.Table(dom)
        for attrs in Orange.misc.counters.LimitedCounter(
                    [len(var.values) for var in dom.features]):
            data.append(Orange.data.Example(dom, attrs + [function(attrs)]))
        return LookupLearner(data)
      

@deprecated_keywords({"learnerForUnknown":"learner_for_unknown"})
def lookup_from_data(examples, weight=0, learner_for_unknown=None):
    if len(examples.domain.features) <= 3:
        lookup = lookup_from_bound(examples.domain.class_var,
                                 examples.domain.features)
        lookup_table = lookup.lookup_table
        for example in examples:
            ind = lookup.get_index(example)
            if not lookup_table[ind].is_special() and (lookup_table[ind] !=
                                                     example.get_class()):
                break
            lookup_table[ind] = example.get_class()
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
