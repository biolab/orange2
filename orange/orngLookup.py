import orange, orngMisc

def lookupFromBound(attribute, bound):
    if not len(bound):
        raise TypeError, "no bound attributes"
    elif len(bound)<=3:
        return apply([orange.ClassifierByLookupTable, orange.ClassifierByLookupTable2, orange.ClassifierByLookupTable3][len(bound)-1], [attribute] + list(bound))
    else:
        return None

    
def lookupFromFunction(attribute, bound, function):
    """
    Constructs ClassifierByExampleTable or ClassifierByLookupTable mirroring the given function
    """
    lookup = lookupFromBound(attribute, bound)
    if lookup:
        lookup.lookupTable = [orange.Value(attribute, function(attributes)) for attributes in orngMisc.LimitedCounter([len(attr.values) for attr in bound])]
        return lookup
    else:
        examples = orange.ExampleTable(orange.Domain(bound, attribute))
        for attributes in orngMisc.LimitedCounter([len(attr.values) for attr in dom.attributes]):
            examples.append(orange.Example(dom, attributes + [function(attributes)]))
        return orange.LookupLearner(examples)
      

def lookupFromExamples(examples, weight = 0, learnerForUnknown = None):
    if len(examples.domain.attributes) <= 3:
        lookup = lookupFromBound(examples.domain.classVar, examples.domain.attributes)
        lookupTable = lookup.lookupTable
        for example in examples:
            ind = lookup.getindex(example)
            if not lookupTable[ind].isSpecial() and (lookupTable[ind] <> example.getclass()):
                break
            lookupTable[ind] = example.getclass()
        else:
            return lookup

        # there are ambiguities; a backup plan is ClassifierByExampleTable, let it deal with them
        return orange.LookupLearner(examples, weight, learnerForUnknown = learnerForUnknown)

    else:
        return orange.LookupLearner(examples, weight, learnerForUnknown = learnerForUnknown)
        
        
def printLookupFunction(func):
    if isinstance(func, orange.Variable):
        if not func.getValueFrom:
            raise TypeError, "attribute '%s' does not have an associated function" % func.name
        else:
            func = func.getValueFrom

    outp = ""
    if isinstance(func, orange.ClassifierByExampleTable):
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


for tpe in [orange.ClassifierByLookupTable, orange.ClassifierByLookupTable2, orange.ClassifierByLookupTable3, orange.ClassifierByExampleTable]:
    orange.setoutput(tpe, "tab", printLookupFunction)
