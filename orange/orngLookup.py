import orange
from Orange.classification.lookup import \
     lookup_from_bound as lookupFromBound, \
     lookup_from_function as lookupFromFunction, \
     lookup_from_data as lookupFromExamples, \
     dump_lookup_function as printLookupFunction


for tpe in [orange.ClassifierByLookupTable, orange.ClassifierByLookupTable2, orange.ClassifierByLookupTable3, orange.ClassifierByExampleTable]:
    orange.setoutput(tpe, "tab", printLookupFunction)
