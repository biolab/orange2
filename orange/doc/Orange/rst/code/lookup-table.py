# Description: Shows how to construct an orange.ClassifierFromExampleTable
# Category:    classification, lookup classifiers, constructive induction, feature construction
# Classes:     ClassifierByExampleTable, LookupLearner
# Uses:        monk1
# Referenced:  lookup.htm

import Orange

table = Orange.data.Table("monks-1")
a, b, e = table.domain["a"], table.domain["b"], table.domain["e"]

table_s = table.select([a, b, e, table.domain.classVar])
abe = Orange.classification.lookup.LookupLearner(table_s)

print len(table_s)
print len(abe.sortedExamples)

for i in abe.sortedExamples[:10]:
    print i
print

for i in abe.sortedExamples[:10]:
    print i, i.getclass().svalue
print

y2 = Orange.data.feature.Discrete("y2", values = ["0", "1"])
abe2 = Orange.classification.lookup.LookupLearner(y2, [a, b, e], table)
for i in abe2.sortedExamples[:10]:
    print i, i.getclass().svalue
print

y2 = Orange.data.feature.Discrete("y2", values = ["0", "1"])
abe2 = Orange.classification.lookup.LookupLearner(y2, [a, b], table)
for i in abe2.sortedExamples:
    print i, i.getclass().svalue
