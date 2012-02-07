# Description: Shows how to use ExampleTable.select and ExampleTable.getitems for sampling
# Category:    basic classes, sampling
# Classes:     ExampleTable, MakeRandomIndices, MakeRandomIndicesCV
# Uses:        
# Referenced:  ExampleTable.htm

import Orange

domain = Orange.data.Domain([Orange.feature.Continuous()])
data = Orange.data.Table(domain)
for i in range(10):
    data.append([i])

cv_indices = Orange.core.MakeRandomIndicesCV(data, 4)
print "Indices: ", cv_indices, "\n"

for fold in range(4):
    train = data.select(cv_indices, fold, negate = 1)
    test  = data.select(cv_indices, fold)
    print "Fold %d: train " % fold
    for inst in train:
        print "    ", inst
    print
    print "      : test  "
    for inst in test:
        print "    ", inst
    print

t = data.select([1, 1, 0, 0, 0,  0, 0, 0, 0, 1])
for inst in t:
    print inst

e = data.get_items([0, 1, 9])
for inst in e:
    print inst
