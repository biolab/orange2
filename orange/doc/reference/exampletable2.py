# Description: Shows how to use ExampleTable.select and ExampleTable.getitems for sampling
# Category:    basic classes, sampling
# Classes:     ExampleTable, MakeRandomIndices, MakeRandomIndicesCV
# Uses:        
# Referenced:  ExampleTable.htm

import orange

domain = orange.Domain([orange.FloatVariable()])
data = orange.ExampleTable(domain)
for i in range(10):
    data.append([i])

for d in data:
    print d,
print "\n"

cv_indices = orange.MakeRandomIndicesCV(data, 4)
print "Indices: ", cv_indices, "\n"

for fold in range(4):
    train = data.select(cv_indices, fold, negate = 1)
    test  = data.select(cv_indices, fold)
    print "Fold %d: train " % fold
    for ex in train:
        print "    ", ex
    print
    print "      : test  "
    for ex in test:
        print "    ", ex
    print

t = data.select([1, 1, 0, 0, 0,  0, 0, 0, 0, 1])
for ex in t:
    print ex

e = data.getitems([0, 1, 9])
for ex in e:
    print ex
