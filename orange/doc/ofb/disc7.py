# Description: Discretize the test set based on discretization of the training set.
# Category:    preprocessing
# Uses:        iris
# Classes:     Preprocessor_discretize, EntropyDiscretization
# Referenced:  o_categorization.htm

import orange
data = orange.ExampleTable("iris")

#split the data to learn and test set
ind = orange.MakeRandomIndices2(data, p0=6)
learn = data.select(ind, 0)
test = data.select(ind, 1)

# discretize learning set, then use its new domain
# to discretize the test set
learnD = orange.Preprocessor_discretize(data, method=orange.EntropyDiscretization())
testD = orange.ExampleTable(learnD.domain, test)

print "Test set, original:"
for i in range(3):
    print test[i]

print "Test set, discretized:"
for i in range(3):
    print testD[i]
