# Author:      J Zabkar
# Version:     1.1
# Description: Grow classification tree with a self-defined stopping criteria
# Category:    modelling
# Uses:        iris.tab
# Referenced:  TODO

import Orange
from random import randint, seed
seed(0)

iris = Orange.data.Table("iris.tab")

print "SOME RANDOMNESS IN STOPING:"
def_stop = Orange.classification.tree.StopCriteria()
f = lambda examples, weightID, contingency: def_stop(examples, weightID, contingency) or randint(1, 5) == 1
l = Orange.classification.tree.TreeLearner(iris, stop=f)
print l

print "\nRANDOM STOPING:"
f = lambda x,y,z: randint(1, 5)==1
l = Orange.classification.tree.TreeLearner(iris, stop=f)
print l
