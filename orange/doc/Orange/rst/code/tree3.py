# Author:      J Zabkar
# Version:     1.1
# Description: Grow classification tree with a self-defined stopping criteria
# Category:    modelling
# Uses:        iris.tab
# Referenced:  TODO

import Orange
from random import randint, seed
seed(0)

data = Orange.data.Table("iris.tab")

print "SOME RANDOMNESS IN STOPING:"
defStop = Orange.classification.tree.StopCriteria()
f = lambda examples, weightID, contingency: defStop(examples, weightID, contingency) or randint(1, 5) == 1
l = Orange.classification.tree.TreeLearner(data, stop=f)
Orange.classification.tree.printTxt(l)

print "\nRANDOM STOPING:"
f = lambda x,y,z: randint(1, 5)==1
l = Orange.classification.tree.TreeLearner(data, stop=f)
Orange.classification.tree.printTxt(l)
