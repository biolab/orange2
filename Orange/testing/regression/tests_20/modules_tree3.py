# Author:      J Zabkar
# Version:     1.0
# Description: Grow classification tree with a self-defined stopping criteria
# Category:    modelling
# Uses:        iris.tab
# Referenced:   orngTree.htm

import orange, orngTree
from random import randint, seed
seed(0)

data = orange.ExampleTable("iris.tab")

print "SOME RANDOMNESS IN STOPING:"
defStop = orange.TreeStopCriteria()
f = lambda examples, weightID, contingency: defStop(examples, weightID, contingency) or randint(1, 5)==1
l = orngTree.TreeLearner(data, stop=f)
orngTree.printTxt(l)

print "\nRANDOM STOPING:"
f = lambda x,y,z: randint(1, 5)==1
l = orngTree.TreeLearner(data, stop=f)
orngTree.printTxt(l)
