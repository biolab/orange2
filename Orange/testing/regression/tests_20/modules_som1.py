# Description: Self Organizing Maps on iris data set
# Category:    modelling
# Uses:        iris
# Referenced:  orngSOM.htm
# Classes:     orngSOM.SOMLearner

import orngSOM
import orange

import random
random.seed(42)

som = orngSOM.SOMLearner(map_shape=(10, 20), initialize=orngSOM.InitializeRandom)
map = som(orange.ExampleTable("iris.tab"))
for n in map:
    print "node:", n.pos[0], n.pos[1]
    for e in n.examples:
        print "\t",e
