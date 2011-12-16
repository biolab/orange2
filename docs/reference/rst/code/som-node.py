# Description: Self Organizing Maps on iris data set
# Category:    projection
# Uses:        iris
# Referenced:  Orange.projection.som
# Classes:     Orange.projection.som.Node

import Orange
som = Orange.projection.som.SOMLearner(map_shape=(5, 5))
map = som(Orange.data.Table("iris.tab"))
node = map[3, 3]

print "Node position: (%d, %d)" % node.pos
print "Data instances in the node:", len(node.examples)
