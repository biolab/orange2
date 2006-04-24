import orngSOM
import orange
l=orngSOM.SOMLearner(xDim=5, yDim=10, parameters=[{"iterations":1000, "radius":5, "alpha":0.02},{"iterations":10000, "alpha":0.05}]) #radius in the second phase will be the same as in the first
c=l(orange.ExampleTable("iris.tab"))
for n in c.nodes:
    print "node:", n.x,n.y
    for e in n.examples:
	print "\t",e
