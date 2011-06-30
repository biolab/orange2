# Description: Shows how to find the nearest neighbours of the given example
# Category:    basic classes, distances
# Classes:     FindNearest, FindNearestConstructor, FindNearest_BruteForce, FindNearestConstructor_BruteForce
# Uses:        lenses
# Referenced:  FindNearest.htm


import orange

data = orange.ExampleTable("lenses")

nnc = orange.FindNearestConstructor_BruteForce()
nnc.distanceConstructor = orange.ExamplesDistanceConstructor_Euclidean()

did = -42
# Note that this is wrong: id should be assigned by
# did = orange.newmetaid()
# We only do this so that the script gives the same output each time it's run
nn = nnc(data, 0, did)

print "*** Reference example: ", data[0]
for ex in nn(data[0], 5):
    print ex
