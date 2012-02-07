# Description: Shows how to construct and use classes for measuring distances between examples
# Category:    distances
# Classes:     ExamplesDistanceConstructor, ExamplesDistance
# Uses:        lenses
# Referenced:  ExamplesDistance.htm


import orange

data = orange.ExampleTable("lenses")

distance = orange.ExamplesDistanceConstructor_Manhattan(data)

ref = data[0]
print "*** Reference example: ", ref
for ex in data:
    print ex, distance(ex, ref)