# Description: Shows how to construct and use classes for measuring distances between examples
# Category:    distances
# Classes:     ExamplesDistanceConstructor, ExamplesDistance
# Uses:        lenses
# Referenced:  ExamplesDistance.htm


import orange

data = orange.ExampleTable("iris")
distance = orange.ExamplesDistanceConstructor_Euclidean(data)

ref = data[0]
refm = orange.Example(ref)
refm[0] = "?"

print "d(%s, %s) = %5.3f" % (ref, ref, distance(ref, ref))
print "d(%s, %s)   = %5.3f" % (ref, refm, distance(ref, refm))
print "d(%s,   %s)   = %5.3f" % (refm, refm, distance(refm, refm))
print

ref = data[50]
refm = orange.Example(ref)
refm[0] = "?"
print "d(%s, %s) = %5.3f" % (ref, ref, distance(ref, ref))
print "d(%s, %s)   = %5.3f" % (ref, refm, distance(ref, refm))
print "d(%s,   %s)   = %5.3f" % (refm, refm, distance(refm, refm))
print

data = orange.ExampleTable("lenses")
distance = orange.ExamplesDistanceConstructor_Euclidean(data)

ref = data[0]
refm = orange.Example(ref)
refm[0] = "?"

print "d(%s, %s) = %5.3f" % (ref, ref, distance(ref, ref))
print "d(%s, %s)   = %5.3f" % (ref, refm, distance(ref, refm))
print "d(%s,   %s)   = %5.3f" % (refm, refm, distance(refm, refm))
print
