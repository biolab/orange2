# Description: Reads a basket file
# Category:    file formats
# Classes:     ExampleTable
# Uses:        inquisition.basket, inquisition2.basket
# Referenced:  fileformats.htm

import orange

print "Sentences in 'inquisition'"
##data = orange.ExampleTable("inquisition")
##for ex in data:
##    print ex
##    print

data = orange.ExampleTable("inquisition2")
for ex in data:
    print ex

# This is to make the regression test independent of the system
# and the version of Python
def printSortedMetas(metas, nm=0):
    l = metas.items()
    if nm:
        l.sort(lambda x, y: cmp(x[0].name, y[0].name))
    else:
        l.sort(lambda x, y: cmp(x[0], y[0]))
    print l

example = data[4]
#printSortedMetas(example.getmetas())
#printSortedMetas(example.getmetas(orange.Variable), 1)
