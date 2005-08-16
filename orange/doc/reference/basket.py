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
