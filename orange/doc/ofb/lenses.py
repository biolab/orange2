# Author:      B Zupan
# Version:     1.0
# Description: Read data, list attributes and class values, print first few data instances
# Category:    description
# Uses:        lenses.tab

import orange
data = orange.ExampleTable("lenses")
print "Attributes:",
for i in data.domain.attributes:
    print i.name,
print
print "Class:", data.domain.classVar.name

print "First 5 data items:"
for i in range(5):
	print data[i]
