# Description: Read data, list attributes and class values, print first few data instances
# Category:    description
# Uses:        lenses.tab
# Classes:     ExampleTable
# Referenced:  load_data.htm

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
