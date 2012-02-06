# Description: Reads a data set, prints out attribute and class names
# Category:    preprocessing
# Uses:        imports-85
# Referenced:  domain.htm

import orange

filename = "imports-85.tab"
data = orange.ExampleTable(filename)
print "%s includes %i attributes and a class variable %s" % \
  (filename, len(data.domain.attributes), data.domain.classVar.name)

print "Attribute names and indices:"
for i in range(len(data.domain.attributes)):
  print "(%2d) %-17s" % (i, data.domain.attributes[i].name),
  if i % 3 == 2: print
