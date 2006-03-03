# Description: Shows how to select examples based on their attribute values
# Category:    preprocessing
# Uses:        imports-85
# Classes:     Domain, select
# Referenced:  domain.htm

import orange

def reportAttributes(dataset, header=None):
  if dataset.domain.classVar:
    print 'Class variable: %s,' % dataset.domain.classVar.name,
  else:
    print 'No Class,',
  if header:
    print '%s:' % header
  for i in range(len(dataset.domain.attributes)):
    print "%s" % dataset.domain.attributes[i].name,
    if i % 6 == 5: print
  print "\n"

filename = "imports-85.tab"
data = orange.ExampleTable(filename)
reportAttributes(data, "Original data set")

newData1 = data.select(range(5))
reportAttributes(newData1, "First five attributes")

newData2 = data.select(['engine-location', 'wheel-base', 'length'])
reportAttributes(newData2, "Attributes selected by name")

domain3 = orange.Domain([data.domain[0], data.domain['curb-weight'], data.domain[2]])
newData3 = data.select(domain3)
reportAttributes(newData3, "Attributes by domain")

domain4 = orange.Domain([data.domain[0], data.domain['curb-weight'], data.domain[2]], 0)
newData4 = data.select(domain4)
reportAttributes(newData4, "Attributes by domain")
