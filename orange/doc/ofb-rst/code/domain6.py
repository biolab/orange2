# Description: Shows how to use Filter_sameValues for instance selection
# Category:    preprocessing
# Uses:        imports-85
# Classes:     
# Referenced:  domain.htm

import orange

def report_prob(header, data):
  print 'Size of %s: %i instances; ' % (header, len(data)), 
  n = 0
  for i in data:
    if int(i.getclass())==0:
      n = n + 1
  if len(data):
    print "p(%s)=%5.3f" % (data.domain.classVar.values[0], float(n)/len(data))
  else:
    print

filename = "../../datasets/adult_sample.tab"
data = orange.ExampleTable(filename)
report_prob('data', data)

data1 = data.select(age=(30,40))
report_prob('data1, age from 30 to 40', data1)

data2 = data.select(age=(40,30))
report_prob('data2, younger than 30 or older than 40', data2)
