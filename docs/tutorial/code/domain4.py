# Description: Shows how to select examples based on their attribute values
# Category:    preprocessing
# Uses:        imports-85
# Referenced:  domain.htm

import orange

def report_prob(header, data):
  print 'Size of %s: %i instances' % (header, len(data))
  n = 0
  for i in data:
    if int(i.getclass())==0:
      n = n + 1
  print "p(%s)=%5.3f" % (data.domain.classVar.values[0], float(n)/len(data))

filename = "../../datasets/adult_sample.tab"
data = orange.ExampleTable(filename)
report_prob('original data set', data)

data1 = data.select(sex='Male')
report_prob('data1', data1)

data2 = data.select(sex='Male', education='Masters')
report_prob('data2', data2)

