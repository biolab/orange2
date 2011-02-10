# Description: Shows how to use Filter_sameValues qith options for conjunction and disjunction of conditions
# Category:    preprocessing
# Uses:        imports-85
# Classes:     Preprocessor_take
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

filter = orange.Preprocessor_take()
filter.values = {data.domain["sex"]: "Male", data.domain["education"]: "Masters"}

filter.conjunction = 1
data1 = filter(data)
report_prob('data1 (conjunction)', data1)

filter.conjunction = 0
data1 = filter(data)
report_prob('data1 (disjunction)', data1)

data2 = data.select(sex='Male', education='Masters')
report_prob('data2 (select, conjuction)', data2)
