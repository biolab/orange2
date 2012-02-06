# Description: Shows how to add class noise and missing attributes to data sets. Also shows how to test a single learner on a range of data sets.
# Category:    preprocessing
# Uses:        imports-85
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

filename = "../datasets/adult_sample.tab"
data = orange.ExampleTable(filename)
report_prob('data', data)

selection = [1]*10 + [0]*(len(data)-10)
data1 = data.select(selection)
report_prob('data1, first ten instances', data1)

data2 = data.select(selection, negate=1)
report_prob('data2, other than first ten instances', data2)

selection = [1]*12 + [2]*12 + [3]*12 + [0]*(len(data)-12*3)
data3 = data.select(selection, 3)
report_prob('data3, third dozen of instances', data3)
