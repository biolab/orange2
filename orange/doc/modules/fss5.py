# Description: Recursively eliminates attributes using Relief measure, until
#              the estimate relevants of all attributes is beyond certain threshold.
#              Makes use of filterRelieff from orngFSS
# Category:    preprocessing
# Uses:        voting.tab
# Referenced:  orngFSS.htm

import orange, orngFSS

def report_relevance(data):
  m = orngFSS.attMeasure(data)
  for i in m:
    print "%5.3f %s" % (i[1], i[0])

data = orange.ExampleTable("../datasets/adult_sample")
print "Before feature subset selection:"; report_relevance(data)

marg = 0.01
ndata = orngFSS.filterRelieff(data, margin=marg)
print "\nAfter feature subset selection with margin %5.3f:" % marg
report_relevance(ndata)
