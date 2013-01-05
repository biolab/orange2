# Author:      B Zupan
# Version:     1.0
# Description: Same as fss5.py but uses FilterRelieff class from orngFSS
# Category:    preprocessing
# Uses:        adult_saple.tab
# Referenced:  o_fss.htm

import orngFSS
import Orange
data = Orange.data.Table("adult_sample.tab")

def report_relevance(data):
  m = Orange.feature.scoring.score_all(data)
  for i in m:
    print "%5.3f %s" % (i[1], i[0])

print "Before feature subset selection (%d attributes):" % len(data.domain.attributes)
report_relevance(data)
data = Orange.data.Table("adult_sample.tab")

marg = 0.01
filter = Orange.feature.selection.FilterRelief(margin=marg)
ndata = filter(data)
print "\nAfter feature subset selection with margin %5.3f (%d attributes):" % (marg, len(ndata.domain.attributes))
report_relevance(ndata)
