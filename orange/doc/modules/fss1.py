# Author:      B Zupan
# Version:     1.0
# Description: Ranking and selection of best N attributes
# Category:    preprocessing
# Uses:        voting.tab

import orange, orngFSS
data = orange.ExampleTable("voting")

print 'Relevance estimate for first three attributes:'
ma = orngFSS.attMeasure(data)
for m in ma[:3]:
  print "%5.3f %s" % (m[1], m[0])

n = 3
best = orngFSS.bestNAtts(ma, n)
print '\nBest %d attributes:' % n
for s in best:
  print s
