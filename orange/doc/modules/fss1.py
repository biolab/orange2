# Description: Ranking and selection of best N attributes
# Category:    preprocessing
# Uses:        voting
# Referenced:  orngFSS.htm
# Classes:     orngFSS.attMeasure, orngFSS.bestNAtts

import orange, orngFSS
data = orange.ExampleTable("voting")

print 'Attribute scores for best three attributes:'
ma = orngFSS.attMeasure(data)
for m in ma[:3]:
  print "%5.3f %s" % (m[1], m[0])

n = 3
best = orngFSS.bestNAtts(ma, n)
print '\nBest %d attributes:' % n
for s in best:
  print s
