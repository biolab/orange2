# Description: Demonstrates the use of attribute evaluation
# Category:    feature scoring, FSS
# Classes:     MeasureAttribute_Distance, MeasureAttribute_MDL
# Uses:        zoo.tab

import orange
import orngEvalAttr
import orngCI
import orngFSS
data = orange.ExampleTable("../datasets/zoo")

print 'Distance(1-D)  MDL    Attribute'

distance = orngEvalAttr.MeasureAttribute_Distance()
ma_d  = orngFSS.attMeasure(data, distance)

mdl = orngEvalAttr.MeasureAttribute_MDL()
ma_mdl = orngFSS.attMeasure(data, mdl)
for i in range(5):
  print "%5.3f          %5.3f  %s" % (ma_d[i][1], ma_mdl[i][1], ma_d[i][0])
