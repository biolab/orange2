# Author:      B Zupan
# Version:     1.0
# Description: Ranking of attributes with two different measures (Relief and gain ratio)
# Category:    preprocessing
# Uses:        voting.tab
# Referenced:  orngFSS.htm

import orange, orngFSS
data = orange.ExampleTable("voting")

print 'Relief GainRt Attribute'
ma_def = orngFSS.attMeasure(data)
gainRatio = orange.MeasureAttribute_gainRatio()
ma_gr  = orngFSS.attMeasure(data, gainRatio)
for i in range(5):
  print "%5.3f  %5.3f  %s" % (ma_def[i][1], ma_gr[i][1], ma_def[i][0])
