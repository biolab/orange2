# Description: Shows how to use outlier detection
# Category:    Outlier detection
# Classes:     orngOutlier
# Uses:        bridges
# Referenced:  OutlierDetection.htm

import orange, orngOutlier

data = orange.ExampleTable("bridges")
outlierDet = orngOutlier.OutlierDetection()
outlierDet.setExamples(data, orange.ExamplesDistanceConstructor_Euclidean(data))
outlierDet.setKNN(3)
zValues = outlierDet.zValues()
sorted = []
for el in zValues: sorted.append(el)
sorted.sort()
for i, el in enumerate(zValues):
	if el > sorted[-6]: print  data[i], "Z-score: %5.3f" % el
