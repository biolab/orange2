# Description: Shows how to use outlier detection
# Category:    Outlier detection
# Classes:     orngOutlier
# Uses:        bridges
# Referenced:  OutlierDetection.htm

import orange, orngOutlier

data = orange.ExampleTable("bridges")
outlierDet = orngOutlier.OutlierDetection()
outlierDet.setExamples(data)
print outlierDet.zValues()
