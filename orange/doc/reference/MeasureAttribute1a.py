# Description: Shows how to assess the quality of attributes not in the dataset
# Category:    attribute quality
# Classes:     EntropyDiscretization, MeasureAttribute, MeasureAttribute_info
# Uses:        iris
# Referenced:  MeasureAttribute.htm

import orange
data = orange.ExampleTable("iris")

d1 = orange.EntropyDiscretization("petal length", data)
print orange.MeasureAttribute_relief(d1, data)

meas = orange.MeasureAttribute_relief()
for t in meas.thresholdFunction("petal length", data):
    print "%5.3f: %5.3f" % t

thresh, score, distr = meas.bestThreshold("petal length", data)
print "\nBest threshold: %5.3f (score %5.3f)" % (thresh, score)