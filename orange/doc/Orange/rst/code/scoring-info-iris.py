# Description: Shows how to assess the quality of attributes not in the dataset
# Category:    attribute quality
# Uses:        iris
# Referenced:  Orange.feature.html#scoring
# Classes:     Orange.feature.scoring.EntropyDiscretization, Orange.feature.scoring.Measure, Orange.feature.scoring.InfoGain

import Orange
table = Orange.data.Table("iris")

d1 = Orange.feature.discretization.EntropyDiscretization("petal length", table)
print Orange.feature.scoring.Relief(d1, table)

meas = Orange.feature.scoring.Relief()
for t in meas.thresholdFunction("petal length", table):
    print "%5.3f: %5.3f" % t

thresh, score, distr = meas.bestThreshold("petal length", table)
print "\nBest threshold: %5.3f (score %5.3f)" % (thresh, score)