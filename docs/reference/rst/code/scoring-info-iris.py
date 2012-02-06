# Description: Shows how to assess the quality of features not in the dataset
# Category:    feature scoring
# Uses:        iris
# Referenced:  Orange.feature.html#scoring
# Classes:     Orange.feature.discretization.EntropyDiscretization, Orange.feature.scoring.Measure, Orange.feature.scoring.InfoGain, Orange.feature.scoring.Relief

import Orange
iris = Orange.data.Table("iris")

d1 = Orange.feature.discretization.EntropyDiscretization("petal length", iris)
print Orange.feature.scoring.InfoGain(d1, iris)

iris = Orange.data.Table("iris")
meas = Orange.feature.scoring.Relief()
for t in meas.threshold_function("petal length", iris):
    print "%5.3f: %5.3f" % t

thresh, score, distr = meas.best_threshold("petal length", iris)
print "\nBest threshold: %5.3f (score %5.3f)" % (thresh, score)
