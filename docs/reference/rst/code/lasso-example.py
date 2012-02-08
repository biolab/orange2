# Description: Lasso regression
# Category:    regression

import Orange
import numpy

numpy.random.seed(0)

housing = Orange.data.Table("housing")
learner = Orange.regression.lasso.LassoRegressionLearner()
classifier = learner(housing)

# prediction for five data instances and comparison to actual values
for ins in housing[:5]:
    print "Actual: %3.2f, predicted: %3.2f " % (ins.get_class(), classifier(ins))

print classifier

