# Description: Lasso regression
# Category:    regression

import Orange
import numpy

numpy.random.seed(0)

housing = Orange.data.Table("housing")
learner = Orange.regression.lasso.LassoRegressionLearner(
    lasso_lambda=1, n_boot=100, n_perm=100)
classifier = learner(housing)

# prediction for five data instances
for ins in housing[:5]:
    print "Actual: %3.2f, predicted: %3.2f" % (
        ins.get_class(), classifier(ins))

print classifier
