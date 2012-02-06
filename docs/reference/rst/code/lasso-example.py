# Description: Lasso regression
# Category:    regression
# Uses:        housing
# Referenced:  Orange.regression.lasso
# Classes:     Orange.regression.lasso.LassoRegressionLearner Orange.regression.lasso.LassoRegression

import Orange

housing = Orange.data.Table("housing")
learner = Orange.regression.lasso.LassoRegressionLearner()
classifier = learner(housing)

# prediction for five data instances and comparison to actual values
for ins in housing[:5]:
    print "Actual: %3.2f, predicted: %3.2f " % (ins.get_class(), classifier(ins))

print classifier

