# Description: Linear regression
# Category:    regression
# Uses:        housing
# Referenced:  Orange.regression.linear
# Classes:     Orange.regression.linear.LinearRegressionLearner Orange.regression.linear.LinearRegression

import Orange

housing = Orange.data.Table("housing")
learner = Orange.regression.linear.LinearRegressionLearner()
classifier = learner(housing)

# prediction for five data instances and comparison to actual values
for ins in housing[:5]:
    print "Actual: %3.2f, predicted: %3.2f " % (ins.get_class(), classifier(ins))

print classifier

# stepwise regression
learner2 = Orange.regression.linear.LinearRegressionLearner(stepwise=True,
                                                           add_sig=0.05,
                                                           remove_sig=0.2)
classifier = learner2(housing)

print classifier

