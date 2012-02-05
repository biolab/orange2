# Description: Linear regression
# Category:    regression
# Uses:        housing
# Referenced:  Orange.regression.linear
# Classes:     Orange.regression.linear.LinearRegressionLearner Orange.regression.linear.LinearRegression

import Orange

housing = Orange.data.Table("housing")
learner = Orange.regression.linear.LinearRegressionLearner()
classifier = learner(housing)

for ins in housing[:5]:
    print classifier(ins)

print classifier    