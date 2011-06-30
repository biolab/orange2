# Description: Demonstrates the use of linear regression
# Category:    regression
# Classes:     LinearRegressionLearner
# Uses:        housing.tab

import orange
import orngRegression

data = orange.ExampleTable("housing")

learners = [orngRegression.LinearRegressionLearner()]


import orngTest, orngStat

results = orngTest.crossValidation(learners, data)

print "Error:"
for i in range(len(learners)):
    print ("%15s: %5.3f") % (learners[i].name, orngStat.MSE(results)[i])
    
print

lr = orngRegression.LinearRegressionLearner(name="lr")

orngRegression.printLinearRegression(lr(data))
