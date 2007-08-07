# Description: Demonstrates the use of different scoring techniques for regression
# Category:    modelling, evaluation
# Uses:        housing
# Classes:     orngTest.crossValidation, orngTree.TreeLearner, orange.kNNLearner, orngRegression.LinearRegressionLearner
# Referenced:  regression.htm

import orange
import orngRegression
import orngTree
import orngStat, orngTest

data = orange.ExampleTable("housing")

# definition of learners (regressors)
lr = orngRegression.LinearRegressionLearner(name="lr")
rt = orngTree.TreeLearner(measure="retis", mForPruning=2,
                          minExamples=20, name="rt")
maj = orange.MajorityLearner(name="maj")
knn = orange.kNNLearner(k=10, name="knn")
learners = [maj, lr, rt, knn]

# evaluation and reporting of scores
results = orngTest.crossValidation(learners, data, folds=10)
scores = [("MSE", orngStat.MSE),
          ("RMSE", orngStat.RMSE),
          ("MAE", orngStat.MAE),
          ("RSE", orngStat.RSE),
          ("RRSE", orngStat.RRSE),
          ("RAE", orngStat.RAE),
          ("R2", orngStat.R2)]

print "Learner  " + "".join(["%-7s" % s[0] for s in scores])
for i in range(len(learners)):
    print "%-8s " % learners[i].name + "".join(["%6.3f " % s[1](results)[i] for s in scores])
