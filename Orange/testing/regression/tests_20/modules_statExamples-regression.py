# Description: Demostrates the use of regression scores
# Category:    evaluation
# Uses:        housing.tab
# Referenced:  orngStat.htm

import orange
import orngRegression as r
import orngTree
import orngStat, orngTest

data = orange.ExampleTable("housing")

# definition of regressors
lr = r.LinearRegressionLearner(name="lr")
rt = orngTree.TreeLearner(measure="retis", mForPruning=2,
                          minExamples=20, name="rt")
maj = orange.MajorityLearner(name="maj")
knn = orange.kNNLearner(k=10, name="knn")

learners = [maj, rt, knn, lr]

# cross validation, selection of scores, report of results
results = orngTest.crossValidation(learners, data, folds=3)
scores = [("MSE", orngStat.MSE),   ("RMSE", orngStat.RMSE),
          ("MAE", orngStat.MAE),   ("RSE", orngStat.RSE),
          ("RRSE", orngStat.RRSE), ("RAE", orngStat.RAE),
          ("R2", orngStat.R2)]

print "Learner   " + "".join(["%-8s" % s[0] for s in scores])
for i in range(len(learners)):
    print "%-8s " % learners[i].name + \
    "".join(["%7.3f " % s[1](results)[i] for s in scores])
