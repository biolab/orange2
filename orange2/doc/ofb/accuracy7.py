# Description: Demostration of use of cross-validation as provided in orngEval module
# Category:    evaluation
# Uses:        voting.tab
# Classes:     orngTest.crossValidation
# Referenced:  c_performance.htm

import orange, orngTest, orngStat, orngTree

# set up the learners
bayes = orange.BayesLearner()
tree = orngTree.TreeLearner(mForPruning=2)
bayes.name = "bayes"
tree.name = "tree"
learners = [bayes, tree]

# compute accuracies on data
data = orange.ExampleTable("voting")
results = orngTest.crossValidation(learners, data, folds=10)

# output the results
print "Learner  CA     IS     Brier    AUC"
for i in range(len(learners)):
    print "%-8s %5.3f  %5.3f  %5.3f  %5.3f" % (learners[i].name, \
        orngStat.CA(results)[i], orngStat.IS(results)[i],
        orngStat.BrierScore(results)[i], orngStat.AUC(results)[i])
