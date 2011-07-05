# Description: Demonstrates the use of random forests from Orange.ensemble.forest module
# Category:    classification, ensembles
# Classes:     RandomForestLearner
# Uses:        bupa.tab
# Referenced:  orngEnsemble.htm

import Orange

forest = Orange.ensemble.forest.RandomForestLearner(trees=50, name="forest")
tree = Orange.classification.tree.TreeLearner(minExamples=2, mForPrunning=2, \
                            sameMajorityPruning=True, name='tree')
learners = [tree, forest]

print "Classification: bupa.tab"
table = Orange.data.Table("bupa.tab")
results = Orange.evaluation.testing.cross_validation(learners, table, folds=3)
print "Learner  CA     Brier  AUC"
for i in range(len(learners)):
    print "%-8s %5.3f  %5.3f  %5.3f" % (learners[i].name, \
        Orange.evaluation.scoring.CA(results)[i], 
        Orange.evaluation.scoring.Brier_score(results)[i],
        Orange.evaluation.scoring.AUC(results)[i])

print "Regression: housing.tab"
table = Orange.data.Table("housing.tab")
results = Orange.evaluation.testing.cross_validation(learners, table, folds=3)
print "Learner  MSE    RSE    R2"
for i in range(len(learners)):
    print "%-8s %5.3f  %5.3f  %5.3f" % (learners[i].name, \
        Orange.evaluation.scoring.MSE(results)[i],
        Orange.evaluation.scoring.RSE(results)[i],
        Orange.evaluation.scoring.R2(results)[i],)
