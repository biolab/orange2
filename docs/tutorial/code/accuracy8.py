# Description: Demostration of use of cross-validation as provided in orngEval module
# Category:    evaluation
# Uses:        voting.tab
# Classes:     orngTest.crossValidation
# Referenced:  c_performance.htm

import orange
import orngTest, orngStat, orngTree

# set up the learners
bayes = orange.BayesLearner()
tree = orngTree.TreeLearner(mForPruning=2)
bayes.name = "bayes"
tree.name = "tree"
learners = [bayes, tree]

# compute accuracies on data
data = orange.ExampleTable("voting")
res = orngTest.crossValidation(learners, data, folds=10)
cm = orngStat.computeConfusionMatrices(res,
        classIndex=data.domain.classVar.values.index('democrat'))

stat = (('CA', lambda res,cm: orngStat.CA(res)),
        ('Sens', lambda res,cm: orngStat.sens(cm)),
        ('Spec', lambda res,cm: orngStat.spec(cm)),
        ('AUC', lambda res,cm: orngStat.AUC(res)),
        ('IS', lambda res,cm: orngStat.IS(res)),
        ('Brier', lambda res,cm: orngStat.BrierScore(res)),
        ('F1', lambda res,cm: orngStat.F1(cm)),
        ('F2', lambda res,cm: orngStat.Falpha(cm, alpha=2.0)),
        ('MCC', lambda res,cm: orngStat.MCC(cm)),
        ('sPi', lambda res,cm: orngStat.scottsPi(cm)),
        )

scores = [s[1](res,cm) for s in stat]
print
print "Learner  " + "".join(["%-7s" % s[0] for s in stat])
for (i, l) in enumerate(learners):
    print "%-8s " % l.name + "".join(["%5.3f  " % s[i] for s in scores])
