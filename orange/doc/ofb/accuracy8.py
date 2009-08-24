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

stat = (('CA', 'CA(res)'),
        ('Sens', 'sens(cm)'),
        ('Spec', 'spec(cm)'),
        ('AUC', 'AUC(res)'),
        ('IS', 'IS(res)'),
        ('Brier', 'BrierScore(res)'),
        ('F1', 'F1(cm)'),
        ('F2', 'Falpha(cm, alpha=2.0)'),
        ('MCC', 'MCC(cm)'))

scores = [eval("orngStat."+s[1]) for s in stat]
print
print "Learner  " + "".join(["%-7s" % s[0] for s in stat])
for (i, l) in enumerate(learners):
    print "%-8s " % l.name + "".join(["%5.3f  " % s[i] for s in scores])
