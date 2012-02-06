import Orange
import time

class SimpleTreeLearnerSetProb():
    """
    Orange.classification.tree.SimpleTreeLearner which sets the skip_prob
    so that on average a square root of the attributes will be 
    randomly choosen for each split.
    """
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __call__(self, examples, weight=0):
        self.wrapped.skip_prob = 1-len(examples.domain.attributes)**0.5/len(examples.domain.attributes)
        return self.wrapped(examples)

#ordinary random forests
tree = Orange.classification.tree.TreeLearner(min_instances=5, measure="gainRatio")
rf_def = Orange.ensemble.forest.RandomForestLearner(trees=50, base_learner=tree, name="for_gain")

#random forests with simple trees - simple trees do random attribute selection by themselves
st = Orange.classification.tree.SimpleTreeLearner(min_instances=5)
stp = SimpleTreeLearnerSetProb(st)
rf_simple = Orange.ensemble.forest.RandomForestLearner(learner=stp, trees=50, name="for_simp")

learners = [ rf_def, rf_simple ]

iris = Orange.data.Table("iris")
results = Orange.evaluation.testing.cross_validation(learners, iris, folds=3)
print "Learner  CA     Brier  AUC"
for i in range(len(learners)):
    print "%-8s %5.3f  %5.3f  %5.3f" % (learners[i].name, \
    Orange.evaluation.scoring.CA(results)[i], 
    Orange.evaluation.scoring.Brier_score(results)[i],
    Orange.evaluation.scoring.AUC(results)[i])

print 

print "Runtimes:"
for l in learners:
    t = time.time()
    l(iris)
    print l.name, time.time() - t
