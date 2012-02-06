import Orange

learner = Orange.classification.tree.TreeLearner()
voting = Orange.data.Table("voting")
tuner = Orange.optimization.Tune1Parameter(object=learner,
                           parameter="minSubset",
                           values=[1, 2, 3, 4, 5, 10, 15, 20],
                           evaluate = Orange.evaluation.scoring.AUC, verbose=2)
classifier = tuner(voting)

print "Optimal setting: ", learner.minSubset

untuned = Orange.classification.tree.TreeLearner()
res = Orange.evaluation.testing.cross_validation([untuned, tuner], voting)
AUCs = Orange.evaluation.scoring.AUC(res)

print "Untuned tree: %5.3f" % AUCs[0]
print "Tuned tree: %5.3f" % AUCs[1]

learner = Orange.classification.tree.TreeLearner(minSubset=10).instance()
voting = Orange.data.Table("voting")
tuner = Orange.optimization.Tune1Parameter(object=learner,
                    parameter=["split.continuousSplitConstructor.minSubset", 
                               "split.discreteSplitConstructor.minSubset"],
                    values=[1, 2, 3, 4, 5, 10, 15, 20],
                    evaluate = Orange.evaluation.scoring.AUC, verbose=2)

classifier = tuner(voting)

print "Optimal setting: ", learner.split.continuousSplitConstructor.minSubset
