import orange, orngTree, orngWrap, orngStat

learner = orngTree.TreeLearner()
data = orange.ExampleTable("voting")
tuner = orngWrap.Tune1Parameter(object=learner,
                                parameter="minSubset",
                                values=[1, 2, 3, 4, 5, 10, 15, 20],
                                evaluate = orngStat.AUC, verbose=2)
classifier = tuner(data)

print "Optimal setting: ", learner.minSubset

import orngTest
untuned = orngTree.TreeLearner()
res = orngTest.crossValidation([untuned, tuner], data)
AUCs = orngStat.AUC(res)

print "Untuned tree: %5.3f" % AUCs[0]
print "Tuned tree: %5.3f" % AUCs[1]


learner = orngTree.TreeLearner(minSubset=10).instance()
data = orange.ExampleTable("voting")
tuner = orngWrap.Tune1Parameter(object=learner,
                                parameter=["split.continuousSplitConstructor.minSubset", "split.discreteSplitConstructor.minSubset"],
                                values=[1, 2, 3, 4, 5, 10, 15, 20],
                                evaluate = orngStat.AUC, verbose=2)
classifier = tuner(data)

print "Optimal setting: ", learner.split.continuousSplitConstructor.minSubset
