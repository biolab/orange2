# Description: Bagging and boosting with k-nearest neighbors
# Category:    modelling
# Uses:        promoters.tab
# Classes:     orngTest.crossValidation, orngEnsemble.BaggedLearner, orngEnsemble.BoostedLearner
# Referenced:  o_ensemble.htm

import orange, orngTest, orngStat, orngEnsemble
data = orange.ExampleTable("promoters")

majority = orange.MajorityLearner()
majority.name = "default"
knn = orange.kNNLearner(k=11)
knn.name = "k-NN (k=11)"

bagged_knn = orngEnsemble.BaggedLearner(knn, t=10)
bagged_knn.name = "bagged k-NN"
boosted_knn = orngEnsemble.BoostedLearner(knn, t=10)
boosted_knn.name = "boosted k-NN"

learners = [majority, knn, bagged_knn, boosted_knn]
results = orngTest.crossValidation(learners, data, folds=10)
print "        Learner   CA     Brier Score"
for i in range(len(learners)):
    print ("%15s:  %5.3f  %5.3f") % (learners[i].name,
        orngStat.CA(results)[i], orngStat.BrierScore(results)[i])

