# Description: Script that tests bayes.py and compares it to kNN from core Orange
# Category:    modelling
# Uses:        voting.tab
# Classes:     orngTest.crossValidation
# Referenced:  c_nb.htm

import orange, orngTest, orngStat, bayes
data = orange.ExampleTable("voting")

bayes = bayes.Learner(m=2, name='my bayes')
knn = orange.kNNLearner(k=10)
knn.name = "knn"

learners = [knn,bayes]
results = orngTest.crossValidation(learners, data)
for i in range(len(learners)):
    print learners[i].name, "%.3f" % orngStat.CA(results)[i]
