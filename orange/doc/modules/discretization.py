# Description: Demonstrates the use of discretization
# Category:    discretization
# Classes:     entropyDiscretization, DiscretizedLearner
# Uses:        iris.tab

import orange
import orngDisc

data = orange.ExampleTable("iris.tab")

disc_data = orngDisc.entropyDiscretization(data)

disc_learner = orngDisc.DiscretizedLearner(orange.BayesLearner(), name="disc-bayes")
learner = orange.BayesLearner(name="bayes")

learners = [learner, disc_learner]

import orngTest, orngStat

results = orngTest.crossValidation(learners, data)
print "Classification Accuracy:"
for i in range(len(learners)):
    print ("%15s: %5.3f") % (learners[i].name, orngStat.CA(results)[i])
