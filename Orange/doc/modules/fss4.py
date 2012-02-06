# Description: Demonstrates the use of orngFSS.FilteredLearner to compare
#              naive Bayesian learner when all or just the most important attribute
#              is used. Shows how to find out which (in ten-fold cross validation)
#              attributes was used the most.
# Category:    preprocessing
# Uses:        voting.tab
# Referenced:  orngFSS.htm
# Classes:     orngFSS.FilteredLearner

import orange, orngFSS, orngTest, orngStat
data = orange.ExampleTable("voting")

nb = orange.BayesLearner()
learners = (orange.BayesLearner(name='bayes'),
            orngFSS.FilteredLearner(nb, filter=orngFSS.FilterBestNAtts(n=1), name='filtered'))
results = orngTest.crossValidation(learners, data, storeClassifiers=1)

# output the results
print "Learner      CA"
for i in range(len(learners)):
  print "%-12s %5.3f" % (learners[i].name, orngStat.CA(results)[i])

# find out which attributes were retained by filtering

print "\nNumber of times attributes were used in cross-validation:"
attsUsed = {}
for i in range(10):
  for a in results.classifiers[i][1].atts():
    if a.name in attsUsed.keys(): attsUsed[a.name] += 1
    else: attsUsed[a.name] = 1
for k in attsUsed.keys():
  print "%2d x %s" % (attsUsed[k], k)
