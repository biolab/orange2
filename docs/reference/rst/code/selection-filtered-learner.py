# Description: Demonstrates the use of Orange.feature.selection.FilteredLearner
#              to compare naive Bayesian learner when all or just the most 
#              important attribute is used. Shows how to find out which (in
#              ten-fold cross validation) attributes was used the most.
# Category:    feature selection
# Uses:        voting
# Referenced:  Orange.feature.html#selection
# Classes:     Orange.feature.selection.FilteredLearner

import Orange

voting = Orange.data.Table("voting")

nb = Orange.classification.bayes.NaiveLearner()
fl = Orange.feature.selection.FilteredLearner(nb,
     filter=Orange.feature.selection.FilterBestN(n=1), name='filtered')
learners = (Orange.classification.bayes.NaiveLearner(name='bayes'), fl)
results = Orange.evaluation.testing.cross_validation(learners, voting, storeClassifiers=1)

# output the results
print "Learner      CA"
for i in range(len(learners)):
    print "%-12s %5.3f" % (learners[i].name, Orange.evaluation.scoring.CA(results)[i])

# find out which attributes were retained by filtering

print "\nNumber of times attributes were used in cross-validation:"
attsUsed = {}
for i in range(10):
    for a in results.classifiers[i][1].atts():
        if a.name in attsUsed.keys():
            attsUsed[a.name] += 1
        else:
            attsUsed[a.name] = 1
for k in attsUsed.keys():
    print "%2d x %s" % (attsUsed[k], k)
