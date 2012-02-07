# Description: Demonstrates the use of logistic regression
# Category:    classification, logistic regression
# Classes:     LogRegLearner, StepWiseFSS_Filter
# Uses:        ionosphere.tab

import orange
import orngFSS, orngTest, orngStat, orngLR

data = orange.ExampleTable("../datasets/ionosphere.tab")

lr = orngLR.LogRegLearner(removeSingular=1)  

learners = [orngLR.LogRegLearner(name='logistic', removeSingular=1),
            orngFSS.FilteredLearner(lr, filter=orngLR.StepWiseFSS_Filter(addCrit=0.05, deleteCrit=0.9),
                                    name='filtered')
            ]

results = orngTest.crossValidation(learners, data, storeClassifiers=1)

print "Learner      CA"
for i in range(len(learners)):
    print "%-12s %5.3f" % (learners[i].name, orngStat.CA(results)[i]) 


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