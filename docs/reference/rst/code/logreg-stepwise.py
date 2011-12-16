import Orange

ionosphere = Orange.data.Table("ionosphere.tab")

lr = Orange.classification.logreg.LogRegLearner(removeSingular=1)
learners = (
  Orange.classification.logreg.LogRegLearner(name='logistic', removeSingular=1),
  Orange.feature.selection.FilteredLearner(lr,
     filter=Orange.classification.logreg.StepWiseFSSFilter(addCrit=0.05, deleteCrit=0.9),
     name='filtered')
)
results = Orange.evaluation.testing.cross_validation(learners, ionosphere, store_classifiers=1)

# output the results
print "Learner      CA"
for i in range(len(learners)):
    print "%-12s %5.3f" % (learners[i].name, Orange.evaluation.scoring.CA(results)[i])

# find out which features were retained by filtering

print "\nNumber of times features were used in cross-validation:"
featuresUsed = {}
for i in range(10):
    for a in results.classifiers[i][1].atts():
        if a.name in featuresUsed.keys():
            featuresUsed[a.name] += 1
        else:
            featuresUsed[a.name] = 1
for k in featuresUsed:
    print "%2d x %s" % (featuresUsed[k], k)
