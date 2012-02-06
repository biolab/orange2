import Orange

ionosphere = Orange.data.Table("ionosphere.tab")

lr = Orange.classification.logreg.LogRegLearner(remove_singular=1)
learners = (
  Orange.classification.logreg.LogRegLearner(name='logistic',
      remove_singular=1),
  Orange.feature.selection.FilteredLearner(lr,
     filter=Orange.classification.logreg.StepWiseFSSFilter(add_crit=0.05,
         delete_crit=0.9), name='filtered')
)
results = Orange.evaluation.testing.cross_validation(learners, ionosphere, store_classifiers=1)

# output the results
print "Learner      CA"
for i in range(len(learners)):
    print "%-12s %5.3f" % (learners[i].name, Orange.evaluation.scoring.CA(results)[i])

# find out which features were retained by filtering

print "\nNumber of times features were used in cross-validation:"
features_used = {}
for i in range(10):
    for a in results.classifiers[i][1].atts():
        if a.name in features_used.keys():
            features_used[a.name] += 1
        else:
            features_used[a.name] = 1
for k in features_used:
    print "%2d x %s" % (features_used[k], k)
