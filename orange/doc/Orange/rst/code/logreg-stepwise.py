import orngTest
import orngFSS
from Orange import *
import Orange.evaluation.testing


def StepWiseFSS_Filter(examples=None, **kwds):
    """Check function StepWiseFSS()."""

    filter = apply(StepWiseFSS_Filter_class, (), kwds)
    if examples:
        return filter(examples)
    else:
        return filter


class StepWiseFSS_Filter_class:
    def __init__(self, addCrit=0.2, deleteCrit=0.3, numAttr=-1):
        self.addCrit = addCrit
        self.deleteCrit = deleteCrit
        self.numAttr = numAttr

    def __call__(self, examples):
        feature = classification.logreg.StepWiseFSS(examples,
          addCrit=self.addCrit, deleteCrit=self.deleteCrit,
          numAttr=self.numAttr)
        return examples.select(data.Domain(feature, examples.domain.classVar))


table = data.Table("ionosphere.tab")

lr = classification.logreg.LogRegLearner(removeSingular=1)
learners = (
  classification.logreg.LogRegLearner(name='logistic', removeSingular=1),
  feature.selection.FilteredLearner(lr,
     filter=StepWiseFSS_Filter(addCrit=0.05, deleteCrit=0.9),
     name='filtered')
)
results = Orange.evaluation.testing.crossValidation(learners, table, storeClassifiers=1)

# output the results
print "Learner      CA"
for i in range(len(learners)):
    print "%-12s %5.3f" % (learners[i].name, evaluation.scoring.CA(results)[i])

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
