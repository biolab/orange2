# Description: Compares naive Bayes with and without feature subset selection
# Category:    feature selection
# Uses:        voting
# Referenced:  Orange.feature.html#selection
# Classes:     Orange.feature.scoring.score_all, Orange.feature.selection.bestNAtts

import Orange


class BayesFSS(object):
    def __new__(cls, examples=None, **kwds):
        learner = object.__new__(cls)
        if examples:
            return learner(examples)
        else:
            return learner
    
    def __init__(self, name='Naive Bayes with FSS', N=5):
        self.name = name
        self.N = 5
      
    def __call__(self, table, weight=None):
        ma = Orange.feature.scoring.score_all(table)
        filtered = Orange.feature.selection.selectBestNAtts(table, ma, self.N)
        model = Orange.classification.bayes.NaiveLearner(filtered)
        return BayesFSS_Classifier(classifier=model, N=self.N, name=self.name)

class BayesFSS_Classifier:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
    
    def __call__(self, example, resultType = Orange.core.GetValue):
        return self.classifier(example, resultType)


# test above wraper on a data set
voting = Orange.data.Table("voting")
learners = (Orange.classification.bayes.NaiveLearner(name='Naive Bayes'),
            BayesFSS(name="with FSS"))
results = Orange.evaluation.testing.cross_validation(learners, voting)

# output the results
print "Learner      CA"
for i in range(len(learners)):
    print "%-12s %5.3f" % (learners[i].name, Orange.evaluation.scoring.CA(results)[i])
