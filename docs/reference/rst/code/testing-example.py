import Orange

iris = Orange.data.Table("iris")
learners = [Orange.classification.bayes.NaiveLearner(),
            Orange.classification.majority.MajorityLearner()]

cv = Orange.evaluation.testing.cross_validation(learners, iris, folds=5)
print Orange.evaluation.scoring.CA(cv)

