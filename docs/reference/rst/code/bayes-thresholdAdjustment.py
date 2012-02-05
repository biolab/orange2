# Description: Naive Bayes Learner with auto adjusted treshold
# Category:    classification
# Uses:        iris
# Referenced:  Orange.classification.bayes
# Classes:     Orange.classification.bayes.NaiveLearner, Orange.classification.bayes.NaiveClassifier

import Orange
from Orange.classification import bayes
from Orange.evaluation import testing, scoring

adult = Orange.data.Table("adult_sample.tab")

nb = bayes.NaiveLearner(name="Naive Bayes")
adjusted_nb = bayes.NaiveLearner(adjust_threshold=True, name="Adjusted Naive Bayes")

results = testing.cross_validation([nb, adjusted_nb], adult)
print scoring.CA(results)