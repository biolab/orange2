# Description: Naive Bayes Learner with auto adjusted treshold
# Category:    classification
# Uses:        iris
# Referenced:  Orange.classification.bayes
# Classes:     Orange.classification.bayes.NaiveLearner, Orange.classification.bayes.NaiveClassifier

import Orange
import orngStat
table = Orange.data.Table("adult_sample.tab")

bayes = Orange.classification.bayes.NaiveLearner(name="Naive Bayes")
adjusted_bayes = Orange.classification.bayes.NaiveLearner(adjust_threshold=True, name="Adjusted Naive Bayes")

results = Orange.evaluation.testing.cross_validation([bayes, adjusted_bayes], table)
print orngStat.CA(results)