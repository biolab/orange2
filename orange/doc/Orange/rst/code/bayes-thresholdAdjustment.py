# Description: Naive Bayes Learner with auto adjusted treshold
# Category:    classification
# Uses:        iris
# Referenced:  Orange.classification.bayes
# Classes:     Orange.classification.bayes.NaiveLearner, Orange.classification.bayes.NaiveClassifier

import Orange
import orngStat
table = Orange.data.Table("adult_sample.tab")

bayes = Orange.classification.bayes.NaiveLearner(name="Naive Bayes")
adjustedBayes = Orange.classification.bayes.NaiveLearner(adjustThreshold=True, name="Adjusted Naive Bayes")

results = Orange.evaluation.testing.crossValidation([bayes, adjustedBayes], table)
print orngStat.CA(results)