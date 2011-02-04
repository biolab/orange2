# Description: Naive Bayes Learner with auto adjusted treshold
# Category:    classification
# Uses:        iris
# Referenced:  Orange.classification.bayes
# Classes:     Orange.classification.bayes.NaiveLearner, Orange.classification.bayes.NaiveClassifier

import Orange
import orngStat
table = Orange.data.Table("iris.tab")
Orange.evaluation.testing.crossValidation()

bayes = Orange.classification.bayes.NaiveLearner(table, name="Naive Bayes")
adjustedBayes = Orange.classification.bayes.NaiveLearner(table, adjustTreshold=True, name="Adjusted Naive Bayes")

results = Orange.evaluation.testing.crossValidation([bayes, adjustedBayes], table)
print orngStat.CA(results)