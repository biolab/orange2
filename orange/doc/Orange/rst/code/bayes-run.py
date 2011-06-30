# Description: Naive Bayes Learner on iris data set
# Category:    classification
# Uses:        titanic
# Referenced:  Orange.classification.bayes
# Classes:     Orange.classification.bayes.NaiveLearner, Orange.classification.bayes.NaiveClassifier

import Orange
table = Orange.data.Table("titanic.tab")

learner = Orange.classification.bayes.NaiveLearner()
classifier = learner(table)

for ex in table[:5]:
    print ex.getclass(), classifier(ex)
