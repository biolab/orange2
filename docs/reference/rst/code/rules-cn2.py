# Description: CN2 Rule Induction on Titanic dataset
# Category:    classification
# Uses:        titanic
# Referenced:  Orange.classification.rules
# Classes:     Orange.classification.rules.CN2Learner

import Orange
# Read some data
titanic = Orange.data.Table("titanic")

# construct the learning algorithm and use it to induce a classifier
cn2_learner = Orange.classification.rules.CN2Learner()
cn2_clasifier = cn2_learner(titanic)

# ... or, in a single step.
cn2_classifier = Orange.classification.rules.CN2Learner(titanic)

# All rule-base classifiers can have their rules printed out like this:
for r in cn2_classifier.rules:
    print Orange.classification.rules.rule_to_string(r)
