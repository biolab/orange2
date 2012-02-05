# Description: Naive Bayes Learner with m-estimate
# Category:    classification
# Uses:        iris
# Referenced:  Orange.classification.bayes
# Classes:     Orange.classification.bayes.NaiveLearner, Orange.classification.bayes.NaiveClassifier

import Orange

lenses = Orange.data.Table("lenses.tab")

bayes_L = Orange.classification.bayes.NaiveLearner(name="Naive Bayes")
bayesWithM_L = Orange.classification.bayes.NaiveLearner(m=2, name="Naive Bayes w/ m-estimate")
bayes = bayes_L(lenses)
bayesWithM = bayesWithM_L(lenses)

print bayes.conditional_distributions
# prints: <<'pre-presbyopic': <0.625, 0.125, 0.250>, 'presbyopic': <0.750, 0.125, 0.125>, ...>>
print bayesWithM.conditional_distributions
# prints: <<'pre-presbyopic': <0.625, 0.133, 0.242>, 'presbyopic': <0.725, 0.133, 0.142>, ...>>

print bayes.distribution
# prints: <0.625, 0.167, 0.208>
print bayesWithM.distribution
# prints: <0.625, 0.167, 0.208>
