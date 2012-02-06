# Description: Reliability estimation with cross-validation
# Category:    evaluation
# Uses:        housing
# Referenced:  Orange.evaluation.reliability
# Classes:     Orange.evaluation.reliability.Learner

import Orange
Orange.evaluation.reliability.select_with_repeat.random_generator = None
Orange.evaluation.reliability.select_with_repeat.randseed = 42

import Orange
table = Orange.data.Table("housing.tab")

knn = Orange.classification.knn.kNNLearner()
reliability = Orange.evaluation.reliability.Learner(knn)

results = Orange.evaluation.testing.cross_validation([reliability], table)

for i, instance in enumerate(results.results[:10]):
    print "Instance", i
    for estimate in instance.probabilities[0].reliability_estimate:
        print "  ", estimate.method_name, estimate.estimate