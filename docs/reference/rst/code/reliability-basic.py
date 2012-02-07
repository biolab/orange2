# Description: Reliability estimation - basic & fast
# Category:    evaluation
# Uses:        housing
# Referenced:  Orange.evaluation.reliability
# Classes:     Orange.evaluation.reliability.Mahalanobis, Orange.evaluation.reliability.LocalCrossValidation, Orange.evaluation.reliability.Learner

import Orange

housing = Orange.data.Table("housing.tab")

knn = Orange.classification.knn.kNNLearner()

estimators = [Orange.evaluation.reliability.Mahalanobis(k=3),
              Orange.evaluation.reliability.LocalCrossValidation(k = 10)]

reliability = Orange.evaluation.reliability.Learner(knn, estimators = estimators)

restimator = reliability(housing)
instance = housing[0]

value, probability = restimator(instance, result_type=Orange.classification.Classifier.GetBoth)

for estimate in probability.reliability_estimate:
    print estimate.method_name, estimate.estimate
