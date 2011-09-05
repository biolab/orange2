import Orange

data = Orange.data.Table("housing.tab")

knn = Orange.classification.knn.kNNLearner()

estimators = [Orange.evaluation.reliability.Mahalanobis(k=3),
              Orange.evaluation.reliability.LocalCrossValidation(k = 10)]

reliability = Orange.evaluation.reliability.Learner(knn, estimators = estimators)

restimator = reliability(data)
instance = data[0]

value, probability = restimator(instance, result_type=Orange.core.GetBoth)

for estimate in probability.reliability_estimate:
    print estimate.method_name, estimate.estimate