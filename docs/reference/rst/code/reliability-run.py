import Orange
table = Orange.data.Table("housing.tab")

knn = Orange.classification.knn.kNNLearner()
reliability = Orange.evaluation.reliability.Learner(knn)

results = Orange.evaluation.testing.cross_validation([reliability], table)

for estimate in results.results[0].probabilities[0].reliability_estimate:
    print estimate.method_name, estimate.estimate