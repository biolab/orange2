import Orange
table = Orange.data.Table("housing.tab")

knn = Orange.classification.knn.kNNLearner()
reliability = Orange.evaluation.reliability.Learner(knn)

results = Orange.evaluation.testing.cross_validation([reliability], table)

for i, instance in enumerate(results.results[:10]):
    print "Instance", i
    for estimate in instance.probabilities[0].reliability_estimate:
        print "  ", estimate.method_name, estimate.estimate