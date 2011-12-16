import Orange

table = Orange.data.Table("prostate.tab")

knn = Orange.classification.knn.kNNLearner()
reliability = Orange.evaluation.reliability.Learner(knn)

res = Orange.evaluation.testing.cross_validation([reliability], table)

reliability_res = Orange.evaluation.reliability.get_pearson_r(res)

print
print "Estimate               r       p"
for estimate in reliability_res:
    print "%-20s %7.3f %7.3f" % (Orange.evaluation.reliability.METHOD_NAME[estimate[3]], \
                                 estimate[0], estimate[1])

reliability = Orange.evaluation.reliability.Learner(knn, estimators=[Orange.evaluation.reliability.SensitivityAnalysis()])

res = Orange.evaluation.testing.cross_validation([reliability], table)

reliability_res = Orange.evaluation.reliability.get_pearson_r(res)

print
print "Estimate               r       p"
for estimate in reliability_res:
    print "%-20s %7.3f %7.3f" % (Orange.evaluation.reliability.METHOD_NAME[estimate[3]], \
                                 estimate[0], estimate[1])

indices = Orange.core.MakeRandomIndices2(table, p0=0.7)
train = table.select(indices, 0)
test = table.select(indices, 1)

reliability = Orange.evaluation.reliability.Learner(knn, icv=True)
res = Orange.evaluation.testing.learn_and_test_on_test_data([reliability], train, test)

print
print "Method used in internal cross-validation: ", Orange.evaluation.reliability.METHOD_NAME[res.results[0].probabilities[0].reliability_estimate[0].method]

top5 = sorted((abs(result.probabilities[0].reliability_estimate[0].estimate), id) for id, result in enumerate(res.results))[:5]
for estimate, id in top5:
    print "%7.3f %i" % (estimate, id)
