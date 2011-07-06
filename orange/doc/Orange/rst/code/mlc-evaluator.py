import Orange

learners = [Orange.multilabel.BinaryRelevanceLearner(name="br")]
data = Orange.data.Table("multidata")

res = Orange.evaluation.testing.cross_validation(learners, data)

loss = Orange.evaluation.scoring.mlc_hamming_loss(res)
print 'loss=', loss

accuracy = Orange.evaluation.scoring.mlc_accuracy(res)
print 'accuracy=', accuracy

precision = Orange.evaluation.scoring.mlc_precision(res)
print 'precision=', precision

recall = Orange.evaluation.scoring.mlc_recall(res)
print 'recall=', recall
