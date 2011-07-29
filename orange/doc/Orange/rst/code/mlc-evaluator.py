import Orange

learners = [Orange.multilabel.BinaryRelevanceLearner(name="br")]
data = Orange.data.Table("multidata.tab")

res = Orange.evaluation.testing.cross_validation(learners, data)
loss = Orange.evaluation.scoring.mlc_hamming_loss(res)
accuracy = Orange.evaluation.scoring.mlc_accuracy(res)
precision = Orange.evaluation.scoring.mlc_precision(res)
recall = Orange.evaluation.scoring.mlc_recall(res)
print 'loss=', loss
print 'accuracy=', accuracy
print 'precision=', precision
print 'recall=', recall

res = Orange.evaluation.testing.leave_one_out(learners, data)
loss = Orange.evaluation.scoring.mlc_hamming_loss(res)
accuracy = Orange.evaluation.scoring.mlc_accuracy(res)
precision = Orange.evaluation.scoring.mlc_precision(res)
recall = Orange.evaluation.scoring.mlc_recall(res)
print 'loss=', loss
print 'accuracy=', accuracy
print 'precision=', precision
print 'recall=', recall

res = Orange.evaluation.testing.proportion_test(learners, data, 0.5)
loss = Orange.evaluation.scoring.mlc_hamming_loss(res)
accuracy = Orange.evaluation.scoring.mlc_accuracy(res)
precision = Orange.evaluation.scoring.mlc_precision(res)
recall = Orange.evaluation.scoring.mlc_recall(res)
print 'loss=', loss
print 'accuracy=', accuracy
print 'precision=', precision
print 'recall=', recall

reses = Orange.evaluation.testing.learning_curve(learners, data)
for res in reses:
    loss = Orange.evaluation.scoring.mlc_hamming_loss(res)
    accuracy = Orange.evaluation.scoring.mlc_accuracy(res)
    precision = Orange.evaluation.scoring.mlc_precision(res)
    recall = Orange.evaluation.scoring.mlc_recall(res)
    print 'loss=', loss,
    print 'accuracy=', accuracy,
    print 'precision=', precision,
    print 'recall=', recall,
    print
