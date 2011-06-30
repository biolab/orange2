import Orange

learners = [Orange.multilabel.BinaryRelevanceLearner(name="br")]

data = Orange.data.Table("multidata")

res = Orange.multilabel.testing.cross_validation(learners, data)
   
loss = Orange.multilabel.scoring.hamming_loss(res)
print 'loss=', loss

accuracy = Orange.multilabel.scoring.accuracy(res)
print 'accuracy=', accuracy

precision = Orange.multilabel.scoring.precision(res)
print 'precision=', precision

recall = Orange.multilabel.scoring.recall(res)
print 'recall=', recall