import Orange

learners = [
        Orange.multilabel.BinaryRelevanceLearner(name="br"),
        #Orange.multilabel.BinaryRelevanceLearner(name="br",base_learner=Orange.classification.knn.kNNLearner),
        #Orange.multilabel.BinaryRelevanceLearner(name="br",base_learner=Orange.classification.bayes.NaiveLearner),
        #Orange.multilabel.LabelPowersetLearner(name="lp",base_learner=Orange.classification.bayes.NaiveLearner),
        Orange.multilabel.LabelPowersetLearner(name="lp"),
        #Orange.multilabel.LabelPowersetLearner(name="lp",base_learner=Orange.classification.knn.kNNLearner),
        #Orange.multilabel.MLkNNLearner(name="mlknn",k=5),
        #Orange.multilabel.BRkNNLearner(name="brknn",k=5),
        #Orange.multilabel.BRkNNLearner(name="brknn",k=5,ext='a'),
        #Orange.multilabel.BRkNNLearner(name="brknn",k=5,ext='b'),
            ]

data = Orange.data.Table("emotions.tab")

res = Orange.evaluation.testing.cross_validation(learners, data,2)
loss = Orange.evaluation.scoring.mlc_hamming_loss(res)
accuracy = Orange.evaluation.scoring.mlc_accuracy(res)
precision = Orange.evaluation.scoring.mlc_precision(res)
recall = Orange.evaluation.scoring.mlc_recall(res)
print 'loss=', loss
print 'accuracy=', accuracy
print 'precision=', precision
print 'recall=', recall

