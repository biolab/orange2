import Orange

def print_results(res):
    loss = Orange.evaluation.scoring.mlc_hamming_loss(res)
    accuracy = Orange.evaluation.scoring.mlc_accuracy(res)
    precision = Orange.evaluation.scoring.mlc_precision(res)
    recall = Orange.evaluation.scoring.mlc_recall(res)
    print 'loss=', loss
    print 'accuracy=', accuracy
    print 'precision=', precision
    print 'recall=', recall
    print

learners = [Orange.multilabel.MLkNNLearner(k=5)]
emotions = Orange.data.Table("emotions.tab")

res = Orange.evaluation.testing.cross_validation(learners, emotions)
print_results(res)