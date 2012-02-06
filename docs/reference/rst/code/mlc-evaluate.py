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

res = Orange.evaluation.testing.leave_one_out(learners, emotions)
print_results(res)

res = Orange.evaluation.testing.proportion_test(learners, emotions, 0.5)
print_results(res)

reses = Orange.evaluation.testing.learning_curve(learners, emotions)
for res in reses:
    print_results(res)
