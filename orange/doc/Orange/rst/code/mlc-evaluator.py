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
data = Orange.data.Table("emotions.tab")

res = Orange.evaluation.testing.cross_validation(learners, data)
print_results(res)

res = Orange.evaluation.testing.leave_one_out(learners, data)
print_results(res)

res = Orange.evaluation.testing.proportion_test(learners, data, 0.5)
print_results(res)

reses = Orange.evaluation.testing.learning_curve(learners, data)
for res in reses:
    print_results(res)
