# This file is badly structured to make it possible to embed parts of it
# into documentation.

import Orange

emotions = Orange.data.Table('emotions')
learner = Orange.multilabel.BRkNNLearner(k=5)
classifier = learner(emotions)
print classifier(emotions[0])

learner = Orange.multilabel.MLkNNLearner(k=5)
classifier = learner(emotions)
print classifier(emotions[0])

learner = Orange.multilabel.BinaryRelevanceLearner()
classifier = learner(emotions)
print classifier(emotions[0])

learner = Orange.multilabel.LabelPowersetLearner()
classifier = learner(emotions)
print classifier(emotions[0])

def test_mlc(data, learners):
    for l in learners:
        c = l(data)
        for e in data[:20]:
            labels, probs = c(e, Orange.classification.Classifier.GetBoth)
            print [val.value for val in labels], "[%s]" % ", ".join("(%.4f, %.4f)" % (p['0'], p['1']) for p in probs)
        print

learners = [Orange.multilabel.BinaryRelevanceLearner(),
            Orange.multilabel.LabelPowersetLearner(),
            Orange.multilabel.MLkNNLearner(k=1),
            Orange.multilabel.MLkNNLearner(k=5),
            Orange.multilabel.BRkNNLearner(k=1),
            Orange.multilabel.BRkNNLearner(k=5),
            Orange.multilabel.BRkNNLearner(k=5,ext='a'),
            Orange.multilabel.BRkNNLearner(k=5,ext='b')
            ]
            
test_mlc(Orange.data.Table("emotions.tab"), learners)
