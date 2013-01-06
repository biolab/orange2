from Orange import core

CostMatrix = core.CostMatrix
Classifier = core.Classifier
ClassifierList = core.ClassifierList
Learner = core.Learner
ClassifierFD = core.ClassifierFD
LearnerFD = core.LearnerFD
ClassifierFromVarFD = core.ClassifierFromVarFD
CartesianClassifier = core.CartesianClassifier
ClassifierFromVar = core.ClassifierFromVar
RandomClassifier = core.RandomClassifier
RandomLearner = core.RandomLearner
ClassifierFromVar = core.ClassifierFromVar
ConstantClassifier = core.DefaultClassifier

class PyLearner(object):
    def __new__(cls, data=None, **kwds):
        learner = object.__new__(cls)
        if data:
            learner.__init__(**kwds) # force init
            return learner(data)
        else:
            return learner  # invokes the __init__

    def __init__(self, name='learner'):
        self.name = name

    def __call__(self, data, weight=None):
        return None

class PyClassifier:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example, resultType = Classifier.GetValue):
        return self.classifier(example, resultType)
