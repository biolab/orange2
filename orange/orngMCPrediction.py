import orange
import orngRegression


class MultiClassPredictionLearner(object):
    def __new__(self, data=None, name='PLS regression', **kwds):
        learner = object.__new__(self, **kwds)
        if data:
            learner.__init__(name) # force init
            return learner(data)
        else:
            return learner  # invokes the __init__

    def __init__(self, name='MultiClassPredictionLearner', baseLearner=orngRegression.PLSRegressionLearner):
        self.name = name
        self.baseLearner = baseLearner
                
    def __call__(self, data, y, x=None, nc=None, weight=None):
        if x == None:
            x = [v for v in data.domain.variables if v not in y]

        if self.baseLearner == orngRegression.PLSRegressionLearner:
            lr = self.baseLearner()
            return MultiClassPrediction(baseClassifier=lr(data, y, x, nc), baseLearner=self.baseLearner)

        if self.baseLearner == orange.SVMLearner:
            dom = orange.Domain(x)
            lr = self.baseLearner()
            lr.svm_type=orange.SVMLearner.NU_SVR
            models = []
            for a in y:
                newDomain = orange.Domain(dom, a)
                newData = orange.ExampleTable(newDomain, data)
                models.append(lr(newData))
            return MultiClassPrediction(baseClassifier=models, baseLearner = self.baseLearner)

         
class MultiClassPrediction:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example):
        if self.baseLearner == orngRegression.PLSRegressionLearner:
            print 'lalala'
            return self.baseClassifier(example)
        elif self.baseLearner == orange.SVMLearner:
            yhat = []
            for cl in self.baseClassifier:
                yhat.append(cl(example))
            print yhat
        





