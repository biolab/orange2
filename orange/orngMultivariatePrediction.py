import orange
import orngRegression

class MultiClassPredictionLearner(object):
    """(self, data, y, x=None)"""
    def __new__(self, data=None, name='multivar pred', **kwds):
        learner = object.__new__(self, **kwds)
        if data:
            learner.__init__(name) # force init
            return learner(data)
        else:
            return learner  # invokes the __init__

    def __init__(self, name='multivar pred', baseLearner=orngRegression.LinearRegressionLearner):
        self.name = name
        self.baseLearner = baseLearner
                
    def __call__(self, data, y, x=None, weight=None):
        if y == None:
            try:
                y = [data.domain.classVar]
            except:
                import warnings
                warnings.warn("multi-class learner requires either specification of response variables or a data domain with a class")
                return None
        if x == None:
            print y
            x = [v for v in data.domain.variables if v not in y]

        models = []
        for a in y:
            newDomain = orange.Domain(x, a)
            newData = orange.ExampleTable(newDomain, data)
            models.append(baseLearner(newData))
        return MultiClassPrediction(x=x, y=y, models=models)
       
class MultiClassPrediction:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example):
        return [m(example) for m in self.models]
