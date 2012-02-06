import Orange
from Orange.core import BayesLearner as _BayesLearner

class MultiLabelLearner(Orange.classification.Learner):
    def __new__(cls, **argkw):
        self = Orange.classification.Learner.__new__(cls, **argkw)
        return self
    
    def __init__(self, **argkw):
        self.__dict__.update(argkw)
        
class MultiLabelClassifier(Orange.classification.Classifier):
    def __init__(self, **argkw):
        self.__dict__.update(argkw)
     
