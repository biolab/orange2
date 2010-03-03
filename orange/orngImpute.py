import orange

class ImputeLearner(orange.Learner):
    def __new__(cls, examples = None, weightID = 0, **keyw):
        self = orange.Learner.__new__(cls, **keyw)
        self.dontImputeClassifier = False
        self.__dict__.update(keyw)
        if examples:
            return self.__call__(examples, weightID)
        else:
            return self
        
    def __call__(self, data, weight=0):
        trained_imputer = self.imputerConstructor(data, weight)
        imputed_data = trained_imputer(data, weight)
        baseClassifier = self.baseLearner(imputed_data, weight)
        if self.dontImputeClassifier:
            return baseClassifier
        else:
            return ImputeClassifier(baseClassifier, trained_imputer)

class ImputeClassifier(orange.Classifier):
    def __init__(self, baseClassifier, imputer, **argkw):
        self.baseClassifier = baseClassifier
        self.imputer = imputer
        self.__dict__.update(argkw)

    def __call__(self, ex, what=orange.GetValue):
        return self.baseClassifier(self.imputer(ex), what)
