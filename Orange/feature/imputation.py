import Orange.core as orange
from orange import\
    Imputer as Imputer,\
    ImputerConstructor as Constructor,\
    ImputerConstructor_minimal as MinimalConstructor,\
    ImputerConstructor_maximal as MaximalConstructor,\
    ImputerConstructor_average as AverageConstructor,\
    ImputerConstructor_model as ModelConstructor,\
    ImputerConstructor_asValue as AsValueConstructor,\
    ImputerConstructor_random as RandomConstructor,\
    Imputer_defaults as Defaults,\
    Imputer_model as Model,\
    Imputer_random as Random,\
    Imputer_asValue as AsValue,\
    ImputeClassifier as Classifier

import Orange.misc

class ImputeLearner(orange.Learner):
    def __new__(cls, data = None, weight_id = 0, **keyw):
        self = orange.Learner.__new__(cls, **keyw)
        self.dont_impute_classifier = False
        self.__dict__.update(keyw)
        if data:
            return self.__call__(data, weight_id)
        else:
            return self
        
    def __call__(self, data, weight=0):
        trained_imputer = self.imputer_constructor(data, weight)
        imputed_data = trained_imputer(data, weight)
        base_classifier = self.base_learner(imputed_data, weight)
        if self.dont_impute_classifier:
            return base_classifier
        else:
            return ImputeClassifier(base_classifier, trained_imputer)

ImputeLearner = Orange.misc.deprecated_members(
  {
      "dontImputeClassifier": "dont_impute_classifier",
      "imputerConstructor": "imputer_constructor",
      "baseLearner": "base_learner",
      "weightID": "weight_id"
  })(ImputeLearner)


class ImputeClassifier(orange.Classifier):
    def __init__(self, base_classifier, imputer, **argkw):
        self.base_classifier = base_classifier
        self.imputer = imputer
        self.__dict__.update(argkw)

    def __call__(self, i, what=orange.GetValue):
        return self.base_classifier(self.imputer(i), what)

ImputeClassifier = Orange.misc.deprecated_members(
  {
      "baseClassifier": "base_classifier"
  })(ImputeClassifier)
