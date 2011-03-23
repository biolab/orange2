import Orange
from Orange.feature.imputation import *

ImputeLearner = Orange.misc.deprecated_members(
  {
      "dontImputeClassifier": "dont_impute_classifier",
      "imputerConstructor": "imputer_constructor",
      "baseLearner": "base_learner"
  })(ImputeLearner)

ImputeClassifier = Orange.misc.deprecated_members(
  {
      "baseClassifier": "base_classifier"
  })(ImputeClassifier)
