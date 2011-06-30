from Orange.classification.bayes import NaiveLearner as BayesLearner, NaiveClassifier as BayesClassifier
from Orange.core import BayesClassifier as _BayesClassifier

def printModel(bayesclassifier):
    """
    DEPRECATED. Replaced by :obj:`BayesClassifier.dump`.
    """
    if isinstance(bayesclassifier, BayesClassifier):
        print bayesclassifier
    elif isinstance(bayesclassifier, _BayesClassifier):
        print BayesClassifier(bayesclassifier)
    else:
        raise TypeError