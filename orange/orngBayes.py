from Orange import NaiveLearner as BayesLearner, NaiveClassifier as BayesClassifier

def printModel(bayesclassifier):
    """
    DEPRECATED. Replaced by :obj:`BayesClassifier.dump`.
    """
    print(model if isinstance(model, TreeClassifier) else NaiveClassifier(model))