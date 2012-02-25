import math
import functools
from operator import add

import numpy

import Orange
from Orange import statc, corn
from Orange.misc import deprecated_keywords, deprecated_function_name, \
    deprecation_warning, environ
from Orange.evaluation import testing

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None
    HAS_MATPLOTLIB = False

#### Private stuff

def log2(x):
    """Calculate logarithm in base 2."""
    return math.log(x) / math.log(2)

def check_non_zero(x):
    """Throw Value Error when x = 0."""
    if x == 0.:
        raise ValueError, "Cannot compute the score: no examples or sum of weights is 0."

def gettotweight(res):
    """Sum all the weights."""
    totweight = reduce(lambda x, y: x + y.weight, res.results, 0)
    if totweight == 0.:
        raise ValueError, "Cannot compute the score: sum of weights is 0."
    return totweight

def gettotsize(res):
    """Get number of result instances."""
    if len(res.results):
        return len(res.results)
    else:
        raise ValueError, "Cannot compute the score: no examples."


def split_by_iterations(res):
    """Split ExperimentResults of a multiple iteratation test into a list
    of ExperimentResults, one for each iteration.
    """
    if res.number_of_iterations < 2:
        return [res]
        
    ress = [Orange.evaluation.testing.ExperimentResults(
                1, res.classifier_names, res.class_values,
                res.weights, classifiers=res.classifiers,
                loaded=res.loaded, test_type=res.test_type, labels=res.labels)
            for _ in range(res.number_of_iterations)]
    for te in res.results:
        ress[te.iteration_number].results.append(te)
    return ress

def split_by_classifiers(res):
    """Split an instance of :obj:`ExperimentResults` into a list of
    :obj:`ExperimentResults`, one for each classifier. 
    """
    split_res = []
    for i in range(len(res.classifierNames)):
        r = Orange.evaluation.testing.ExperimentResults(res.numberOfIterations,
                    [res.classifierNames[i]], res.classValues,
                    weights=res.weights, baseClass=res.baseClass,
                    classifiers=[res.classifiers[i]] if res.classifiers else [],
                    test_type=res.test_type, labels=res.labels)
        r.results = []
        for te in res.results:
            r.results.append(Orange.evaluation.testing.TestedExample(
                te.iterationNumber, te.actualClass, n=1, weight=te.weight))
            r.results[-1].classes = [te.classes[i]]
            r.results[-1].probabilities = [te.probabilities[i]]
        split_res.append(r)
    return split_res


def class_probabilities_from_res(res, **argkw):
    """Calculate class probabilities."""
    probs = [0.] * len(res.class_values)
    if argkw.get("unweighted", 0) or not res.weights:
        for tex in res.results:
            probs[int(tex.actual_class)] += 1.
        totweight = gettotsize(res)
    else:
        totweight = 0.
        for tex in res.results:
            probs[tex.actual_class] += tex.weight
            totweight += tex.weight
        check_non_zero(totweight)
    return [prob / totweight for prob in probs]


@deprecated_keywords({
    "foldN": "fold_n",
    "reportSE": "report_se",
    "iterationIsOuter": "iteration_is_outer"})
def statistics_by_folds(stats, fold_n, report_se, iteration_is_outer):
    # remove empty folds, turn the matrix so that learner is outer
    if iteration_is_outer:
        if not stats:
            raise ValueError, "Cannot compute the score: no examples or sum of weights is 0."
        number_of_learners = len(stats[0])
        stats = filter(lambda (x, fN): fN > 0, zip(stats, fold_n))
        stats = [[x[lrn] / fN for x, fN in stats]
                 for lrn in range(number_of_learners)]
    else:
        stats = [[x / Fn for x, Fn in filter(lambda (x, Fn): Fn > 0,
                 zip(lrnD, fold_n))] for lrnD in stats]

    if not stats:
        raise ValueError, "Cannot compute the score: no classifiers"
    if not stats[0]:
        raise ValueError, "Cannot compute the score: no examples or sum of weights is 0."
    
    if report_se:
        return [(statc.mean(x), statc.sterr(x)) for x in stats]
    else:
        return [statc.mean(x) for x in stats]
    
def ME(res, **argkw):
    MEs = [0.] * res.number_of_learners

    if argkw.get("unweighted", 0) or not res.weights:
        for tex in res.results:
            MEs = map(lambda res, cls, ac = float(tex.actual_class):
                      res + abs(float(cls) - ac), MEs, tex.classes)
        totweight = gettotsize(res)
    else:
        for tex in res.results:
            MEs = map(lambda res, cls, ac=float(tex.actual_class), tw=tex.weight:
                       res + tw * abs(float(cls) - ac), MEs, tex.classes)
        totweight = gettotweight(res)

    return [x / totweight for x in MEs]

MAE = ME


class ConfusionMatrix:
    """
    Classification result summary.
    """
    #: True Positive predictions
    TP = 0.
    #:True Negative predictions
    TN = 0.
    #:False Positive predictions
    FP = 0.
    #: False Negative predictions
    FN = 0.

    @deprecated_keywords({"predictedPositive": "predicted_positive",
                          "isPositive": "is_positive"})
    def addTFPosNeg(self, predicted_positive, is_positive, weight=1.0):
        """
        Update confusion matrix with result of a single classification.

        :param predicted_positive: positive class value was predicted
        :param is_positive: correct class value is positive
        :param weight: weight of the selected instance
         """
        if predicted_positive:
            if is_positive:
                self.TP += weight
            else:
                self.FP += weight
        else:
            if is_positive:
                self.FN += weight
            else:
                self.TN += weight



#########################################################################
# PERFORMANCE MEASURES:
# Scores for evaluation of numeric predictions

def check_argkw(dct, lst):
    """Return True if any item from lst has a non-zero value in dct."""
    return reduce(lambda x, y: x or y, [dct.get(k, 0) for k in lst])

def regression_error(res, **argkw):
    """Return the regression error (default: MSE)."""
    if argkw.get("SE", 0) and res.number_of_iterations > 1:
        # computes the scores for each iteration, then averages
        scores = [[0.] * res.number_of_iterations
                  for _ in range(res.number_of_learners)]
        norm = None
        if argkw.get("norm-abs", 0) or argkw.get("norm-sqr", 0):
            norm = [0.] * res.number_of_iterations

        # counts examples in each iteration
        nIter = [0] * res.number_of_iterations
        # average class in each iteration
        a = [0] * res.number_of_iterations
        for tex in res.results:
            nIter[tex.iteration_number] += 1
            a[tex.iteration_number] += float(tex.actual_class)
        a = [a[i] / nIter[i] for i in range(res.number_of_iterations)]

        if argkw.get("unweighted", 0) or not res.weights:
            # iterate accross test cases
            for tex in res.results:
                ai = float(tex.actual_class)
                nIter[tex.iteration_number] += 1

                # compute normalization, if required
                if argkw.get("norm-abs", 0):
                    norm[tex.iteration_number] += abs(ai - a[tex.iteration_number])
                elif argkw.get("norm-sqr", 0):
                    norm[tex.iteration_number] += (ai - a[tex.iteration_number])**2

                # iterate accross results of different regressors
                for i, cls in enumerate(tex.classes):
                    if argkw.get("abs", 0):
                        scores[i][tex.iteration_number] += abs(float(cls) - ai)
                    else:
                        scores[i][tex.iteration_number] += (float(cls) - ai)**2
        else: # unweighted != 0
            raise NotImplementedError, "weighted error scores with SE not implemented yet"

        if argkw.get("norm-abs") or argkw.get("norm-sqr"):
            scores = [[x / n for x, n in zip(y, norm)] for y in scores]
        else:
            scores = [[x / ni for x, ni in zip(y, nIter)] for y in scores]

        if argkw.get("R2"):
            scores = [[1.0 - x for x in y] for y in scores]

        if argkw.get("sqrt", 0):
            scores = [[math.sqrt(x) for x in y] for y in scores]

        return [(statc.mean(x), statc.std(x)) for x in scores]
        
    else: # single iteration (testing on a single test set)
        scores = [0.] * res.number_of_learners
        norm = 0.

        if argkw.get("unweighted", 0) or not res.weights:
            a = sum([tex.actual_class for tex in res.results]) \
                / len(res.results)
            for tex in res.results:
                if argkw.get("abs", 0):
                    scores = map(lambda res, cls, ac=float(tex.actual_class):
                                 res + abs(float(cls) - ac), scores, tex.classes)
                else:
                    scores = map(lambda res, cls, ac=float(tex.actual_class):
                                 res + (float(cls) - ac)**2, scores, tex.classes)

                if argkw.get("norm-abs", 0):
                    norm += abs(tex.actual_class - a)
                elif argkw.get("norm-sqr", 0):
                    norm += (tex.actual_class - a)**2
            totweight = gettotsize(res)
        else:
            # UNFINISHED
            MSEs = [0.] * res.number_of_learners
            for tex in res.results:
                MSEs = map(lambda res, cls, ac=float(tex.actual_class),
                           tw=tex.weight:
                           res + tw * (float(cls) - ac)**2, MSEs, tex.classes)
            totweight = gettotweight(res)

        if argkw.get("norm-abs", 0) or argkw.get("norm-sqr", 0):
            scores = [s / norm for s in scores]
        else: # normalize by number of instances (or sum of weights)
            scores = [s / totweight for s in scores]

        if argkw.get("R2"):
            scores = [1. - s for s in scores]

        if argkw.get("sqrt", 0):
            scores = [math.sqrt(x) for x in scores]

        return scores

def MSE(res, **argkw):
    """Compute mean-squared error."""
    return regression_error(res, **argkw)
    
def RMSE(res, **argkw):
    """Compute root mean-squared error."""
    argkw.setdefault("sqrt", True)
    return regression_error(res, **argkw)

def MAE(res, **argkw):
    """Compute mean absolute error."""
    argkw.setdefault("abs", True)
    return regression_error(res, **argkw)

def RSE(res, **argkw):
    """Compute relative squared error."""
    argkw.setdefault("norm-sqr", True)
    return regression_error(res, **argkw)

def RRSE(res, **argkw):
    """Compute relative squared error."""
    argkw.setdefault("norm-sqr", True)
    argkw.setdefault("sqrt", True)
    return regression_error(res, **argkw)

def RAE(res, **argkw):
    """Compute relative absolute error."""
    argkw.setdefault("abs", True)
    argkw.setdefault("norm-abs", True)
    return regression_error(res, **argkw)

def R2(res, **argkw):
    """Compute the coefficient of determination, R-squared."""
    argkw.setdefault("norm-sqr", True)
    argkw.setdefault("R2", True)
    return regression_error(res, **argkw)

def MSE_old(res, **argkw):
    """Compute mean-squared error."""
    if argkw.get("SE", 0) and res.number_of_iterations > 1:
        MSEs = [[0.] * res.number_of_iterations
                for _ in range(res.number_of_learners)]
        nIter = [0] * res.number_of_iterations
        if argkw.get("unweighted", 0) or not res.weights:
            for tex in res.results:
                ac = float(tex.actual_class)
                nIter[tex.iteration_number] += 1
                for i, cls in enumerate(tex.classes):
                    MSEs[i][tex.iteration_number] += (float(cls) - ac)**2
        else:
            raise ValueError, "weighted RMSE with SE not implemented yet"
        MSEs = [[x / ni for x, ni in zip(y, nIter)] for y in MSEs]
        if argkw.get("sqrt", 0):
            MSEs = [[math.sqrt(x) for x in y] for y in MSEs]
        return [(statc.mean(x), statc.std(x)) for x in MSEs]
        
    else:
        MSEs = [0.] * res.number_of_learners
        if argkw.get("unweighted", 0) or not res.weights:
            for tex in res.results:
                MSEs = map(lambda res, cls, ac=float(tex.actual_class):
                           res + (float(cls) - ac)**2, MSEs, tex.classes)
            totweight = gettotsize(res)
        else:
            for tex in res.results:
                MSEs = map(lambda res, cls, ac=float(tex.actual_class),
                           tw=tex.weight: res + tw * (float(cls) - ac)**2,
                           MSEs, tex.classes)
            totweight = gettotweight(res)

        if argkw.get("sqrt", 0):
            MSEs = [math.sqrt(x) for x in MSEs]
        return [x / totweight for x in MSEs]

def RMSE_old(res, **argkw):
    """Compute root mean-squared error."""
    argkw.setdefault("sqrt", 1)
    return MSE_old(res, **argkw)

#########################################################################
# PERFORMANCE MEASURES:
# Scores for evaluation of classifiers

class CA(list):
    """
    Compute percentage of matches between predicted and actual class.

    :param test_results: :obj:`~Orange.evaluation.testing.ExperimentResults`
                         or list of :obj:`ConfusionMatrix`.
    :param report_se: include standard error in result.
    :param ignore_weights: ignore instance weights.
    :rtype: list of scores, one for each learner.

    Standard errors are estimated from deviation of CAs across folds (if
    test_results were produced by cross_validation) or approximated under
    the assumption of normal distribution otherwise.
    """
    CONFUSION_MATRIX = 0
    CONFUSION_MATRIX_LIST = 1
    CLASSIFICATION = 2
    CROSS_VALIDATION = 3

    @deprecated_keywords({"reportSE": "report_se",
                          "unweighted": "ignore_weights"})
    def __init__(self, test_results, report_se = False, ignore_weights=False):
        super(CA, self).__init__()
        self.report_se = report_se
        self.ignore_weights = ignore_weights

        input_type = self.get_input_type(test_results)
        if input_type == self.CONFUSION_MATRIX:
            self[:] = [self.from_confusion_matrix(test_results)]
        elif input_type == self.CONFUSION_MATRIX_LIST:
            self[:] = self.from_confusion_matrix_list(test_results)
        elif input_type == self.CLASSIFICATION:
            self[:] = self.from_classification_results(test_results)
        elif input_type == self.CROSS_VALIDATION:
            self[:] = self.from_crossvalidation_results(test_results)

    def from_confusion_matrix(self, cm):
        all_predictions = 0.
        correct_predictions = 0.
        if isinstance(cm, ConfusionMatrix):
            all_predictions += cm.TP + cm.FN + cm.FP + cm.TN
            correct_predictions += cm.TP + cm.TN
        else:
            for r, row in enumerate(cm):
                for c, column in enumerate(row):
                    if r == c:
                        correct_predictions += column
                    all_predictions += column

        check_non_zero(all_predictions)
        ca = correct_predictions / all_predictions

        if self.report_se:
            return ca, ca * (1 - ca) / math.sqrt(all_predictions)
        else:
            return ca

    def from_confusion_matrix_list(self, confusion_matrices):
        return [self.from_confusion_matrix(cm) for cm in confusion_matrices]

    def from_classification_results(self, test_results):
        CAs = [0.] * test_results.number_of_learners
        totweight = 0.
        for tex in test_results.results:
            w = 1. if self.ignore_weights else tex.weight
            CAs = map(lambda res, cls: res + (cls == tex.actual_class and w),
                      CAs, tex.classes)
            totweight += w
        check_non_zero(totweight)
        ca = [x / totweight for x in CAs]

        if self.report_se:
            return [(x, x * (1 - x) / math.sqrt(totweight)) for x in ca]
        else:
            return ca

    def from_crossvalidation_results(self, test_results):
        CAsByFold = [[0.] * test_results.number_of_iterations
                     for _ in range(test_results.number_of_learners)]
        foldN = [0.] * test_results.number_of_iterations

        for tex in test_results.results:
            w = 1. if self.ignore_weights else tex.weight
            for lrn in range(test_results.number_of_learners):
                CAsByFold[lrn][tex.iteration_number] += (tex.classes[lrn] == 
                    tex.actual_class) and w
            foldN[tex.iteration_number] += w

        return statistics_by_folds(CAsByFold, foldN, self.report_se, False)

    def get_input_type(self, test_results):
        if isinstance(test_results, ConfusionMatrix):
            return self.CONFUSION_MATRIX
        elif isinstance(test_results, testing.ExperimentResults):
            if test_results.number_of_iterations == 1:
                return self.CLASSIFICATION
            else:
                return self.CROSS_VALIDATION
        elif isinstance(test_results, list):
            return self.CONFUSION_MATRIX_LIST


@deprecated_keywords({"reportSE": "report_se",
                      "unweighted": "ignore_weights"})
def AP(res, report_se=False, ignore_weights=False, **argkw):
    """Compute the average probability assigned to the correct class."""
    if res.number_of_iterations == 1:
        APs=[0.] * res.number_of_learners
        if ignore_weights or not res.weights:
            for tex in res.results:
                APs = map(lambda res, probs: res + probs[tex.actual_class],
                          APs, tex.probabilities)
            totweight = gettotsize(res)
        else:
            totweight = 0.
            for tex in res.results:
                APs = map(lambda res, probs: res + probs[tex.actual_class] *
                          tex.weight, APs, tex.probabilities)
                totweight += tex.weight
        check_non_zero(totweight)
        return [AP / totweight for AP in APs]

    APsByFold = [[0.] * res.number_of_learners
                 for _ in range(res.number_of_iterations)]
    foldN = [0.] * res.number_of_iterations
    if ignore_weights or not res.weights:
        for tex in res.results:
            APsByFold[tex.iteration_number] = map(lambda res, probs:
                res + probs[tex.actual_class],
                APsByFold[tex.iteration_number], tex.probabilities)
            foldN[tex.iteration_number] += 1
    else:
        for tex in res.results:
            APsByFold[tex.iteration_number] = map(lambda res, probs:
                res + probs[tex.actual_class] * tex.weight,
                APsByFold[tex.iteration_number], tex.probabilities)
            foldN[tex.iteration_number] += tex.weight

    return statistics_by_folds(APsByFold, foldN, report_se, True)


@deprecated_keywords({"reportSE": "report_se",
                      "unweighted": "ignore_weights"})
def Brier_score(res, report_se=False, ignore_weights=False, **argkw):
    """Compute the Brier score, defined as the average (over test instances)
    of :math:`\sum_x(t(x) - p(x))^2`, where x is a class value, t(x) is 1 for
    the actual class value and 0 otherwise, and p(x) is the predicted
    probability of x.
    """
    # Computes an average (over examples) of sum_x(t(x) - p(x))^2, where
    #    x is class,
    #    t(x) is 0 for 'wrong' and 1 for 'correct' class
    #    p(x) is predicted probabilty.
    # There's a trick: since t(x) is zero for all classes but the
    # correct one (c), we compute the sum as sum_x(p(x)^2) - 2*p(c) + 1
    # Since +1 is there for each example, it adds 1 to the average
    # We skip the +1 inside the sum and add it just at the end of the function
    # We take max(result, 0) to avoid -0.0000x due to rounding errors

    if res.number_of_iterations == 1:
        MSEs=[0.] * res.number_of_learners
        if ignore_weights or not res.weights:
            totweight = 0.
            for tex in res.results:
                MSEs = map(lambda res, probs: res + reduce(
                    lambda s, pi: s + pi**2, probs, 0) - 
                    2 * probs[tex.actual_class], MSEs, tex.probabilities)
                totweight += tex.weight
        else:
            for tex in res.results:
                MSEs = map(lambda res, probs: res + tex.weight * reduce(
                    lambda s, pi: s + pi**2, probs, 0) - 
                    2 * probs[tex.actual_class], MSEs, tex.probabilities)
            totweight = gettotweight(res)
        check_non_zero(totweight)
        if report_se:
            ## change this, not zero!!!
            return [(max(x / totweight + 1., 0), 0) for x in MSEs]
        else:
            return [max(x / totweight + 1., 0) for x in MSEs]

    BSs = [[0.] * res.number_of_learners
           for _ in range(res.number_of_iterations)]
    foldN = [0.] * res.number_of_iterations

    if ignore_weights or not res.weights:
        for tex in res.results:
            BSs[tex.iteration_number] = map(lambda rr, probs: rr + reduce(
                lambda s, pi: s + pi**2, probs, 0) -
                2 * probs[tex.actual_class], BSs[tex.iteration_number],
                tex.probabilities)
            foldN[tex.iteration_number] += 1
    else:
        for tex in res.results:
            BSs[tex.iteration_number] = map(lambda res, probs:
                res + tex.weight * reduce(lambda s, pi: s + pi**2, probs, 0) -
                2 * probs[tex. actual_class], BSs[tex.iteration_number],
                tex.probabilities)
            foldN[tex.iteration_number] += tex.weight

    stats = statistics_by_folds(BSs, foldN, report_se, True)
    if report_se:
        return [(x + 1., y) for x, y in stats]
    else:
        return [x + 1. for x in stats]

def BSS(res, **argkw):
    return [1 - x / 2 for x in apply(Brier_score, (res, ), argkw)]

def IS_ex(Pc, P):
    """Pc aposterior probability, P aprior"""
    if Pc >= P:
        return -log2(P) + log2(Pc)
    else:
        return -(-log2(1 - P) + log2(1 - Pc))


@deprecated_keywords({"reportSE": "report_se"})
def IS(res, apriori=None, report_se=False, **argkw):
    """Compute the information score as defined by 
    `Kononenko and Bratko (1991) \
    <http://www.springerlink.com/content/g5p7473160476612/>`_.
    Argument :obj:`apriori` gives the apriori class
    distribution; if it is omitted, the class distribution is computed from
    the actual classes of examples in :obj:`res`.
    """
    if not apriori:
        apriori = class_probabilities_from_res(res)

    if res.number_of_iterations == 1:
        ISs = [0.] * res.number_of_learners
        if argkw.get("unweighted", 0) or not res.weights:
            for tex in res.results:
              for i in range(len(tex.probabilities)):
                    cls = tex.actual_class
                    ISs[i] += IS_ex(tex.probabilities[i][cls], apriori[cls])
            totweight = gettotsize(res)
        else:
            for tex in res.results:
              for i in range(len(tex.probabilities)):
                    cls = tex.actual_class
                    ISs[i] += (IS_ex(tex.probabilities[i][cls], apriori[cls]) *
                               tex.weight)
            totweight = gettotweight(res)
        if report_se:
            return [(IS / totweight, 0) for IS in ISs]
        else:
            return [IS / totweight for IS in ISs]

        
    ISs = [[0.] * res.number_of_iterations
           for _ in range(res.number_of_learners)]
    foldN = [0.] * res.number_of_iterations

    # compute info scores for each fold    
    if argkw.get("unweighted", 0) or not res.weights:
        for tex in res.results:
            for i in range(len(tex.probabilities)):
                cls = tex.actual_class
                ISs[i][tex.iteration_number] += IS_ex(tex.probabilities[i][cls],
                                                apriori[cls])
            foldN[tex.iteration_number] += 1
    else:
        for tex in res.results:
            for i in range(len(tex.probabilities)):
                cls = tex.actual_class
                ISs[i][tex.iteration_number] += IS_ex(tex.probabilities[i][cls],
                                                apriori[cls]) * tex.weight
            foldN[tex.iteration_number] += tex.weight

    return statistics_by_folds(ISs, foldN, report_se, False)
    

def Wilcoxon(res, statistics, **argkw):
    res1, res2 = [], []
    for ri in split_by_iterations(res):
        stats = statistics(ri, **argkw)
        if len(stats) != 2:
            raise TypeError, "Wilcoxon compares two classifiers, no more, no less"
        res1.append(stats[0])
        res2.append(stats[1])
    return statc.wilcoxont(res1, res2)

def rank_difference(res, statistics, **argkw):
    if not res.results:
        raise TypeError, "no experiments"

    k = len(res.results[0].classes)
    if k < 2:
        raise TypeError, "nothing to compare (less than two classifiers given)"
    if k == 2:
        return apply(Wilcoxon, (res, statistics), argkw)
    else:
        return apply(Friedman, (res, statistics), argkw)


@deprecated_keywords({"res": "test_results",
                      "classIndex": "class_index",
                      "unweighted": "ignore_weights"})
def confusion_matrices(test_results, class_index=-1,
                       ignore_weights=False, cutoff=.5):
    """
    Return confusion matrices for test_results.

    :param test_results: test results
    :param class_index: index of class value for which the confusion matrices
                        are to be computed.
    :param ignore_weights: ignore instance weights.
    :params cutoff: cutoff for probability

    :rtype: list of :obj:`ConfusionMatrix`
    """
    tfpns = [ConfusionMatrix() for _ in range(test_results.number_of_learners)]
    
    if class_index < 0:
        numberOfClasses = len(test_results.class_values)
        if class_index < -1 or numberOfClasses > 2:
            cm = [[[0.] * numberOfClasses for _ in range(numberOfClasses)]
                  for _ in range(test_results.number_of_learners)]
            if ignore_weights or not test_results.weights:
                for tex in test_results.results:
                    trueClass = int(tex.actual_class)
                    for li, pred in enumerate(tex.classes):
                        predClass = int(pred)
                        if predClass < numberOfClasses:
                            cm[li][trueClass][predClass] += 1
            else:
                for tex in test_results.results:
                    trueClass = int(tex.actual_class)
                    for li, pred in tex.classes:
                        predClass = int(pred)
                        if predClass < numberOfClasses:
                            cm[li][trueClass][predClass] += tex.weight
            return cm
            
        elif test_results.baseClass >= 0:
            class_index = test_results.baseClass
        else:
            class_index = 1

    if cutoff != .5:
        if ignore_weights or not test_results.weights:
            for lr in test_results.results:
                isPositive=(lr.actual_class==class_index)
                for i in range(test_results.number_of_learners):
                    tfpns[i].addTFPosNeg(lr.probabilities[i][class_index] >
                                         cutoff, isPositive)
        else:
            for lr in test_results.results:
                isPositive=(lr.actual_class == class_index)
                for i in range(test_results.number_of_learners):
                    tfpns[i].addTFPosNeg(lr.probabilities[i][class_index] > 
                                         cutoff, isPositive, lr.weight)
    else:
        if ignore_weights or not test_results.weights:
            for lr in test_results.results:
                isPositive = (lr.actual_class == class_index)
                for i in range(test_results.number_of_learners):
                    tfpns[i].addTFPosNeg(lr.classes[i] == class_index,
                                         isPositive)
        else:
            for lr in test_results.results:
                isPositive = (lr.actual_class == class_index)
                for i in range(test_results.number_of_learners):
                    tfpns[i].addTFPosNeg(lr.classes[i] == class_index,
                                         isPositive, lr.weight)
    return tfpns


# obsolete (renamed)
compute_confusion_matrices = confusion_matrices


@deprecated_keywords({"confusionMatrix": "confusion_matrix"})
def confusion_chi_square(confusion_matrix):
    """
    Return chi square statistic of the confusion matrix
    (higher value indicates that prediction is not by chance).
    """
    if isinstance(confusion_matrix, ConfusionMatrix) or \
       not isinstance(confusion_matrix[1], list):
        return _confusion_chi_square(confusion_matrix)
    else:
        return map(_confusion_chi_square, confusion_matrix)

def _confusion_chi_square(confusion_matrix):
    if isinstance(confusion_matrix, ConfusionMatrix):
        c = confusion_matrix
        confusion_matrix = [[c.TP, c.FN], [c.FP, c.TN]]
    dim = len(confusion_matrix)
    rowPriors = [sum(r) for r in confusion_matrix]
    colPriors = [sum(r[i] for r in confusion_matrix) for i in range(dim)]
    total = sum(rowPriors)
    rowPriors = [r / total for r in rowPriors]
    colPriors = [r / total for r in colPriors]
    ss = 0
    for ri, row in enumerate(confusion_matrix):
        for ci, o in enumerate(row):
            e = total * rowPriors[ri] * colPriors[ci]
            if not e:
                return -1, -1, -1
            ss += (o - e)**2 / e
    df = (dim - 1)**2
    return ss, df, statc.chisqprob(ss, df)

@deprecated_keywords({"confm": "confusion_matrix"})
def sens(confusion_matrix):
    """
    Return `sensitivity
    <http://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_ (proportion
    of actual positives which are correctly identified as such).
    """
    if type(confusion_matrix) == list:
        return [sens(cm) for cm in confusion_matrix]
    else:
        tot = confusion_matrix.TP+confusion_matrix.FN
        if tot < 1e-6:
            import warnings
            warnings.warn("Can't compute sensitivity: one or both classes have no instances")
            return -1

        return confusion_matrix.TP / tot


@deprecated_keywords({"confm": "confusion_matrix"})
def recall(confusion_matrix):
    """
    Return `recall <http://en.wikipedia.org/wiki/Precision_and_recall>`_
    (fraction of relevant instances that are retrieved).
    """
    return sens(confusion_matrix)


@deprecated_keywords({"confm": "confusion_matrix"})
def spec(confusion_matrix):
    """
    Return `specificity
    <http://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_
    (proportion of negatives which are correctly identified).
    """
    if type(confusion_matrix) == list:
        return [spec(cm) for cm in confusion_matrix]
    else:
        tot = confusion_matrix.FP+confusion_matrix.TN
        if tot < 1e-6:
            import warnings
            warnings.warn("Can't compute specificity: one or both classes have no instances")
            return -1
        return confusion_matrix.TN / tot


@deprecated_keywords({"confm": "confusion_matrix"})
def PPV(confusion_matrix):
    """
    Return `positive predictive value
    <http://en.wikipedia.org/wiki/Positive_predictive_value>`_ (proportion of
    subjects with positive test results who are correctly diagnosed)."""
    if type(confusion_matrix) == list:
        return [PPV(cm) for cm in confusion_matrix]
    else:
        tot = confusion_matrix.TP + confusion_matrix.FP
        if tot < 1e-6:
            import warnings
            warnings.warn("Can't compute PPV: one or both classes have no instances")
            return -1
        return confusion_matrix.TP/tot


@deprecated_keywords({"confm": "confusion_matrix"})
def precision(confusion_matrix):
    """
    Return `precision <http://en.wikipedia.org/wiki/Precision_and_recall>`_
    (retrieved instances that are relevant).
    """
    return PPV(confusion_matrix)

@deprecated_keywords({"confm": "confusion_matrix"})
def NPV(confusion_matrix):
    """
    Return `negative predictive value
    <http://en.wikipedia.org/wiki/Negative_predictive_value>`_ (proportion of
    subjects with a negative test result who are correctly diagnosed).
     """
    if type(confusion_matrix) == list:
        return [NPV(cm) for cm in confusion_matrix]
    else:
        tot = confusion_matrix.FN + confusion_matrix.TN
        if tot < 1e-6:
            import warnings
            warnings.warn("Can't compute NPV: one or both classes have no instances")
            return -1
        return confusion_matrix.TN / tot

@deprecated_keywords({"confm": "confusion_matrix"})
def F1(confusion_matrix):
    """
    Return `F1 score <http://en.wikipedia.org/wiki/F1_score>`_
    (harmonic mean of precision and recall).
    """
    if type(confusion_matrix) == list:
        return [F1(cm) for cm in confusion_matrix]
    else:
        p = precision(confusion_matrix)
        r = recall(confusion_matrix)
        if p + r > 0:
            return 2. * p * r / (p + r)
        else:
            import warnings
            warnings.warn("Can't compute F1: P + R is zero or not defined")
            return -1


@deprecated_keywords({"confm": "confusion_matrix"})
def Falpha(confusion_matrix, alpha=1.0):
    """
    Return the alpha-mean of precision and recall over the given confusion
    matrix.
    """
    if type(confusion_matrix) == list:
        return [Falpha(cm, alpha=alpha) for cm in confusion_matrix]
    else:
        p = precision(confusion_matrix)
        r = recall(confusion_matrix)
        return (1. + alpha) * p * r / (alpha * p + r)


@deprecated_keywords({"confm": "confusion_matrix"})
def MCC(confusion_matrix):
    """
    Return `Matthew correlation coefficient
    <http://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_
    (correlation coefficient between the observed and predicted binary
    classifications).
    """
    # code by Boris Gorelik
    if type(confusion_matrix) == list:
        return [MCC(cm) for cm in confusion_matrix]
    else:
        truePositive = confusion_matrix.TP
        trueNegative = confusion_matrix.TN
        falsePositive = confusion_matrix.FP
        falseNegative = confusion_matrix.FN
          
        try:   
            r = (((truePositive * trueNegative) -
                  (falsePositive * falseNegative)) /
                 math.sqrt((truePositive + falsePositive) *
                           (truePositive + falseNegative) * 
                           (trueNegative + falsePositive) * 
                           (trueNegative + falseNegative)))
        except ZeroDivisionError:
            # Zero difision occurs when there is either no true positives 
            # or no true negatives i.e. the problem contains only one 
            # type of classes.
            import warnings
            warnings.warn("Can't compute MCC: TP or TN is zero or not defined")
            r = None

    return r


@deprecated_keywords({"bIsListOfMatrices": "b_is_list_of_matrices"})
def scotts_pi(confusion_matrix, b_is_list_of_matrices=True):
   """Compute Scott's Pi for measuring inter-rater agreement for nominal data

   http://en.wikipedia.org/wiki/Scott%27s_Pi
   Scott's Pi is a statistic for measuring inter-rater reliability for nominal
   raters.

   @param confusion_matrix: confusion matrix, or list of confusion matrices.
                            To obtain non-binary confusion matrix, call
                            Orange.evaluation.scoring.confusion_matrices
                            and set the class_index parameter to -2.
   @param b_is_list_of_matrices: specifies whether confm is list of matrices.
                            This function needs to operate on non-binary
                            confusion matrices, which are represented by python
                            lists, therefore one needs a way to distinguish
                            between a single matrix and list of matrices
   """

   if b_is_list_of_matrices:
       try:
           return [scotts_pi(cm, b_is_list_of_matrices=False)
                   for cm in confusion_matrix]
       except TypeError:
           # Nevermind the parameter, maybe this is a "conventional" binary
           # confusion matrix and bIsListOfMatrices was specified by mistake
           return scottsPiSingle(confusion_matrix, bIsListOfMatrices=False)
   else:
       if isinstance(confusion_matrix, ConfusionMatrix):
           confusion_matrix = numpy.array([[confusion_matrix.TP,
               confusion_matrix.FN], [confusion_matrix.FP,
               confusion_matrix.TN]], dtype=float)
       else:
           confusion_matrix = numpy.array(confusion_matrix, dtype=float)

       marginalSumOfRows = numpy.sum(confusion_matrix, axis=0)
       marginalSumOfColumns = numpy.sum(confusion_matrix, axis=1)
       jointProportion = (marginalSumOfColumns + marginalSumOfRows) / \
                           (2. * numpy.sum(confusion_matrix))
       # In the eq. above, 2. is what the Wikipedia page calls
       # the number of annotators. Here we have two annotators:
       # the observed (true) labels (annotations) and the predicted by
       # the learners.

       prExpected = numpy.sum(jointProportion ** 2)
       prActual = numpy.sum(numpy.diag(confusion_matrix)) / \
                  numpy.sum(confusion_matrix)

       ret = (prActual - prExpected) / (1.0 - prExpected)
       return ret

@deprecated_keywords({"classIndex": "class_index",
                      "unweighted": "ignore_weights"})
def AUCWilcoxon(res, class_index=-1, ignore_weights=False, **argkw):
    """Compute the area under ROC (AUC) and its standard error using
    Wilcoxon's approach proposed by Hanley and McNeal (1982). If 
    :obj:`class_index` is not specified, the first class is used as
    "the positive" and others are negative. The result is a list of
    tuples (aROC, standard error).

    If test results consist of multiple folds, you need to split them using
    :obj:`split_by_iterations` and perform this test on each fold separately.
    """
    useweights = res.weights and not ignore_weights
    problists, tots = corn.computeROCCumulative(res, class_index, useweights)

    results = []

    totPos, totNeg = tots[1], tots[0]
    N = totPos + totNeg
    for plist in problists:
        highPos, lowNeg = totPos, 0.
        W, Q1, Q2 = 0., 0., 0.
        for prob in plist:
            thisPos, thisNeg = prob[1][1], prob[1][0]
            highPos -= thisPos
            W += thisNeg * (highPos + thisPos / 2.)
            Q2 += thisPos * (lowNeg**2  + lowNeg*thisNeg  + thisNeg**2 / 3.)
            Q1 += thisNeg * (highPos**2 + highPos*thisPos + thisPos**2 / 3.)

            lowNeg += thisNeg

        W  /= (totPos*totNeg)
        Q1 /= (totNeg*totPos**2)
        Q2 /= (totPos*totNeg**2)

        SE = math.sqrt((W * (1 - W) + (totPos - 1) * (Q1 - W**2) +
                       (totNeg - 1) * (Q2 - W**2)) / (totPos * totNeg))
        results.append((W, SE))
    return results

AROC = AUCWilcoxon # for backward compatibility, AROC is obsolete


@deprecated_keywords({"classIndex": "class_index",
                      "unweighted": "ignore_weights"})
def compare_2_AUCs(res, lrn1, lrn2, class_index=-1,
                   ignore_weights=False, **argkw):
    return corn.compare2ROCs(res, lrn1, lrn2, class_index,
                             res.weights and not ignore_weights)

# for backward compatibility, compare_2_AROCs is obsolete
compare_2_AROCs = compare_2_AUCs 


@deprecated_keywords({"classIndex": "class_index"})
def compute_ROC(res, class_index=-1):
    """Compute a ROC curve as a list of (x, y) tuples, where x is 
    1-specificity and y is sensitivity.
    """
    problists, tots = corn.computeROCCumulative(res, class_index)

    results = []
    totPos, totNeg = tots[1], tots[0]

    for plist in problists:
        curve=[(1., 1.)]
        TP, TN = totPos, 0.
        FN, FP = 0., totNeg
        for prob in plist:
            thisPos, thisNeg = prob[1][1], prob[1][0]
            # thisPos go from TP to FN
            TP -= thisPos
            FN += thisPos
            # thisNeg go from FP to TN
            TN += thisNeg
            FP -= thisNeg

            sens = TP / (TP + FN)
            spec = TN / (FP + TN)
            curve.append((1 - spec, sens))
        results.append(curve)

    return results    

## TC's implementation of algorithms, taken from:
## T Fawcett: ROC Graphs: Notes and Practical Considerations for
## Data Mining Researchers, submitted to KDD Journal.
def ROC_slope((P1x, P1y, P1fscore), (P2x, P2y, P2fscore)):
    if P1x == P2x:
        return 1e300
    return (P1y - P2y) / (P1x - P2x)


@deprecated_keywords({"keepConcavities": "keep_concavities"})
def ROC_add_point(P, R, keep_concavities=1):
    if keep_concavities:
        R.append(P)
    else:
        while True:
            if len(R) < 2:
                R.append(P)
                return R
            else:
                T = R.pop()
                T2 = R[-1]
                if ROC_slope(T2, T) > ROC_slope(T, P):
                    R.append(T)
                    R.append(P)
                    return R
    return R


@deprecated_keywords({"classIndex": "class_index",
                      "keepConcavities": "keep_concavities"})
def TC_compute_ROC(res, class_index=-1, keep_concavities=1):
    problists, tots = corn.computeROCCumulative(res, class_index)

    results = []
    P, N = tots[1], tots[0]

    for plist in problists:
        ## corn gives an increasing by scores list, we need a decreasing
        plist.reverse()
        TP = 0.
        FP = 0.
        curve=[]
        fPrev = 10e300 # "infinity" score at 0., 0.
        for prob in plist:
            f = prob[0]
            if f != fPrev:
                if P:
                    tpr = TP / P
                else:
                    tpr = 0.
                if N:
                    fpr = FP / N
                else:
                    fpr = 0.
                curve = ROC_add_point((fpr, tpr, fPrev), curve,
                                      keep_concavities)
                fPrev = f
            thisPos, thisNeg = prob[1][1], prob[1][0]
            TP += thisPos
            FP += thisNeg
        if P:
            tpr = TP / P
        else:
            tpr = 0.
        if N:
            fpr = FP / N
        else:
            fpr = 0.
        curve = ROC_add_point((fpr, tpr, f), curve, keep_concavities) ## ugly
        results.append(curve)

    return results

## returns a list of points at the intersection of the tangential
## iso-performance line and the given ROC curve
## for given values of FPcost, FNcost and pval
def TC_best_thresholds_on_ROC_curve(FPcost, FNcost, pval, curve):
    m = (FPcost * (1. - pval)) / (FNcost * pval)

    ## put the iso-performance line in point (0., 1.)
    x0, y0 = (0., 1.)
    x1, y1 = (1., 1. + m)
    d01 = math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))

    ## calculate and find the closest point to the line
    firstp = 1
    mind = 0.
    a = x0 * y1 - x1 * y0
    closestPoints = []
    for (x, y, fscore) in curve:
        d = ((y0 - y1) * x + (x1 - x0) * y + a) / d01
        d = abs(d)
        if firstp or d < mind:
            mind, firstp = d, 0
            closestPoints = [(x, y, fscore)]
        else:
            if abs(d - mind) <= 0.0001: ## close enough
                closestPoints.append( (x, y, fscore) )
    return closestPoints          

def frange(start, end=None, inc=None):
    """A range function, that does accept float increments..."""

    if end is None:
        end = start + 0.
        start = 0.

    if inc is None or inc == 0:
        inc = 1.

    L = [start]
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            L.append(end)
            break
        elif inc < 0 and next <= end:
            L.append(end)
            break
        L.append(next)
        
    return L

## input ROCcurves are of form [ROCcurves1, ROCcurves2, ... ROCcurvesN],
## where ROCcurvesX is a set of ROC curves,
## where a (one) ROC curve is a set of (FP, TP) points
##
## for each (sub)set of input ROC curves
## returns the average ROC curve and an array of (vertical) standard deviations
@deprecated_keywords({"ROCcurves": "roc_curves"})
def TC_vertical_average_ROC(roc_curves, samples = 10):
    def INTERPOLATE((P1x, P1y, P1fscore), (P2x, P2y, P2fscore), X):
        if (P1x == P2x) or P1x < X > P2x or P1x > X < P2x:
            raise ValueError, "assumptions for interpolation are not met: P1 = %f,%f P2 = %f,%f X = %f" % (P1x, P1y, P2x, P2y, X)
        dx = float(P2x) - float(P1x)
        dy = float(P2y) - float(P1y)
        m = dy / dx
        return P1y + m * (X - P1x)

    def TP_FOR_FP(FPsample, ROC, npts):
        i = 0
        while i < npts - 1:
            (fp, _, _) = ROC[i + 1]
            if fp <= FPsample:
                i += 1
            else:
                break
        (fp, tp, _) = ROC[i]
        if fp == FPsample:
            return tp
        elif fp < FPsample and i + 1 < len(ROC):
            return INTERPOLATE(ROC[i], ROC[i + 1], FPsample)
        elif fp < FPsample and i + 1 == len(ROC): # return the last
            return ROC[i][1]
        raise ValueError, "cannot compute: TP_FOR_FP in TC_vertical_average_ROC"
        #return 0.

    average = []
    stdev = []
    for ROCS in roc_curves:
        npts = []
        for c in ROCS:
            npts.append(len(c))
        nrocs = len(ROCS)

        TPavg = []
        TPstd = []
        for FPsample in frange(0., 1., 1. / samples):
            TPsum = []
            for i in range(nrocs):
                ##TPsum = TPsum + TP_FOR_FP(FPsample, ROCS[i], npts[i])
                TPsum.append(TP_FOR_FP(FPsample, ROCS[i], npts[i]))
            TPavg.append((FPsample, statc.mean(TPsum)))
            if len(TPsum) > 1:
                stdv = statc.std(TPsum)
            else:
                stdv = 0.
            TPstd.append(stdv)

        average.append(TPavg)
        stdev.append(TPstd)

    return average, stdev

## input ROCcurves are of form [ROCcurves1, ROCcurves2, ... ROCcurvesN],
## where ROCcurvesX is a set of ROC curves,
## where a (one) ROC curve is a set of (FP, TP) points
##
## for each (sub)set of input ROC curves
## returns the average ROC curve, an array of vertical standard deviations and an array of horizontal standard deviations
@deprecated_keywords({"ROCcurves": "roc_curves"})
def TC_threshold_average_ROC(roc_curves, samples=10):
    def POINT_AT_THRESH(ROC, npts, thresh):
        i = 0
        while i < npts - 1:
            (px, py, pfscore) = ROC[i]
            if pfscore > thresh:
                i += 1
            else:
                break
        return ROC[i]

    average = []
    stdevV = []
    stdevH = []
    for ROCS in roc_curves:
        npts = []
        for c in ROCS:
            npts.append(len(c))
        nrocs = len(ROCS)

        T = []
        for c in ROCS:
            for (px, py, pfscore) in c:
##                try:
##                    T.index(pfscore)
##                except:
                T.append(pfscore)
        T.sort()
        T.reverse() ## ugly

        TPavg = []
        TPstdV = []
        TPstdH = []
        for tidx in frange(0, (len(T) - 1.), float(len(T)) / samples):
            FPsum = []
            TPsum = []
            for i in range(nrocs):
                (fp, tp, _) = POINT_AT_THRESH(ROCS[i], npts[i], T[int(tidx)])
                FPsum.append(fp)
                TPsum.append(tp)
            TPavg.append((statc.mean(FPsum), statc.mean(TPsum)))
            ## vertical standard deviation
            if len(TPsum) > 1:
                stdv = statc.std(TPsum)
            else:
                stdv = 0.
            TPstdV.append( stdv )
            ## horizontal standard deviation
            if len(FPsum) > 1:
                stdh = statc.std(FPsum)
            else:
                stdh = 0.
            TPstdH.append( stdh )

        average.append(TPavg)
        stdevV.append(TPstdV)
        stdevH.append(TPstdH)

    return average, stdevV, stdevH

## Calibration Curve
## returns an array of (curve, yesClassPredictions, noClassPredictions)
## elements, where:
##  - curve is an array of points (x, y) on the calibration curve
##  - yesClassRugPoints is an array of (x, 1) points
##  - noClassRugPoints is an array of (x, 0) points
@deprecated_keywords({"classIndex": "class_index"})
def compute_calibration_curve(res, class_index=-1):
    ## merge multiple iterations into one
    mres = Orange.evaluation.testing.ExperimentResults(1, res.classifier_names,
        res.class_values, res.weights, classifiers=res.classifiers,
        loaded=res.loaded, test_type=res.test_type, labels=res.labels)
    for te in res.results:
        mres.results.append(te)

    problists, tots = corn.computeROCCumulative(mres, class_index)

    results = []

    bins = 10 ## divide interval between 0. and 1. into N bins

    for plist in problists:
        yesClassRugPoints = [] 
        noClassRugPoints = []

        yesBinsVals = [0] * bins
        noBinsVals = [0] * bins
        for (f, (thisNeg, thisPos)) in plist:
            yesClassRugPoints.append((f, thisPos)) # 1.
            noClassRugPoints.append((f, thisNeg)) # 1.

            index = int(f * bins )
            index = min(index, bins - 1) ## just in case for value 1.
            yesBinsVals[index] += thisPos
            noBinsVals[index] += thisNeg

        curve = []
        for cn in range(bins):
            f = float(cn * 1. / bins) + (1. / 2. / bins)
            yesVal = yesBinsVals[cn]
            noVal = noBinsVals[cn]
            allVal = yesVal + noVal
            if allVal == 0.: continue
            y = float(yesVal) / float(allVal)
            curve.append((f,  y))

        ## smooth the curve
        maxnPoints = 100
        if len(curve) >= 3:
#            loessCurve = statc.loess(curve, -3, 0.6)
            loessCurve = statc.loess(curve, maxnPoints, 0.5, 3)
        else:
            loessCurve = curve
        clen = len(loessCurve)
        if clen > maxnPoints:
            df = clen / maxnPoints
            if df < 1: df = 1
            curve = [loessCurve[i]  for i in range(0, clen, df)]
        else:
            curve = loessCurve
        ## remove the third value (variance of epsilon?) that suddenly
        ## appeared in the output of the statc.loess function
        curve = [c[:2] for c in curve]
        results.append((curve, yesClassRugPoints, noClassRugPoints))

    return results


## Lift Curve
## returns an array of curve elements, where:
##  - curve is an array of points ((TP + FP) / (P + N), TP / P, (th, FP / N))
##    on the Lift Curve
@deprecated_keywords({"classIndex": "class_index"})
def compute_lift_curve(res, class_index=-1):
    ## merge multiple iterations into one
    mres = Orange.evaluation.testing.ExperimentResults(1, res.classifier_names,
        res.class_values, res.weights, classifiers=res.classifiers,
        loaded=res.loaded, test_type=res.test_type, labels=res.labels)
    for te in res.results:
        mres.results.append(te)

    problists, tots = corn.computeROCCumulative(mres, class_index)

    results = []
    P, N = tots[1], tots[0]
    for plist in problists:
        ## corn gives an increasing by scores list, we need a decreasing
        plist.reverse()
        TP = 0.
        FP = 0.
        curve = [(0., 0., (10e300, 0.))]
        for (f, (thisNeg, thisPos)) in plist:
            TP += thisPos
            FP += thisNeg
            curve.append(((TP + FP) / (P + N), TP, (f, FP / (N or 1))))
        results.append(curve)

    return P, N, results


class CDT:
  """Store the number of concordant (C), discordant (D) and tied (T) pairs."""
  def __init__(self, C=0., D=0., T=0.):
    self.C, self.D, self.T = C, D, T
   
def is_CDT_empty(cdt):
    return cdt.C + cdt.D + cdt.T < 1e-20


@deprecated_keywords({"classIndex": "class_index",
                      "unweighted": "ignore_weights"})
def compute_CDT(res, class_index=-1, ignore_weights=False, **argkw):
    """Obsolete, don't use."""
    if class_index < 0:
        if res.baseClass >= 0:
            class_index = res.baseClass
        else:
            class_index = 1
            
    useweights = res.weights and not ignore_weights
    weightByClasses = argkw.get("weightByClasses", True)

    if res.number_of_iterations>1:
        CDTs = [CDT() for _ in range(res.number_of_learners)]
        iterationExperiments = split_by_iterations(res)
        for exp in iterationExperiments:
            expCDTs = corn.computeCDT(exp, class_index, useweights)
            for i in range(len(CDTs)):
                CDTs[i].C += expCDTs[i].C
                CDTs[i].D += expCDTs[i].D
                CDTs[i].T += expCDTs[i].T
        for i in range(res.number_of_learners):
            if is_CDT_empty(CDTs[0]):
                return corn.computeCDT(res, class_index, useweights)
        
        return CDTs
    else:
        return corn.computeCDT(res, class_index, useweights)

# Backward compatibility
def replace_use_weights(fun):
    if environ.orange_no_deprecated_members:
        return fun

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        use_weights = kwargs.pop("useWeights", None)
        if use_weights is not None:
            deprecation_warning("useWeights", "ignore_weights")
            kwargs["ignore_weights"] = not use_weights
        return fun(*args, **kwargs)
    return wrapped

class AUC(list):
    """
    Compute the area under ROC curve given a set of experimental results.
    For multivalued class problems, return the result of
    :obj:`by_weighted_pairs`.
    If testing consisted of multiple folds, each fold is scored and the
    average score is returned. If a fold contains only instances with the
    same class value, folds will be merged.

    :param test_results: test results to score
    :param ignore_weights: ignore instance weights when calculating score
    :param method: DEPRECATED, call the appropriate method directly.
    """

    @replace_use_weights
    def __init__(self, test_results=None, method=0, ignore_weights=False):

        super(AUC, self).__init__()

        self.ignore_weights=ignore_weights
        self.method=method

        if test_results is not None:
            self.__call__(test_results)

    def __call__(self, test_results):
        if len(test_results.class_values) < 2:
            raise ValueError("Cannot compute AUC on a single-class problem")
        elif len(test_results.class_values) == 2:
            self._compute_for_binary_class(test_results)
        else:
            self._compute_for_multi_value_class(test_results, self.method)

    @classmethod
    def by_weighted_pairs(cls, res, ignore_weights=False):
        """
        Compute AUC for each pair of classes (ignoring instances of all other
        classes) and average the results, weighting them by the number of
        pairs of instances from these two classes (e.g. by the product of
        probabilities of the two classes). AUC computed in this way still
        behaves as the concordance index, e.g., gives the probability that two
        randomly chosen instances from different classes will be correctly
        recognized (if the classifier knows from which two classes the
        instances came).
        """
        auc = AUC(ignore_weights=ignore_weights)
        auc._compute_for_multi_value_class(res, method=cls.ByWeightedPairs)
        return auc

    @classmethod
    def by_pairs(cls, res, ignore_weights=False):
        """
        Similar to by_weighted_pairs, except that the average over class pairs
        is not weighted. This AUC is, like the binary version, independent of
        class distributions, but it is not related to the concordance index
        any more.
        """
        auc = AUC(ignore_weights=ignore_weights)
        auc._compute_for_multi_value_class(res, method=cls.ByPairs)
        return auc

    @classmethod
    def weighted_one_against_all(cls, res, ignore_weights=False):
        """
        For each class, it computes AUC for this class against all others (that
        is, treating other classes as one class). The AUCs are then averaged by
        the class probabilities. This is related to the concordance index in
        which we test the classifier's (average) capability of distinguishing
        the instances from a specified class from those that come from other
        classes.
        Unlike the binary AUC, the measure is not independent of class
        distributions.
        """
        auc = AUC(ignore_weights=ignore_weights)
        auc._compute_for_multi_value_class(res,
            method=cls.WeightedOneAgainstAll)
        return auc

    @classmethod
    def one_against_all(cls, res, ignore_weights=False):
        """
        Similar to weighted_one_against_all, except that the average
        is not weighted.
        """
        auc = AUC(ignore_weights=ignore_weights)
        auc._compute_for_multi_value_class(res, method=cls.OneAgainstAll)
        return auc

    @classmethod
    def single_class(cls, res, class_index=-1, ignore_weights=False):
        """
        Compute AUC where the class with the given class_index is singled
        out and all other classes are treated as a single class.
        """
        if class_index < 0:
            if res.base_class >= 0:
                class_index = res.base_class
            else:
                class_index = 1

        auc = AUC(ignore_weights=ignore_weights)
        auc._compute_for_single_class(res, class_index)
        return auc

    @classmethod
    def pair(cls, res, class_index1, class_index2, ignore_weights=False):
        """
        Computes AUC between a pair of classes, ignoring instances from all
        other classes.
        """
        auc = AUC(ignore_weights=ignore_weights)
        auc._compute_for_pair_of_classes(res, class_index1, class_index2)
        return auc

    @classmethod
    def matrix(cls, res, ignore_weights=False):
        """
        Compute a (lower diagonal) matrix with AUCs for all pairs of classes.
        If there are empty classes, the corresponding elements in the matrix
        are -1.
        """
        auc = AUC(ignore_weights=ignore_weights)
        auc._compute_matrix(res)
        return auc

    def _compute_for_binary_class(self, res):
        """AUC for binary classification problems"""
        if res.number_of_iterations > 1:
            return self._compute_for_multiple_folds(
                self._compute_one_class_against_all,
                split_by_iterations(res),
                (-1, res, res.number_of_iterations))
        else:
            auc, _ = self._compute_one_class_against_all(res, -1)
            self[:] = auc
            return self

    def _compute_for_multi_value_class(self, res, method=0):
        """AUC for multiclass classification problems"""
        numberOfClasses = len(res.class_values)

        if res.number_of_iterations > 1:
            iterations = split_by_iterations(res)
            all_ite = res
        else:
            iterations = [res]
            all_ite = None

        # by pairs
        sum_aucs = [0.] * res.number_of_learners
        usefulClassPairs = 0.

        prob = None
        if method in [self.ByWeightedPairs, self.WeightedOneAgainstAll]:
            prob = class_probabilities_from_res(res)

        if method in [self.ByWeightedPairs, self.ByPairs]:
            for classIndex1 in range(numberOfClasses):
                for classIndex2 in range(classIndex1):
                    subsum_aucs = self._compute_for_multiple_folds(
                        self._compute_one_class_against_another, iterations,
                        (classIndex1, classIndex2, all_ite,
                        res.number_of_iterations))
                    if subsum_aucs:
                        if method == self.ByWeightedPairs:
                            p_ij = prob[classIndex1] * prob[classIndex2]
                            subsum_aucs = [x * p_ij  for x in subsum_aucs]
                            usefulClassPairs += p_ij
                        else:
                            usefulClassPairs += 1
                        sum_aucs = map(add, sum_aucs, subsum_aucs)
        else:
            for classIndex in range(numberOfClasses):
                subsum_aucs = self._compute_for_multiple_folds(
                    self._compute_one_class_against_all,
                    iterations, (classIndex, all_ite,
                                 res.number_of_iterations))
                if subsum_aucs:
                    if method == self.ByWeightedPairs:
                        p_i = prob[classIndex]
                        subsum_aucs = [x * p_i  for x in subsum_aucs]
                        usefulClassPairs += p_i
                    else:
                        usefulClassPairs += 1
                    sum_aucs = map(add, sum_aucs, subsum_aucs)

        if usefulClassPairs > 0:
            sum_aucs = [x/usefulClassPairs for x in sum_aucs]

        self[:] = sum_aucs
        return self

    # computes the average AUC over folds using "AUCcomputer" (AUC_i or AUC_ij)
    # it returns the sum of what is returned by the computer,
    # unless at a certain fold the computer has to resort to computing
    # over all folds or even this failed;
    # in these cases the result is returned immediately
    @deprecated_keywords({"AUCcomputer": "auc_computer",
                          "computerArgs": "computer_args"})
    def _compute_for_multiple_folds(self, auc_computer, iterations,
                                 computer_args):
        """Compute the average AUC over folds using :obj:`auc_computer`."""
        subsum_aucs = [0.] * iterations[0].number_of_learners
        for ite in iterations:
            aucs, foldsUsed = auc_computer(*(ite, ) + computer_args)
            if not aucs:
                return None
            if not foldsUsed:
                return aucs
            subsum_aucs = map(add, subsum_aucs, aucs)
        self[:] = subsum_aucs
        return self

    # Computes AUC
    # in multivalued class problem, AUC is computed as one against all
    # results over folds are averages
    # if some folds examples from one class only, the folds are merged
    def _compute_for_single_class(self, res, class_index):
        if res.number_of_iterations > 1:
            self._compute_for_multiple_folds(
                self._compute_one_class_against_all, split_by_iterations(res),
                (class_index, res, res.number_of_iterations))
        else:
            self._compute_one_class_against_all(res, class_index)

    # Computes AUC for a pair of classes (as if there were no other classes)
    # results over folds are averages
    # if some folds have examples from one class only, the folds are merged
    def _compute_for_pair_of_classes(self, res, class_index1, class_index2):
        if res.number_of_iterations > 1:
            self._compute_for_multiple_folds(
                self._compute_one_class_against_another,
                split_by_iterations(res),
                (class_index1, class_index2, res, res.number_of_iterations))
        else:
            self._compute_one_class_against_another(res, class_index1,
                                                    class_index2)

    # computes AUC between class i and the other classes
    # (treating them as the same class)
    @deprecated_keywords({"classIndex": "class_index",
                          "divideByIfIte": "divide_by_if_ite"})
    def _compute_one_class_against_all(self, ite, class_index, all_ite=None,
                                      divide_by_if_ite=1.0):
        """Compute AUC between class i and all the other classes)"""
        return self._compute_auc(corn.computeCDT, ite, all_ite,
            divide_by_if_ite, (class_index, not self.ignore_weights))


    # computes AUC between classes i and j as if there are no other classes
    def _compute_one_class_against_another(
        self, ite, class_index1, class_index2,
        all_ite=None, divide_by_if_ite=1.):
        """
        Compute AUC between classes i and j as if there are no other classes.
        """
        return self._compute_auc(corn.computeCDTPair, ite,
            all_ite, divide_by_if_ite,
            (class_index1, class_index2, not self.ignore_weights))

    # computes AUC using a specified 'cdtComputer' function
    # It tries to compute AUCs from 'ite' (examples from a single iteration)
    # and, if C+D+T=0, from 'all_ite' (entire test set). In the former case,
    # the AUCs are divided by 'divideByIfIte'.
    # Additional flag is returned which is True in the former case,
    # or False in the latter.
    @deprecated_keywords({"cdt_computer": "cdtComputer",
                          "divideByIfIte": "divide_by_if_ite",
                          "computerArgs": "computer_args"})
    def _compute_auc(self, cdt_computer, ite, all_ite, divide_by_if_ite,
                     computer_args):
        """
        Compute AUC using a :obj:`cdt_computer`.
        """
        cdts = cdt_computer(*(ite, ) + computer_args)
        if not is_CDT_empty(cdts[0]):
            return [(cdt.C + cdt.T / 2) / (cdt.C + cdt.D + cdt.T) /
                    divide_by_if_ite for cdt in cdts], True

        if all_ite:
            cdts = cdt_computer(*(all_ite, ) + computer_args)
            if not is_CDT_empty(cdts[0]):
                return [(cdt.C + cdt.T / 2) / (cdt.C + cdt.D + cdt.T)
                        for cdt in cdts], False

        return False, False

    def _compute_matrix(self, res):
        numberOfClasses = len(res.class_values)
        number_of_learners = res.number_of_learners
        if res.number_of_iterations > 1:
            iterations, all_ite = split_by_iterations(res), res
        else:
            iterations, all_ite = [res], None
        aucs = [[[] for _ in range(numberOfClasses)]
                for _ in range(number_of_learners)]
        for classIndex1 in range(numberOfClasses):
            for classIndex2 in range(classIndex1):
                pair_aucs = self._compute_for_multiple_folds(
                    self._compute_one_class_against_another, iterations,
                    (classIndex1, classIndex2, all_ite,
                     res.number_of_iterations))
                if pair_aucs:
                    for lrn in range(number_of_learners):
                        aucs[lrn][classIndex1].append(pair_aucs[lrn])
                else:
                    for lrn in range(number_of_learners):
                        aucs[lrn][classIndex1].append(-1)
        self[:] = aucs
        return aucs

#Backward compatibility
AUC.ByWeightedPairs = 0
AUC.ByPairs = 1
AUC.WeightedOneAgainstAll = 2
AUC.OneAgainstAll = 3

@replace_use_weights
def AUC_binary(res, ignore_weights=False):
    auc = deprecated_function_name(AUC)(ignore_weights=ignore_weights)
    auc._compute_for_binary_class(res)
    return auc

@replace_use_weights
def AUC_multi(res, ignore_weights=False, method=0):
    auc = deprecated_function_name(AUC)(ignore_weights=ignore_weights,
        method=method)
    auc._compute_for_multi_value_class(res)
    return auc

def AUC_iterations(auc_computer, iterations, computer_args):
    auc = deprecated_function_name(AUC)()
    auc._compute_for_multiple_folds(auc_computer, iterations, computer_args)
    return auc

def AUC_x(cdtComputer, ite, all_ite, divide_by_if_ite, computer_args):
    auc = deprecated_function_name(AUC)()
    result = auc._compute_auc(cdtComputer, ite, all_ite, divide_by_if_ite,
                              computer_args)
    return result

@replace_use_weights
def AUC_i(ite, class_index, ignore_weights=False, all_ite=None,
          divide_by_if_ite=1.):
    auc = deprecated_function_name(AUC)()
    result = auc._compute_one_class_against_another(ite, class_index,
        all_ite=None, divide_by_if_ite=1.)
    return result


@replace_use_weights
def AUC_ij(ite, class_index1, class_index2, ignore_weights=False,
           all_ite=None, divide_by_if_ite=1.):
    auc = deprecated_function_name(AUC)(ignore_weights=ignore_weights)
    result = auc._compute_one_class_against_another(
        ite, class_index1, class_index2, all_ite=None, divide_by_if_ite=1.)
    return result




#AUC_binary = replace_use_weights(deprecated_function_name(AUC()._compute_for_binary_class))
#AUC_multi = replace_use_weights(deprecated_function_name(AUC._compute_for_multi_value_class))
#AUC_iterations = replace_use_weights(deprecated_function_name(AUC._compute_for_multiple_folds))
#AUC_x = replace_use_weights(deprecated_function_name(AUC._compute_auc))
#AUC_i = replace_use_weights(deprecated_function_name(AUC._compute_one_class_against_all))
#AUC_ij = replace_use_weights(deprecated_function_name(AUC._compute_one_class_against_another))

AUC_single = replace_use_weights(
             deprecated_keywords({"classIndex": "class_index"})(
             deprecated_function_name(AUC.single_class)))
AUC_pair = replace_use_weights(
           deprecated_keywords({"classIndex1": "class_index1",
                                "classIndex2": "class_index2"})(
           deprecated_function_name(AUC.pair)))
AUC_matrix = replace_use_weights(deprecated_function_name(AUC.matrix))


@deprecated_keywords({"unweighted": "ignore_weights"})
def McNemar(res, ignore_weights=False, **argkw):
    """
    Compute a triangular matrix with McNemar statistics for each pair of
    classifiers. The statistics is distributed by chi-square distribution with
    one degree of freedom; critical value for 5% significance is around 3.84.
    """
    nLearners = res.number_of_learners
    mcm = []
    for i in range(nLearners):
       mcm.append([0.] * res.number_of_learners)

    if not res.weights or ignore_weights:
        for i in res.results:
            actual = i.actual_class
            classes = i.classes
            for l1 in range(nLearners):
                for l2 in range(l1, nLearners):
                    if classes[l1] == actual:
                        if classes[l2] != actual:
                            mcm[l1][l2] += 1
                    elif classes[l2] == actual:
                        mcm[l2][l1] += 1
    else:
        for i in res.results:
            actual = i.actual_class
            classes = i.classes
            for l1 in range(nLearners):
                for l2 in range(l1, nLearners):
                    if classes[l1] == actual:
                        if classes[l2] != actual:
                            mcm[l1][l2] += i.weight
                    elif classes[l2] == actual:
                        mcm[l2][l1] += i.weight

    for l1 in range(nLearners):
        for l2 in range(l1, nLearners):
            su = mcm[l1][l2] + mcm[l2][l1]
            if su:
                mcm[l2][l1] = (abs(mcm[l1][l2] - mcm[l2][l1]) - 1)**2 / su
            else:
                mcm[l2][l1] = 0

    for l1 in range(nLearners):
        mcm[l1]=mcm[l1][:l1]

    return mcm


def McNemar_of_two(res, lrn1, lrn2, ignore_weights=False):
    """
    McNemar_of_two computes a McNemar statistics for a pair of classifier,
    specified by indices learner1 and learner2.
    """
    tf = ft = 0.
    if not res.weights or ignore_weights:
        for i in res.results:
            actual = i.actual_class
            if i.classes[lrn1] == actual:
                if i.classes[lrn2] != actual:
                    tf += i.weight
            elif i.classes[lrn2] == actual:
                    ft += i.weight
    else:
        for i in res.results:
            actual = i.actual_class
            if i.classes[lrn1] == actual:
                if i.classes[lrn2] != actual:
                    tf += 1.
            elif i.classes[lrn2] == actual:
                    ft += 1.

    su = tf + ft
    if su:
        return (abs(tf - ft) - 1)**2 / su
    else:
        return 0


def Friedman(res, stat=CA):
    """
    Compare classifiers with Friedman test, treating folds as different examles.
    Returns F, p and average ranks.
    """
    res_split = split_by_iterations(res)
    res = [stat(r) for r in res_split]
    
    N = len(res)
    k = len(res[0])
    sums = [0.] * k
    for r in res:
        ranks = [k - x + 1 for x in statc.rankdata(r)]
        if stat == Brier_score: # reverse ranks for Brier_score (lower better)
            ranks = [k + 1 - x for x in ranks]
        sums = map(add, ranks, sums)

    T = sum(x * x for x in sums)
    sums = [x / N for x in sums]

    F = 12. / (N * k * (k + 1)) * T  - 3 * N * (k + 1)

    return F, statc.chisqprob(F, k - 1), sums


def Wilcoxon_pairs(res, avgranks, stat=CA):
    """
    Return a triangular matrix, where element[i][j] stores significance of
    difference between the i-th and the j-th classifier, as computed by the
    Wilcoxon test. The element is positive if the i-th is better than the j-th,
    negative if it is worse, and 1 if they are equal.
    Arguments are ExperimentResults, average ranks (as returned by Friedman)
    and, optionally, a statistics; greater values should mean better results.
    """
    res_split = split_by_iterations(res)
    res = [stat(r) for r in res_split]

    k = len(res[0])
    bt = []
    for m1 in range(k):
        nl = []
        for m2 in range(m1 + 1, k):
            t, p = statc.wilcoxont([r[m1] for r in res], [r[m2] for r in res])
            if avgranks[m1]<avgranks[m2]:
                nl.append(p)
            elif avgranks[m2]<avgranks[m1]:
                nl.append(-p)
            else:
                nl.append(1)
        bt.append(nl)
    return bt


@deprecated_keywords({"allResults": "all_results",
                      "noConfidence": "no_confidence"})
def plot_learning_curve_learners(file, all_results, proportions, learners,
                                 no_confidence=0):
    plot_learning_curve(file, all_results, proportions,
        [Orange.misc.getobjectname(learners[i], "Learner %i" % i)
        for i in range(len(learners))], no_confidence)


@deprecated_keywords({"allResults": "all_results",
                      "noConfidence": "no_confidence"})
def plot_learning_curve(file, all_results, proportions, legend,
                        no_confidence=0):
    import types
    fopened = 0
    if type(file) == types.StringType:
        file = open(file, "wt")
        fopened = 1
        
    file.write("set yrange [0:1]\n")
    file.write("set xrange [%f:%f]\n" % (proportions[0], proportions[-1]))
    file.write("set multiplot\n\n")
    CAs = [CA(x, report_se=True) for x in all_results]

    file.write("plot \\\n")
    for i in range(len(legend) - 1):
        if not no_confidence:
            file.write("'-' title '' with yerrorbars pointtype %i,\\\n" % (i+1))
        file.write("'-' title '%s' with linespoints pointtype %i,\\\n" % (legend[i], i+1))
    if not no_confidence:
        file.write("'-' title '' with yerrorbars pointtype %i,\\\n" % (len(legend)))
    file.write("'-' title '%s' with linespoints pointtype %i\n" % (legend[-1], len(legend)))

    for i in range(len(legend)):
        if not no_confidence:
            for p in range(len(proportions)):
                file.write("%f\t%f\t%f\n" % (proportions[p], CAs[p][i][0], 1.96*CAs[p][i][1]))
            file.write("e\n\n")

        for p in range(len(proportions)):
            file.write("%f\t%f\n" % (proportions[p], CAs[p][i][0]))
        file.write("e\n\n")

    if fopened:
        file.close()


def print_single_ROC_curve_coordinates(file, curve):
    import types
    fopened=0
    if type(file)==types.StringType:
        file=open(file, "wt")
        fopened=1

    for coord in curve:
        file.write("%5.3f\t%5.3f\n" % tuple(coord))

    if fopened:
        file.close()


def plot_ROC_learners(file, curves, learners):
    plot_ROC(file, curves, [Orange.misc.getobjectname(learners[i], "Learner %i" % i) for i in range(len(learners))])
    
def plot_ROC(file, curves, legend):
    import types
    fopened=0
    if type(file)==types.StringType:
        file=open(file, "wt")
        fopened=1

    file.write("set yrange [0:1]\n")
    file.write("set xrange [0:1]\n")
    file.write("set multiplot\n\n")

    file.write("plot \\\n")
    for leg in legend:
        file.write("'-' title '%s' with lines,\\\n" % leg)
    file.write("'-' title '' with lines\n")

    for curve in curves:
        for coord in curve:
            file.write("%5.3f\t%5.3f\n" % tuple(coord))
        file.write("e\n\n")

    file.write("1.0\t1.0\n0.0\t0.0e\n\n")          

    if fopened:
        file.close()


@deprecated_keywords({"allResults": "all_results"})
def plot_McNemar_curve_learners(file, all_results, proportions, learners, reference=-1):
    plot_McNemar_curve(file, all_results, proportions, [Orange.misc.getobjectname(learners[i], "Learner %i" % i) for i in range(len(learners))], reference)


@deprecated_keywords({"allResults": "all_results"})
def plot_McNemar_curve(file, all_results, proportions, legend, reference=-1):
    if reference<0:
        reference=len(legend)-1
        
    import types
    fopened=0
    if type(file)==types.StringType:
        file=open(file, "wt")
        fopened=1
        
    #file.write("set yrange [0:1]\n")
    #file.write("set xrange [%f:%f]\n" % (proportions[0], proportions[-1]))
    file.write("set multiplot\n\n")
    file.write("plot \\\n")
    tmap=range(reference)+range(reference+1, len(legend))
    for i in tmap[:-1]:
        file.write("'-' title '%s' with linespoints pointtype %i,\\\n" % (legend[i], i+1))
    file.write("'-' title '%s' with linespoints pointtype %i\n" % (legend[tmap[-1]], tmap[-1]))
    file.write("\n")

    for i in tmap:
        for p in range(len(proportions)):
            file.write("%f\t%f\n" % (proportions[p], McNemar_of_two(all_results[p], i, reference)))
        file.write("e\n\n")

    if fopened:
        file.close()

default_point_types=("{$\\circ$}", "{$\\diamond$}", "{$+$}", "{$\\times$}", "{$|$}")+tuple([chr(x) for x in range(97, 122)])
default_line_types=("\\setsolid", "\\setdashpattern <4pt, 2pt>", "\\setdashpattern <8pt, 2pt>", "\\setdashes", "\\setdots")

@deprecated_keywords({"allResults": "all_results"})
def learning_curve_learners_to_PiCTeX(file, all_results, proportions, **options):
    return apply(learning_curve_to_PiCTeX, (file, all_results, proportions), options)


@deprecated_keywords({"allResults": "all_results"})
def learning_curve_to_PiCTeX(file, all_results, proportions, **options):
    import types
    fopened=0
    if type(file)==types.StringType:
        file=open(file, "wt")
        fopened=1

    nexamples=len(all_results[0].results)
    CAs = [CA(x, report_se=True) for x in all_results]

    graphsize=float(options.get("graphsize", 10.0)) #cm
    difprop=proportions[-1]-proportions[0]
    ntestexamples=nexamples*proportions[-1]
    xunit=graphsize/ntestexamples

    yshift=float(options.get("yshift", -ntestexamples/20.))
    
    pointtypes=options.get("pointtypes", default_point_types)
    linetypes=options.get("linetypes", default_line_types)

    if options.has_key("numberedx"):
        numberedx=options["numberedx"]
        if type(numberedx)==types.IntType:
            if numberedx>0:
                numberedx=[nexamples*proportions[int(i/float(numberedx)*len(proportions))] for i in range(numberedx)]+[proportions[-1]*nexamples]
            elif numberedx<0:
                numberedx = -numberedx
                newn=[]
                for i in range(numberedx+1):
                    wanted=proportions[0]+float(i)/numberedx*difprop
                    best=(10, 0)
                    for t in proportions:
                        td=abs(wanted-t)
                        if td<best[0]:
                            best=(td, t)
                    if not best[1] in newn:
                        newn.append(best[1])
                newn.sort()
                numberedx=[nexamples*x for x in newn]
        elif type(numberedx[0])==types.FloatType:
            numberedx=[nexamples*x for x in numberedx]
    else:
        numberedx=[nexamples*x for x in proportions]

    file.write("\\mbox{\n")
    file.write("  \\beginpicture\n")
    file.write("  \\setcoordinatesystem units <%10.8fcm, %5.3fcm>\n\n" % (xunit, graphsize))    
    file.write("  \\setplotarea x from %5.3f to %5.3f, y from 0 to 1\n" % (0, ntestexamples))    
    file.write("  \\axis bottom invisible\n")# label {#examples}\n")
    file.write("      ticks short at %s /\n" % reduce(lambda x,y:x+" "+y, ["%i"%(x*nexamples+0.5) for x in proportions]))
    if numberedx:
        file.write("            long numbered at %s /\n" % reduce(lambda x,y:x+y, ["%i " % int(x+0.5) for x in numberedx]))
    file.write("  /\n")
    file.write("  \\axis left invisible\n")# label {classification accuracy}\n")
    file.write("      shiftedto y=%5.3f\n" % yshift)
    file.write("      ticks short from 0.0 to 1.0 by 0.05\n")
    file.write("            long numbered from 0.0 to 1.0 by 0.25\n")
    file.write("  /\n")
    if options.has_key("default"):
        file.write("  \\setdashpattern<1pt, 1pt>\n")
        file.write("  \\plot %5.3f %5.3f %5.3f %5.3f /\n" % (0., options["default"], ntestexamples, options["default"]))
    
    for i in range(len(CAs[0])):
        coordinates=reduce(lambda x,y:x+" "+y, ["%i %5.3f" % (proportions[p]*nexamples, CAs[p][i][0]) for p in range(len(proportions))])
        if linetypes:
            file.write("  %s\n" % linetypes[i])
            file.write("  \\plot %s /\n" % coordinates)
        if pointtypes:
            file.write("  \\multiput %s at %s /\n" % (pointtypes[i], coordinates))

    file.write("  \\endpicture\n")
    file.write("}\n")
    if fopened:
        file.close()
    file.close()
    del file

def legend_learners_to_PiCTeX(file, learners, **options):
  return apply(legend_to_PiCTeX, (file, [Orange.misc.getobjectname(learners[i], "Learner %i" % i) for i in range(len(learners))]), options)
    
def legend_to_PiCTeX(file, legend, **options):
    import types
    fopened=0
    if type(file)==types.StringType:
        file=open(file, "wt")
        fopened=1

    pointtypes=options.get("pointtypes", default_point_types)
    linetypes=options.get("linetypes", default_line_types)

    file.write("\\mbox{\n")
    file.write("  \\beginpicture\n")
    file.write("  \\setcoordinatesystem units <5cm, 1pt>\n\n")
    file.write("  \\setplotarea x from 0.000 to %5.3f, y from 0 to 12\n" % len(legend))

    for i in range(len(legend)):
        if linetypes:
            file.write("  %s\n" % linetypes[i])
            file.write("  \\plot %5.3f 6 %5.3f 6 /\n" % (i, i+0.2))
        if pointtypes:
            file.write("  \\put {%s} at %5.3f 6\n" % (pointtypes[i], i+0.1))
        file.write("  \\put {%s} [lb] at %5.3f 0\n" % (legend[i], i+0.25))

    file.write("  \\endpicture\n")
    file.write("}\n")
    if fopened:
        file.close()
    file.close()
    del file


def compute_friedman(avranks, N):
    """ Returns a tuple composed of (friedman statistic, degrees of freedom)
    and (Iman statistic - F-distribution, degrees of freedoma) given average
    ranks and a number of tested data sets N.
    """

    k = len(avranks)

    def friedman(N, k, ranks):
        return 12*N*(sum([rank**2.0 for rank in ranks]) - (k*(k+1)*(k+1)/4.0) )/(k*(k+1))

    def iman(fried, N, k):
        return (N-1)*fried/(N*(k-1) - fried)

    f = friedman(N, k, avranks)
    im = iman(f, N, k)
    fdistdof = (k-1, (k-1)*(N-1))

    return (f, k-1), (im, fdistdof)

def compute_CD(avranks, N, alpha="0.05", type="nemenyi"):
    """ Returns critical difference for Nemenyi or Bonferroni-Dunn test
    according to given alpha (either alpha="0.05" or alpha="0.1") for average
    ranks and number of tested data sets N. Type can be either "nemenyi" for
    for Nemenyi two tailed test or "bonferroni-dunn" for Bonferroni-Dunn test.
    """

    k = len(avranks)
   
    d = {("nemenyi", "0.05"): [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
                               2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
                               3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
                               3.426041, 3.458425, 3.488685, 3.517073, 3.543799]
        , ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
                               2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
                               2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
                               3.195743, 3.229723, 3.261461, 3.291224, 3.319233]
        , ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
                                        2.638, 2.690, 2.724, 2.773],
         ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
                                      2.394, 2.450, 2.498, 2.539]}

    #can be computed in R as qtukey(0.95, n, Inf)**0.5
    #for (x in c(2:20)) print(qtukey(0.95, x, Inf)/(2**0.5)

    q = d[(type, alpha)]

    cd = q[k]*(k*(k+1)/(6.0*N))**0.5

    return cd
 

def graph_ranks(filename, avranks, names, cd=None, cdmethod=None, lowv=None, highv=None, width=6, textspace=1, reverse=False, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods' 
    performance.
    See Janez Demsar, Statistical Comparisons of Classifiers over 
    Multiple Data Sets, 7(Jan):1--30, 2006. 

    Needs matplotlib to work.

    :param filename: Output file name (with extension). Formats supported 
                     by matplotlib can be used.
    :param avranks: List of average methods' ranks.
    :param names: List of methods' names.

    :param cd: Critical difference. Used for marking methods that whose
               difference is not statistically significant.
    :param lowv: The lowest shown rank, if None, use 1.
    :param highv: The highest shown rank, if None, use len(avranks).
    :param width: Width of the drawn figure in inches, default 6 in.
    :param textspace: Space on figure sides left for the description
                      of methods, default 1 in.
    :param reverse:  If True, the lowest rank is on the right. Default\: False.
    :param cdmethod: None by default. It can be an index of element in avranks
                     or or names which specifies the method which should be
                     marked with an interval.
    """
    if not HAS_MATPLOTLIB:
        import sys
        print >> sys.stderr, "Function requires matplotlib. Please install it."
        return

    width = float(width)
    textspace = float(textspace)

    def nth(l,n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l,n)
        return [ a[n] for a in l ]

    def lloc(l,n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0])+n
        else:
            return n 

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5]) 
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            #it can work with single numbers
            index = lr[0]
            if type(1) == type(index):
                index = [ index ]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))


    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    tempsort =  sorted([ (a,i) for i,a in  enumerate(sums) ], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [ names[x] for x in sortidx ]
    
    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2*textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace+scalewidth/(highv-lowv)*a

    distanceh = 0.25

    if cd and cdmethod is None:
    
        #get pairs of non significant methods

        def get_lines(sums, hsd):

            #get all pairs
            lsums = len(sums)
            allpairs = [ (i,j) for i,j in mxrange([[lsums], [lsums]]) if j > i ]
            #remove not significant
            notSig = [ (i,j) for i,j in allpairs if abs(sums[i]-sums[j]) <= hsd ]
            #keep only longest
            
            def no_longer((i,j), notSig):
                for i1,j1 in notSig:
                    if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                        return False
                return True

            longest = [ (i,j) for i,j in notSig if no_longer((i,j),notSig) ]
            
            return longest

        lines = get_lines(ssums, cd)
        linesblank = 0.2 + 0.2 + (len(lines)-1)*0.1

        #add scale
        distanceh = 0.25
        cline += distanceh

    #calculate height needed height of an image
    minnotsignificant = max(2*0.2, linesblank)
    height = cline + ((k+1)/2)*0.2 + minnotsignificant

    fig = Figure(figsize=(width, height))
    ax = fig.add_axes([0,0,1,1]) #reverse y axis
    ax.set_axis_off()

    hf = 1./height # height factor
    wf = 1./width

    def hfl(l): 
        return [ a*hf for a in l ]

    def wfl(l): 
        return [ a*wf for a in l ]


    # Upper left corner is (0,0).

    ax.plot([0,1], [0,1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l,0)), hfl(nth(l,1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf*x, hf*y, s, *args, **kwargs)

    line([(textspace, cline), (width-textspace, cline)], linewidth=0.7)
    
    bigtick = 0.1
    smalltick = 0.05


    import numpy
    tick = None
    for a in list(numpy.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a): tick = bigtick
        line([(rankpos(a), cline-tick/2),(rankpos(a), cline)], linewidth=0.7)

    for a in range(lowv, highv+1):
        text(rankpos(a), cline-tick/2-0.05, str(a), ha="center", va="bottom")

    k = len(ssums)

    for i in range((k+1)/2):
        chei = cline+ minnotsignificant + i *0.2
        line([(rankpos(ssums[i]), cline), (rankpos(ssums[i]), chei), (textspace-0.1, chei)], linewidth=0.7)
        text(textspace-0.2, chei, nnames[i], ha="right", va="center")

    for i in range((k+1)/2, k):
        chei = cline + minnotsignificant + (k-i-1)*0.2
        line([(rankpos(ssums[i]), cline), (rankpos(ssums[i]), chei), (textspace+scalewidth+0.1, chei)], linewidth=0.7)
        text(textspace+scalewidth+0.2, chei, nnames[i], ha="left", va="center")

    if cd and cdmethod is None:

        #upper scale
        if not reverse:
            begin, end = rankpos(lowv), rankpos(lowv+cd)
        else:
            begin, end = rankpos(highv), rankpos(highv - cd)
            
        line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
        line([(begin, distanceh + bigtick/2), (begin, distanceh - bigtick/2)], linewidth=0.7)
        line([(end, distanceh + bigtick/2), (end, distanceh - bigtick/2)], linewidth=0.7)
        text((begin+end)/2, distanceh - 0.05, "CD", ha="center", va="bottom")

        #non significance lines    
        def draw_lines(lines, side=0.05, height=0.1):
            start = cline + 0.2
            for l,r in lines:  
                line([(rankpos(ssums[l])-side, start), (rankpos(ssums[r])+side, start)], linewidth=2.5) 
                start += height

        draw_lines(lines)

    elif cd:
        begin = rankpos(avranks[cdmethod]-cd)
        end = rankpos(avranks[cdmethod]+cd)
        line([(begin, cline), (end, cline)], linewidth=2.5) 
        line([(begin, cline + bigtick/2), (begin, cline - bigtick/2)], linewidth=2.5)
        line([(end, cline + bigtick/2), (end, cline - bigtick/2)], linewidth=2.5)
 
    print_figure(fig, filename, **kwargs)

def mlc_hamming_loss(res):
    """
    Schapire and Singer (2000) presented Hamming Loss, which id defined as: 
    
    :math:`HammingLoss(H,D)=\\frac{1}{|D|} \\sum_{i=1}^{|D|} \\frac{Y_i \\vartriangle Z_i}{|L|}`
    """
    losses = [0]*res.number_of_learners
    label_num = len(res.labels)
    example_num = gettotsize(res)

    for e in res.results:
        aclass = e.actual_class
        for i, labels in enumerate(e.classes):
            labels = map(int, labels)
            if len(labels) <> len(aclass):
                raise ValueError, "The dimensions of the classified output and the actual class array do not match."
            for j in range(label_num):
                if labels[j] != aclass[j]:
                    losses[i] += 1
            
    return [float(x)/(label_num*example_num) for x in losses]

def mlc_accuracy(res, forgiveness_rate = 1.0):
    """
    Godbole & Sarawagi, 2004 uses the metrics accuracy, precision, recall as follows:
     
    :math:`Accuracy(H,D)=\\frac{1}{|D|} \\sum_{i=1}^{|D|} \\frac{|Y_i \\cap Z_i|}{|Y_i \\cup Z_i|}`
    
    Boutell et al. (2004) give a more generalized version using a parameter :math:`\\alpha \\ge 0`, 
    called forgiveness rate:
    
    :math:`Accuracy(H,D)=\\frac{1}{|D|} \\sum_{i=1}^{|D|} (\\frac{|Y_i \\cap Z_i|}{|Y_i \\cup Z_i|})^{\\alpha}`
    """
    accuracies = [0.0]*res.number_of_learners
    example_num = gettotsize(res)
    
    for e in res.results:
        aclass = e.actual_class
        for i, labels in enumerate(e.classes):
            labels = map(int, labels)
            if len(labels) <> len(aclass):
                raise ValueError, "The dimensions of the classified output and the actual class array do not match."
            
            intersection = 0.0
            union = 0.0
            for real, pred in zip(labels, aclass):
                if real and pred:
                    intersection += 1
                if real or pred:
                    union += 1

            if union:
                accuracies[i] += intersection / union
            
    return [math.pow(x/example_num,forgiveness_rate) for x in accuracies]

def mlc_precision(res):
    """
    :math:`Precision(H,D)=\\frac{1}{|D|} \\sum_{i=1}^{|D|} \\frac{|Y_i \\cap Z_i|}{|Z_i|}`
    """
    precisions = [0.0]*res.number_of_learners
    example_num = gettotsize(res)
    
    for e in res.results:
        aclass = e.actual_class
        for i, labels in enumerate(e.classes):
            labels = map(int, labels)
            if len(labels) <> len(aclass):
                raise ValueError, "The dimensions of the classified output and the actual class array do not match."
            
            intersection = 0.0
            predicted = 0.0
            for real, pred in zip(labels, aclass):
                if real and pred:
                    intersection += 1
                if real:
                    predicted += 1
            if predicted:
                precisions[i] += intersection / predicted
            
    return [x/example_num for x in precisions]

def mlc_recall(res):
    """
    :math:`Recall(H,D)=\\frac{1}{|D|} \\sum_{i=1}^{|D|} \\frac{|Y_i \\cap Z_i|}{|Y_i|}`
    """
    recalls = [0.0]*res.number_of_learners
    example_num = gettotsize(res)
    
    for e in res.results:
        aclass = e.actual_class
        for i, labels in enumerate(e.classes):
            labels = map(int, labels)
            if len(labels) <> len(aclass):
                raise ValueError, "The dimensions of the classified output and the actual class array do not match."
            
            intersection = 0.0
            actual = 0.0
            for real, pred in zip(labels, aclass):
                if real and pred:
                    intersection += 1
                if pred:
                    actual += 1
            if actual:
                recalls[i] += intersection / actual
            
    return [x/example_num for x in recalls]

#def mlc_ranking_loss(res):
#    pass
#
#def mlc_average_precision(res):
#    pass
#
#def mlc_hierarchical_loss(res):
#    pass


def mt_average_score(res, score, weights=None):
    """
    Compute individual scores for each target and return the (weighted) average.

    One method can be used to compute scores for all targets or a list of
    scoring methods can be passed to use different methods for different
    targets. In the latter case, care has to be taken if the ranges of scoring
    methods differ.
    For example, when the first target is scored from -1 to 1 (1 best) and the
    second from 0 to 1 (0 best), using `weights=[0.5,-1]` would scale both
    to a span of 1, and invert the second so that higher scores are better.

    :param score: Single-target scoring method or a list of such methods
                  (one for each target).
    :param weights: List of real weights, one for each target,
                    for a weighted average.

    """
    if not len(res.results):
        raise ValueError, "Cannot compute the score: no examples."
    if res.number_of_learners < 1:
        return []
    n_classes = len(res.results[0].actual_class)
    if weights is None:
        weights = [1.] * n_classes
    if not hasattr(score, '__len__'):
        score = [score] * n_classes
    elif len(score) != n_classes:
        raise ValueError, "Number of scoring methods and targets do not match."
    # save original classes
    clsss = [te.classes for te in res.results]
    aclsss = [te.actual_class for te in res.results]
    # compute single target scores
    single_scores = []
    for i in range(n_classes):
        for te, clss, aclss in zip(res.results, clsss, aclsss):
            te.classes = [cls[i] for cls in clss]
            te.actual_class = aclss[i]
        single_scores.append(score[i](res))
    # restore original classes
    for te, clss, aclss in zip(res.results, clsss, aclsss):
        te.classes = clss
        te.actual_class = aclss
    return [sum(w * s for w, s in zip(weights, scores)) / sum(weights)
        for scores in zip(*single_scores)]

def mt_flattened_score(res, score):
    """
    Flatten (concatenate into a single list) the predictions of multiple
    targets and compute a single-target score.
    
    :param score: Single-target scoring method.
    """
    res2 = Orange.evaluation.testing.ExperimentResults(res.number_of_iterations,
        res.classifier_names, class_values=res.class_values,
        weights=res.weights, classifiers=res.classifiers, loaded=res.loaded,
        test_type=Orange.evaluation.testing.TEST_TYPE_SINGLE, labels=res.labels)
    for te in res.results:
        for i, ac in enumerate(te.actual_class):
            te2 = Orange.evaluation.testing.TestedExample(
                iteration_number=te.iteration_number, actual_class=ac)
            for c, p in zip(te.classes, te.probabilities):
                te2.add_result(c[i], p[i])
            res2.results.append(te2)
    return score(res2)


################################################################################
if __name__ == "__main__":
    avranks =  [3.143, 2.000, 2.893, 1.964]
    names = ["prva", "druga", "tretja", "cetrta" ]
    cd = compute_CD(avranks, 14)
    #cd = compute_CD(avranks, 10, type="bonferroni-dunn")
    print cd

    print compute_friedman(avranks, 14)

    #graph_ranks("test.eps", avranks, names, cd=cd, cdmethod=0, width=6, textspace=1.5)
