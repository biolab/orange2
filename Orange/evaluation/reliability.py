import Orange

import random
from Orange import statc
import math
import warnings
import numpy

from collections import defaultdict
from itertools import izip

# Labels and final variables
labels = ["SAvar", "SAbias", "BAGV", "CNK", "LCV", "BVCK", "Mahalanobis", "ICV"]

"""
# All the estimators calculation constants
DO_SA = 0
DO_BAGV = 1
DO_CNK = 2
DO_LCV = 3
DO_BVCK = 4
DO_MAHAL = 5
"""

# All the estimator method constants
SAVAR_ABSOLUTE = 0
SABIAS_SIGNED = 1
SABIAS_ABSOLUTE = 2
BAGV_ABSOLUTE = 3
CNK_SIGNED = 4
CNK_ABSOLUTE = 5
LCV_ABSOLUTE = 6
BVCK_ABSOLUTE = 7
MAHAL_ABSOLUTE = 8
BLENDING_ABSOLUTE = 9
ICV_METHOD = 10
MAHAL_TO_CENTER_ABSOLUTE = 13

# Type of estimator constant
SIGNED = 0
ABSOLUTE = 1

# Names of all the estimator methods
METHOD_NAME = {0: "SAvar absolute", 1: "SAbias signed", 2: "SAbias absolute",
               3: "BAGV absolute", 4: "CNK signed", 5: "CNK absolute",
               6: "LCV absolute", 7: "BVCK_absolute", 8: "Mahalanobis absolute",
               9: "BLENDING absolute", 10: "ICV", 11: "RF Variance", 12: "RF Std",
               13: "Mahalanobis to center"}

select_with_repeat = Orange.core.MakeRandomIndicesMultiple()
select_with_repeat.random_generator = Orange.misc.Random()

def get_reliability_estimation_list(res, i):
    return [result.probabilities[0].reliability_estimate[i].estimate for result in res.results], res.results[0].probabilities[0].reliability_estimate[i].signed_or_absolute, res.results[0].probabilities[0].reliability_estimate[i].method

def get_prediction_error_list(res):
    return [result.actualClass - result.classes[0] for result in res.results]

def get_description_list(res, i):
    return [result.probabilities[0].reliability_estimate[i].text_description for result in res.results]

def get_pearson_r(res):
    """
    :param res: results of evaluation, done using learners,
        wrapped into :class:`Orange.evaluation.reliability.Classifier`.
    :type res: :class:`Orange.evaluation.testing.ExperimentResults`

    Return Pearson's coefficient between the prediction error and each of the
    used reliability estimates. Also, return the p-value of each of
    the coefficients.
    """
    prediction_error = get_prediction_error_list(res)
    results = []
    for i in xrange(len(res.results[0].probabilities[0].reliability_estimate)):
        reliability_estimate, signed_or_absolute, method = get_reliability_estimation_list(res, i)
        try:
            if signed_or_absolute == SIGNED:
                r, p = statc.pearsonr(prediction_error, reliability_estimate)
            else:
                r, p = statc.pearsonr([abs(pe) for pe in prediction_error], reliability_estimate)
        except Exception:
            r = p = float("NaN")
        results.append((r, p, signed_or_absolute, method))
    return results

def get_spearman_r(res):
    """
    :param res: results of evaluation, done using learners,
        wrapped into :class:`Orange.evaluation.reliability.Classifier`.
    :type res: :class:`Orange.evaluation.testing.ExperimentResults`

    Return Spearman's coefficient between the prediction error and each of the
    used reliability estimates. Also, return the p-value of each of
    the coefficients.
    """
    prediction_error = get_prediction_error_list(res)
    results = []
    for i in xrange(len(res.results[0].probabilities[0].reliability_estimate)):
        reliability_estimate, signed_or_absolute, method = get_reliability_estimation_list(res, i)
        try:
            if signed_or_absolute == SIGNED:
                r, p = statc.spearmanr(prediction_error, reliability_estimate)
            else:
                r, p = statc.spearmanr([abs(pe) for pe in prediction_error], reliability_estimate)
        except Exception:
            r = p = float("NaN")
        results.append((r, p, signed_or_absolute, method))
    return results

def get_pearson_r_by_iterations(res):
    """
    :param res: results of evaluation, done using learners,
        wrapped into :class:`Orange.evaluation.reliability.Classifier`.
    :type res: :class:`Orange.evaluation.testing.ExperimentResults`

    Return average Pearson's coefficient over all folds between prediction error
    and each of the used estimates.
    """
    results_by_fold = Orange.evaluation.scoring.split_by_iterations(res)
    number_of_estimates = len(res.results[0].probabilities[0].reliability_estimate)
    number_of_instances = len(res.results)
    number_of_folds = len(results_by_fold)
    results = [0 for _ in xrange(number_of_estimates)]
    sig = [0 for _ in xrange(number_of_estimates)]
    method_list = [0 for _ in xrange(number_of_estimates)]
    
    for res in results_by_fold:
        prediction_error = get_prediction_error_list(res)
        for i in xrange(number_of_estimates):
            reliability_estimate, signed_or_absolute, method = get_reliability_estimation_list(res, i)
            try:
                if signed_or_absolute == SIGNED:
                    r, _ = statc.pearsonr(prediction_error, reliability_estimate)
                else:
                    r, _ = statc.pearsonr([abs(pe) for pe in prediction_error], reliability_estimate)
            except Exception:
                r = float("NaN")
            results[i] += r
            sig[i] = signed_or_absolute
            method_list[i] = method
    
    # Calculate p-values
    results = [float(res) / number_of_folds for res in results]
    ps = [p_value_from_r(r, number_of_instances) for r in results]
    
    return zip(results, ps, sig, method_list)

def p_value_from_r(r, n):
    """
    Calculate p-value from the paerson coefficient and the sample size.
    """
    df = n - 2
    t = r * (df /((-r + 1.0 + 1e-30) * (r + 1.0 + 1e-30)) )**0.5
    return statc.betai (df * 0.5, 0.5, df/(df + t*t))

class Estimate:
    """
    Reliability estimate. Contains attributes that describe the results of
    reliability estimation.

    .. attribute:: estimate

        A numerical reliability estimate.

    .. attribute:: signed_or_absolute

        Determines whether the method used gives a signed or absolute result.
        Has a value of either :obj:`SIGNED` or :obj:`ABSOLUTE`.

    .. attribute:: method

        An integer ID of reliability estimation method used.

    .. attribute:: method_name

        Name (string) of reliability estimation method used.

    .. attribute:: icv_method

        An integer ID of reliability estimation method that performed best,
        as determined by ICV, and of which estimate is stored in the
        :obj:`estimate` field. (:obj:`None` when ICV was not used.)

    .. attribute:: icv_method_name

        Name (string) of reliability estimation method that performed best,
        as determined by ICV. (:obj:`None` when ICV was not used.)

    """
    def __init__(self, estimate, signed_or_absolute, method, icv_method = -1):
        self.estimate = estimate
        self.signed_or_absolute = signed_or_absolute
        self.method = method
        self.method_name = METHOD_NAME[method]
        self.icv_method = icv_method
        self.icv_method_name = METHOD_NAME[icv_method] if icv_method != -1 else ""
        self.text_description = None

class DescriptiveAnalysis:
    def __init__(self, estimator, desc=["high", "medium", "low"], procentage=[0.00, 0.33, 0.66]):
        self.desc = desc
        self.procentage = procentage
        self.estimator = estimator
    
    def __call__(self, instances, weight=None, **kwds):
        
        # Calculate borders using cross validation
        res = Orange.evaluation.testing.cross_validation([self.estimator], instances)
        all_borders = []
        for i in xrange(len(res.results[0].probabilities[0].reliability_estimate)):
            estimates, signed_or_absolute, method = get_reliability_estimation_list(res, i)
            sorted_estimates = sorted( abs(x) for x in estimates)
            borders = [sorted_estimates[int(len(estimates)*p)-1]  for p in self.procentage]
            all_borders.append(borders)
        
        # Learn on whole train data
        estimator_classifier = self.estimator(instances)
        
        return DescriptiveAnalysisClassifier(estimator_classifier, all_borders, self.desc)

class DescriptiveAnalysisClassifier:
    def __init__(self, estimator_classifier, all_borders, desc):
        self.estimator_classifier = estimator_classifier
        self.all_borders = all_borders
        self.desc = desc
    
    def __call__(self, instance, result_type=Orange.core.GetValue):
        predicted, probabilities = self.estimator_classifier(instance, Orange.core.GetBoth)
        
        for borders, estimate in zip(self.all_borders, probabilities.reliability_estimate):
            estimate.text_description = self.desc[0]
            for lower_border, text_desc in zip(borders, self.desc):
                if estimate.estimate >= lower_border:
                    estimate.text_description = text_desc
        
        # Return the appropriate type of result
        if result_type == Orange.core.GetValue:
            return predicted
        elif result_type == Orange.core.GetProbabilities:
            return probabilities
        else:
            return predicted, probabilities

class SensitivityAnalysis:
    """
    
    :param e: List of possible :math:`\epsilon` values for SAvar and SAbias
        reliability estimates.
    :type e: list of floats
    
    :rtype: :class:`Orange.evaluation.reliability.SensitivityAnalysisClassifier`
    
    To estimate the reliability of prediction for given instance,
    the learning set is extended with this instance, labeled with
    :math:`K + \epsilon (l_{max} - l_{min})`,
    where :math:`K` denotes the initial prediction,
    :math:`\epsilon` is sensitivity parameter and :math:`l_{min}` and
    :math:`l_{max}` denote lower and the upper bound of the learning
    instances' labels. After computing different sensitivity predictions
    using different values of :math:`\epsilon`, the prediction are combined
    into SAvar and SAbias. SAbias can be used in a signed or absolute form.

    :math:`SAvar = \\frac{\sum_{\epsilon \in E}(K_{\epsilon} - K_{-\epsilon})}{|E|}`

    :math:`SAbias = \\frac{\sum_{\epsilon \in E} (K_{\epsilon} - K ) + (K_{-\epsilon} - K)}{2 |E|}`
    
    
    """
    def __init__(self, e=[0.01, 0.1, 0.5, 1.0, 2.0]):
        self.e = e
    
    def __call__(self, instances, learner):
        min_value = max_value = instances[0].getclass().value
        for ex in instances:
            if ex.getclass().value > max_value:
                max_value = ex.getclass().value
            if ex.getclass().value < min_value:
                min_value = ex.getclass().value
        return SensitivityAnalysisClassifier(self.e, instances, min_value, max_value, learner)
    
class SensitivityAnalysisClassifier:
    def __init__(self, e, instances, min_value, max_value, learner):
        self.e = e
        self.instances = instances
        self.max_value = max_value
        self.min_value = min_value
        self.learner = learner
    
    def __call__(self, instance, predicted, probabilities):
        # Create new dataset
        r_data = Orange.data.Table(self.instances)
        
        # Create new instance
        modified_instance = Orange.data.Instance(instance)
        
        # Append it to the data
        r_data.append(modified_instance)
        
        # Calculate SAvar & SAbias
        SAvar = SAbias = 0
        
        for eps in self.e:
            # +epsilon
            r_data[-1].setclass(predicted.value + eps*(self.max_value - self.min_value))
            c = self.learner(r_data)
            k_plus = c(instance, Orange.core.GetValue)
            
            # -epsilon
            r_data[-1].setclass(predicted.value - eps*(self.max_value - self.min_value))
            c = self.learner(r_data)
            k_minus = c(instance, Orange.core.GetValue)
            #print len(r_data)
            #print eps*(self.max_value - self.min_value)
            #print k_plus
            #print k_minus
            # calculate part SAvar and SAbias
            SAvar += k_plus.value - k_minus.value
            SAbias += k_plus.value + k_minus.value - 2*predicted.value
        
        SAvar /= len(self.e)
        SAbias /= 2*len(self.e)
        
        return [Estimate(SAvar, ABSOLUTE, SAVAR_ABSOLUTE),
                Estimate(SAbias, SIGNED, SABIAS_SIGNED),
                Estimate(abs(SAbias), ABSOLUTE, SABIAS_ABSOLUTE)]
    
class BaggingVariance:
    """
    
    :param m: Number of bagging models to be used with BAGV estimate
    :type m: int
    
    :rtype: :class:`Orange.evaluation.reliability.BaggingVarianceClassifier`
    
    :math:`m` different bagging models are constructed and used to estimate
    the value of dependent variable for a given instance. The variance of
    those predictions is used as a prediction reliability estimate.

    :math:`BAGV = \\frac{1}{m} \sum_{i=1}^{m} (K_i - K)^2`

    where :math:`K = \\frac{\sum_{i=1}^{m} K_i}{m}` and :math:`K_i` are
    predictions of individual constructed models.
    
    """
    def __init__(self, m=50):
        self.m = m
    
    def __call__(self, instances, learner):
        classifiers = []
        
        # Create bagged classifiers using sampling with replacement
        for _ in xrange(self.m):
            selection = select_with_repeat(len(instances))
            data = instances.select(selection)
            classifiers.append(learner(data))
        return BaggingVarianceClassifier(classifiers)

class BaggingVarianceClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers
    
    def __call__(self, instance, *args):
        BAGV = 0
        
        # Calculate the bagging variance
        bagged_values = [c(instance, Orange.core.GetValue).value for c in self.classifiers if c is not None]
        
        k = sum(bagged_values) / len(bagged_values)
        
        BAGV = sum( (bagged_value - k)**2 for bagged_value in bagged_values) / len(bagged_values)
        
        return [Estimate(BAGV, ABSOLUTE, BAGV_ABSOLUTE)]
        
class LocalCrossValidation:
    """
    
    :param k: Number of nearest neighbours used in LCV estimate
    :type k: int
    
    :rtype: :class:`Orange.evaluation.reliability.LocalCrossValidationClassifier`
    
    :math:`k` nearest neighbours to the given instance are found and put in
    a separate data set. On this data set, a leave-one-out validation is
    performed. Reliability estimate is then the distance weighted absolute
    prediction error.

    If a special value 0 is passed as :math:`k` (as is by default),
    it is set as 1/20 of data set size (or 5, whichever is greater).
    
    1. Determine the set of k nearest neighours :math:`N = { (x_1, c_1),...,
       (x_k, c_k)}`.
    2. On this set, compute leave-one-out predictions :math:`K_i` and
       prediction errors :math:`E_i = | C_i - K_i |`.
    3. :math:`LCV(x) = \\frac{ \sum_{(x_i, c_i) \in N} d(x_i, x) * E_i }{ \sum_{(x_i, c_i) \in N} d(x_i, x) }`
    
    """
    def __init__(self, k=0):
        self.k = k
    
    def __call__(self, instances, learner):
        nearest_neighbours_constructor = Orange.classification.knn.FindNearestConstructor()
        nearest_neighbours_constructor.distanceConstructor = Orange.distance.Euclidean()
        
        distance_id = Orange.feature.Descriptor.new_meta_id()
        nearest_neighbours = nearest_neighbours_constructor(instances, 0, distance_id)
        
        if self.k == 0:
            self.k = max(5, len(instances)/20)
        
        return LocalCrossValidationClassifier(distance_id, nearest_neighbours, self.k, learner)

class LocalCrossValidationClassifier:
    def __init__(self, distance_id, nearest_neighbours, k, learner):
        self.distance_id = distance_id
        self.nearest_neighbours = nearest_neighbours
        self.k = k
        self.learner = learner
    
    def __call__(self, instance, *args):
        LCVer = 0
        LCVdi = 0
        
        # Find k nearest neighbors
        
        knn = [ex for ex in self.nearest_neighbours(instance, self.k)]
        
        # leave one out of prediction error
        for i in xrange(len(knn)):
            train = knn[:]
            del train[i]
            
            classifier = self.learner(Orange.data.Table(train))
            
            returned_value = classifier(knn[i], Orange.core.GetValue)
            
            e = abs(knn[i].getclass().value - returned_value.value)
            
            LCVer += e * math.exp(-knn[i][self.distance_id])
            LCVdi += math.exp(-knn[i][self.distance_id])
        
        LCV = LCVer / LCVdi if LCVdi != 0 else 0
        if math.isnan(LCV):
            LCV = 0.0
        return [ Estimate(LCV, ABSOLUTE, LCV_ABSOLUTE) ]

class CNeighbours:
    """
    
    :param k: Number of nearest neighbours used in CNK estimate
    :type k: int
    
    :rtype: :class:`Orange.evaluation.reliability.CNeighboursClassifier`
    
    CNK is defined for an unlabeled instance as a difference between average
    label of its nearest neighbours and its prediction. CNK can be used as a
    signed or absolute estimate.
    
    :math:`CNK = \\frac{\sum_{i=1}^{k}C_i}{k} - K`
    
    where :math:`k` denotes number of neighbors, C :sub:`i` denotes neighbours'
    labels and :math:`K` denotes the instance's prediction.
    
    """
    def __init__(self, k=5):
        self.k = k
    
    def __call__(self, instances, learner):
        nearest_neighbours_constructor = Orange.classification.knn.FindNearestConstructor()
        nearest_neighbours_constructor.distanceConstructor = Orange.distance.Euclidean()
        
        distance_id = Orange.feature.Descriptor.new_meta_id()
        nearest_neighbours = nearest_neighbours_constructor(instances, 0, distance_id)
        return CNeighboursClassifier(nearest_neighbours, self.k)

class CNeighboursClassifier:
    def __init__(self, nearest_neighbours, k):
        self.nearest_neighbours = nearest_neighbours
        self.k = k
    
    def __call__(self, instance, predicted, probabilities):
        CNK = 0
        
        # Find k nearest neighbors
        
        knn = [ex for ex in self.nearest_neighbours(instance, self.k)]
        
        # average label of neighbors
        for ex in knn:
            CNK += ex.getclass().value
        
        CNK /= self.k
        CNK -= predicted.value
        
        return [Estimate(CNK, SIGNED, CNK_SIGNED),
                Estimate(abs(CNK), ABSOLUTE, CNK_ABSOLUTE)]
    
class Mahalanobis:
    """
    
    :param k: Number of nearest neighbours used in Mahalanobis estimate.
    :type k: int
    
    :rtype: :class:`Orange.evaluation.reliability.MahalanobisClassifier`
    
    Mahalanobis distance reliability estimate is defined as
    `mahalanobis distance <http://en.wikipedia.org/wiki/Mahalanobis_distance>`_
    to the evaluated instance's :math:`k` nearest neighbours.

    
    """
    def __init__(self, k=3):
        self.k = k
    
    def __call__(self, instances, *args):
        nnm = Orange.classification.knn.FindNearestConstructor()
        nnm.distanceConstructor = Orange.distance.Mahalanobis()
        
        mid = Orange.feature.Descriptor.new_meta_id()
        nnm = nnm(instances, 0, mid)
        return MahalanobisClassifier(self.k, nnm, mid)

class MahalanobisClassifier:
    def __init__(self, k, nnm, mid):
        self.k = k
        self.nnm = nnm
        self.mid = mid
    
    def __call__(self, instance, *args):
        mahalanobis_distance = 0
        
        mahalanobis_distance = sum(ex[self.mid].value for ex in self.nnm(instance, self.k))
        
        return [ Estimate(mahalanobis_distance, ABSOLUTE, MAHAL_ABSOLUTE) ]

class MahalanobisToCenter:
    """
    :rtype: :class:`Orange.evaluation.reliability.MahalanobisToCenterClassifier`
    
    Mahalanobis distance to center reliability estimate is defined as a
    `mahalanobis distance <http://en.wikipedia.org/wiki/Mahalanobis_distance>`_
    between the predicted instance and the centroid of the data.

    
    """
    def __init__(self):
        pass
    
    def __call__(self, instances, *args):
        dc = Orange.core.DomainContinuizer()
        dc.classTreatment = Orange.core.DomainContinuizer.Ignore
        dc.continuousTreatment = Orange.core.DomainContinuizer.NormalizeBySpan
        dc.multinomialTreatment = Orange.core.DomainContinuizer.NValues
        
        new_domain = dc(instances)
        new_instances = instances.translate(new_domain)
        
        X, _, _ = new_instances.to_numpy()
        instance_avg = numpy.average(X, 0)
        
        distance_constructor = Orange.distance.Mahalanobis()
        distance = distance_constructor(new_instances)
        
        average_instance = Orange.data.Instance(new_instances.domain, list(instance_avg) + ["?"])
        
        return MahalanobisToCenterClassifier(distance, average_instance, new_domain)

class MahalanobisToCenterClassifier:
    def __init__(self, distance, average_instance, new_domain):
        self.distance = distance
        self.average_instance = average_instance
        self.new_domain = new_domain
    
    def __call__(self, instance, *args):
        
        inst = Orange.data.Instance(self.new_domain, instance)
        
        mahalanobis_to_center = self.distance(inst, self.average_instance)
        
        return [ Estimate(mahalanobis_to_center, ABSOLUTE, MAHAL_TO_CENTER_ABSOLUTE) ]


class BaggingVarianceCNeighbours:
    """
    
    :param bagv: Instance of Bagging Variance estimator.
    :type bagv: :class:`Orange.evaluation.reliability.BaggingVariance`
    
    :param cnk: Instance of CNK estimator.
    :type cnk: :class:`Orange.evaluation.reliability.CNeighbours`
    
    :rtype: :class:`Orange.evaluation.reliability.BaggingVarianceCNeighboursClassifier`
    
    BVCK is a combination (average) of Bagging variance and local modeling of
    prediction error.
    
    """
    def __init__(self, bagv=BaggingVariance(), cnk=CNeighbours()):
        self.bagv = bagv
        self.cnk = cnk
    
    def __call__(self, instances, learner):
        bagv_classifier = self.bagv(instances, learner)
        cnk_classifier = self.cnk(instances, learner)
        return BaggingVarianceCNeighboursClassifier(bagv_classifier, cnk_classifier)

class BaggingVarianceCNeighboursClassifier:
    def __init__(self, bagv_classifier, cnk_classifier):
        self.bagv_classifier = bagv_classifier
        self.cnk_classifier = cnk_classifier
    
    def __call__(self, instance, predicted, probabilities):
        bagv_estimates = self.bagv_classifier(instance, predicted, probabilities)
        cnk_estimates = self.cnk_classifier(instance, predicted, probabilities)
        
        bvck_value = (bagv_estimates[0].estimate + cnk_estimates[1].estimate)/2
        bvck_estimates = [ Estimate(bvck_value, ABSOLUTE, BVCK_ABSOLUTE) ]
        bvck_estimates.extend(bagv_estimates)
        bvck_estimates.extend(cnk_estimates)
        return bvck_estimates

class ErrorPredicting:
    def __init__(self):
        pass
    
    def __call__(self, instances, learner):
        res = Orange.evaluation.testing.cross_validation([learner], instances)
        prediction_errors = get_prediction_error_list(res)
        
        new_domain = Orange.data.Domain(instances.domain.attributes, Orange.core.FloatVariable("pe"))
        new_dataset = Orange.data.Table(new_domain, instances)
        
        for instance, prediction_error in izip(new_dataset, prediction_errors):
            instance.set_class(prediction_error)
        
        rf = Orange.ensemble.forest.RandomForestLearner()
        rf_classifier = rf(new_dataset)
        
        return ErrorPredictingClassification(rf_classifier, new_domain)
        
class ErrorPredictingClassification:
    def __init__(self, rf_classifier, new_domain):
        self.rf_classifier = rf_classifier
        self.new_domain = new_domain
    
    def __call__(self, instance, predicted, probabilities):
        new_instance = Orange.data.Instance(self.new_domain, instance)
        value = self.rf_classifier(new_instance, Orange.core.GetValue)
        
        return [Estimate(value.value, SIGNED, SABIAS_SIGNED)]

class Learner:
    """
    Reliability estimation wrapper around a learner we want to test.
    Different reliability estimation algorithms can be used on the
    chosen learner. This learner works as any other and can be used as one,
    but it returns the classifier, wrapped into an instance of
    :class:`Orange.evaluation.reliability.Classifier`.
    
    :param box_learner: Learner we want to wrap into a reliability estimation
        classifier.
    :type box_learner: learner
    
    :param estimators: List of different reliability estimation methods we
                       want to use on the chosen learner.
    :type estimators: list of reliability estimators
    
    :param name: Name of this reliability learner
    :type name: string
    
    :rtype: :class:`Orange.evaluation.reliability.Learner`
    """
    def __init__(self, box_learner, name="Reliability estimation",
                 estimators = [SensitivityAnalysis(),
                               LocalCrossValidation(),
                               BaggingVarianceCNeighbours(),
                               Mahalanobis(),
                               MahalanobisToCenter()
                               ],
                 **kwds):
        self.__dict__.update(kwds)
        self.name = name
        self.estimators = estimators
        self.box_learner = box_learner
        self.blending = False
        
    
    def __call__(self, instances, weight=None, **kwds):
        """Learn from the given table of data instances.
        
        :param instances: Data instances to learn from.
        :type instances: Orange.data.Table
        :param weight: Id of meta attribute with weights of instances
        :type weight: integer
        :rtype: :class:`Orange.evaluation.reliability.Classifier`
        """
        
        blending_classifier = None
        new_domain = None
        
        if instances.domain.class_var.var_type != Orange.feature.Continuous.Continuous:
            raise Exception("This method only works on data with continuous class.")
        
        return Classifier(instances, self.box_learner, self.estimators, self.blending, new_domain, blending_classifier)
    
    def internal_cross_validation(self, instances, folds=10):
        """ Perform the internal cross validation for getting the best
        reliability estimate. It uses the reliability estimators defined in
        estimators attribute.

        Returns the id of the method that scored the best.

        :param instances: Data instances to use for ICV.
        :type instances: :class:`Orange.data.Table`
        :param folds: number of folds for ICV.
        :type folds: int
        :rtype: int

        """
        res = Orange.evaluation.testing.cross_validation([self], instances, folds=folds)
        results = get_pearson_r(res)
        sorted_results = sorted(results)
        return sorted_results[-1][3]
    
    def internal_cross_validation_testing(self, instances, folds=10):
        """ Perform internal cross validation (as in Automatic selection of
        reliability estimates for individual regression predictions,
        Zoran Bosnic, 2010) and return id of the method
        that scored best on this data.

        :param instances: Data instances to use for ICV.
        :type instances: :class:`Orange.data.Table`
        :param folds: number of folds for ICV.
        :type folds: int
        :rtype: int

        """
        cv_indices = Orange.core.MakeRandomIndicesCV(instances, folds)
        
        list_of_rs = []
        
        sum_of_rs = defaultdict(float)
        
        for fold in xrange(folds):
            data = instances.select(cv_indices, fold)
            if len(data) < 10:
                res = Orange.evaluation.testing.leave_one_out([self], data)
            else:
                res = Orange.evaluation.testing.cross_validation([self], data)
            results = get_pearson_r(res)
            for r, _, _, method in results:
                sum_of_rs[method] += r
        sorted_sum_of_rs = sorted(sum_of_rs.items(), key=lambda estimate: estimate[1], reverse=True)
        return sorted_sum_of_rs[0][0]
    
    labels = ["SAvar", "SAbias", "BAGV", "CNK", "LCV", "BVCK", "Mahalanobis", "ICV"]

class Classifier:
    """
    A reliability estimation wrapper for classifiers.

    What distinguishes this classifier is that the returned probabilities (if
    :obj:`Orange.classification.Classifier.GetProbabilities` or
    :obj:`Orange.classification.Classifier.GetBoth` is passed) contain an
    additional attribute :obj:`reliability_estimate`, which is an instance of
    :class:`~Orange.evaluation.reliability.Estimate`.

    """

    def __init__(self, instances, box_learner, estimators, blending, blending_domain, rf_classifier, **kwds):
        self.__dict__.update(kwds)
        self.instances = instances
        self.box_learner = box_learner
        self.estimators = estimators
        self.blending = blending
        self.blending_domain = blending_domain
        self.rf_classifier = rf_classifier
        
        # Train the learner with original data
        self.classifier = box_learner(instances)
        
        # Train all the estimators and create their classifiers
        self.estimation_classifiers = [estimator(instances, box_learner) for estimator in estimators]
    
    def __call__(self, instance, result_type=Orange.core.GetValue):
        """
        Classify and estimate reliability of estimation for a new instance.
        When :obj:`result_type` is set to
        :obj:`Orange.classification.Classifier.GetBoth` or
        :obj:`Orange.classification.Classifier.GetProbabilities`,
        an additional attribute :obj:`reliability_estimate`,
        which is an instance of
        :class:`~Orange.evaluation.reliability.Estimate`,
        is added to the distribution object.
        
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """
        predicted, probabilities = self.classifier(instance, Orange.core.GetBoth)
        
        # Create a place holder for estimates
        if probabilities is None:
            probabilities = Orange.statistics.distribution.Continuous()
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        probabilities.setattr('reliability_estimate', [])
        
        # Calculate all the estimates and add them to the results
        for estimate in self.estimation_classifiers:
            probabilities.reliability_estimate.extend(estimate(instance, predicted, probabilities))
        
        # Return the appropriate type of result
        if result_type == Orange.core.GetValue:
            return predicted
        elif result_type == Orange.core.GetProbabilities:
            return probabilities
        else:
            return predicted, probabilities
