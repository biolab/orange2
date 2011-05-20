"""
.. index:: Reliability Estimation

.. index::
   single: reliability; Reliability Estimation for Regression

*************************************
Reliability Estimation for Regression
*************************************

This module includes different implementations of algorithm used for
predicting reliability of single predictions. Most of the algorithm are taken
from Comparison of approaches for estimating reliability of individual
regression predictions, Zoran Bosnic 2008.

The following example shows a basic usage of reliability estimates
(`reliability-run.py`_, uses `housing.tab`_):

.. literalinclude:: code/reliability-run.py

Reliability estimation methods are computationaly quite hard so it may take
a bit of time for this script to produce a result.

Reliability Methods
===================

Sensitivity Analysis (SAvar and SAbias)
---------------------------------------

To estimate the reliabilty for given example we extend the learning set 
with given example and labeling it with K + e(l\ :sub:`max` \ - l\ :sub:`min` \),
where K denotes the initial prediction, e is sensitivity parameter and
l\ :sub:`min` \ and l\ :sub:`max` \ denote lower and the upper bound of
the learning examples. After computing different sensitivity predictions
using different values of e, the prediction are combined into SAvar and SAbias.
SAbias can be used as signed estimate or as absolute value of SAbias. 

Variance of bagged models (BAGV)
--------------------------------

We construct m different bagging models of the original chosen learner and use
those predictions of given example to calculate the variance, which we use as
reliability estimator.

Local cross validation reliability estimate (LCV)
-------------------------------------------------

We find k nearest neighbours to the given example and put them in
seperate dataset. On this dataset we do leave one out
validation using given model. Reliability estimate is then distance
weighted absolute prediction error.

Local modeling of prediction error (CNK)
----------------------------------------

Estimate CNK is defined for unlabeled example as difference between
average label of the nearest neighbours and the examples prediction. CNK can
be used as a signed estimate or only as absolute value. 

Bagging variance c-neighbours (BVCK)
------------------------------------

BVCK is a combination of Bagging variance and local modeling of prediction
error, for this estimate we take the average of both.

Mahalanobis distance
--------------------

Mahalanobis distance estimate is defined as mahalanobis distance to the
three nearest neighbours of chosen example.


Reliability estimate learner
============================

.. autoclass:: Learner

.. autofunction:: get_pearson_r

.. autofunction:: get_pearson_r_by_iterations

Referencing
===========

These methods can be referenced using constants inside the module. For setting,
which methods to use after creating learner, like this::

  reliability.use[Orange.evaluation.reliability.DO_SA] = False
  reliability.use[Orange.evaluation.reliability.DO_BAGV] = True
  reliability.use[Orange.evaluation.reliability.DO_CNK] = False
  reliability.use[Orange.evaluation.reliability.DO_LCV] = True
  reliability.use[Orange.evaluation.reliability.DO_BVCK] = False
  reliability.use[Orange.evaluation.reliability.DO_MAHAL] = False

There is also a dictionary named :data:`METHOD_NAME` which has stored names of
all the reliability estimates::

  METHOD_NAME = {0: "SAvar absolute", 1: "SAbias signed", 2: "SAbias absolute",
                 3: "BAGV absolute", 4: "CNK signed", 5: "CNK absolute",
                 6: "LCV absolute", 7: "BVCK_absolute", 8: "Mahalanobis absolute",
                 10: "ICV"}

and also two constants for saying whether the estimate is signed or it's an
absolute value::

  SIGNED = 0
  ABSOLUTE = 1

Example of usage
================

Here we will walk through a bit longer example of how to use the reliability
estimate module (`reliability-long.py`_, uses `prostate.tab`_):.

.. literalinclude:: code/reliability-long.py
    :lines: 1-16

After loading the Orange library we open out dataset. We chose to work with
the kNNLearner, that also works on regression problems. Create out reliability
estimate learner and test it with cross validation. 
Estimates are then compared using Pearson's coefficient to the prediction error.
The p-values are also computed::

  Estimate               r       p
  SAvar absolute        -0.077   0.454
  SAbias signed         -0.165   0.105
  SAbias absolute       -0.099   0.333
  BAGV absolute          0.104   0.309
  CNK signed             0.233   0.021
  CNK absolute           0.057   0.579
  LCV absolute           0.069   0.504
  BVCK_absolute          0.092   0.368
  Mahalanobis absolute   0.091   0.375

.. literalinclude:: code/reliability-long.py
    :lines: 18-28

Outputs::

  Estimate               r       p
  BAGV absolute          0.126   0.220
  CNK signed             0.233   0.021
  CNK absolute           0.057   0.579
  LCV absolute           0.069   0.504
  BVCK_absolute          0.105   0.305
  Mahalanobis absolute   0.091   0.375


As you can see in the above code you can also chose with reliability estimation
method do you want to use. You might want to do this to reduce computation time 
or because you think they don't perform good enough.

.. literalinclude:: code/reliability-long.py
    :lines: 30-43

In this part of the example we have a usual prediction problem, we have a 
training part of dataset and testing part of dataset. We wrap out learner and
choose to use internal cross validation and no other reliability estimate. 

Internal cross validation is performed on the training part of dataset and it
chooses the best method. Now this method is training on whole training dataset
and used on test dataset to estimate the reliabiliy.

We are interested in the most reliable examples in our testing dataset. We
extract the estimates and id's, sort them and output them.

.. _reliability-run.py: code/reliability-run.py
.. _housing.tab: code/housing.tab

.. _reliability-long.py: code/reliability-long.py
.. _prostate.tab: code/prostate.tab

"""
import Orange

import random
import statc
import math
import warnings

# Labels and final variables
labels = ["SAvar", "SAbias", "BAGV", "CNK", "LCV", "BVCK", "Mahalanobis", "ICV"]

# All the estimators calculation constants
DO_SA = 0
DO_BAGV = 1
DO_CNK = 2
DO_LCV = 3
DO_BVCK = 4
DO_MAHAL = 5

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
ICV_METHOD = 10

# Type of estimator constant
SIGNED = 0
ABSOLUTE = 1

# Names of all the estimator methods
METHOD_NAME = {0: "SAvar absolute", 1: "SAbias signed", 2: "SAbias absolute",
               3: "BAGV absolute", 4: "CNK signed", 5: "CNK absolute",
               6: "LCV absolute", 7: "BVCK_absolute", 8: "Mahalanobis absolute",
               10: "ICV"}

class Learner:
    """
    Reliability estimation wrapper around a learner we want to test.
    Different reliability estimation algorithms can be used on the
    chosen learner. This learner works as any other and can be used as one.
    The only difference is when the classifier is called with a given
    example instead of only return the value and probabilities, it also
    attaches a list of reliability estimates to 
    :data:`probabilities.reliability_estimate`.
    Each reliability estimate consists of a tuple 
    (estimate, signed_or_absolute, method).
    
    :param e: List of possible e value for SAvar and SAbias reliability estimate
    :type e: list of floats
    
    :param m: Number of bagged models to be used with BAGV estimate
    :type m: int
    
    :param cnk_k: Number of nearest neighbours used in CNK estimate
    :type cnk_k: int
    
    :param lcv_k: Number of nearest neighbours used in LCV estimate
    :type cnk_k: int
    
    :param icv: Use internal cross-validation. Internal cross-validation calculates all
                the reliability estimates on the training data using cross-validation.
                Then it chooses the most successful estimate and uses it on the test
                dataset.
    :type icv: boolean
    
    :param use: List of booleans saying which reliability methods should be
                used in our experiment and which not.
    :type use: list of booleans
    
    :param use_with_icv: List of booleans saying which reliability methods
                         should be used in inside cross validation and
                         which not.
    
    :type use_with_icv: list of booleans
    
    :rtype: :class:`Orange.evaluation.reliability.Learner`
    """
    def __init__(self, learner, name="Reliability estimation",
                 e=[0.01, 0.1, 0.5, 1.0, 2.0], m=50, cnk_k=5, lcv_k=5,
                 use=[True, True, True, True, True, True],
                 use_with_icv=[True, True, True, True, True, True],
                 icv=False, **kwds):
        self.__dict__.update(kwds)
        self.name = name
        self.e = e
        self.m = m
        self.cnk_k = cnk_k
        self.lcv_k = lcv_k
        self.use = use
        self.use_with_icv = use_with_icv
        self.icv = icv
        self.learner = learner
        
    
    def __call__(self, examples, weight=None, **kwds):
        """Learn from the given table of data instances.
        
        :param instances: Data instances to learn from.
        :type instances: Orange.data.Table
        :param weight: Id of meta attribute with weights of instances
        :type weight: integer
        :rtype: :class:`estimator.Classifier`
        """
        return Classifier(examples, self.learner, self.e, self.m, self.cnk_k,
                          self.lcv_k, self.icv, self.use, self.use_with_icv)
    
    def internal_cross_validation(self, examples, folds=10):
        """ Performs internal cross validation (as in Automatic selection of
        reliability estimates for individual regression predictions,
        Zoran Bosnic 2010) and return id of the method
        that scored best on this data. """
        cv_indices = Orange.core.MakeRandomIndicesCV(examples, folds)
        
        list_of_rs = []
        
        sum_of_rs = defaultdict(float)
        
        for fold in xrange(folds):
            data = examples.select(cv_indices, fold)
            res = Orange.evaluation.testing.crossValidation([self], data)
            results = get_pearson_r(res)
            for r, _, _, method in results:
                sum_of_rs[method] += r
        sorted_sum_of_rs = sorted(sum_of_rs.items(), key=lambda estimate: estimate[1], reverse=True)
        return sorted_sum_of_rs[0][0]
    
    labels = ["SAvar", "SAbias", "BAGV", "CNK", "LCV", "BVCK", "Mahalanobis", "ICV"]

class Classifier:
    def __init__(self, examples, learner, e, m, cnk_k, lcv_k, icv, use, use_with_icv, **kwds):
        self.__dict__.update(kwds)
        self.examples = examples
        self.learner = learner
        self.e = e
        self.m = m
        self.cnk_k = cnk_k
        self.lcv_k = lcv_k
        self.icv = icv
        self.use = use
        self.use_with_icv = use_with_icv
        
        self.icv_method = -1
        
        # Train the learner with original data
        self.classifier = learner(examples)
        
        # Do the internal cross validation if needed
        if self.icv:
            estimator = Learner(learner=self.learner, e=self.e, m=self.m,
                                cnk_k=self.cnk_k, lcv_k=self.lcv_k, use=self.use_with_icv)
            res = Orange.evaluation.testing.cross_validation([estimator], self.examples)
            result_list = get_pearson_r(res)
            
            best_method = max (result_list)
            
            self.icv_method = best_method[3]
            self.icv_signed_or_absolute = best_method[2]
            
        # Sensitivity Analysis
        
        if self.use[DO_SA] or self.icv_method in [SAVAR_ABSOLUTE, SABIAS_SIGNED, SABIAS_ABSOLUTE]:
        
            #Get the maximum and minimum value of class
            self.min_value = self.max_value = examples[0].getclass().value
            for ex in examples:
                if ex.getclass().value > self.max_value:
                    self.max_value = ex.getclass().value
                if ex.getclass().value < self.min_value:
                    self.min_value = ex.getclass().value
        
        if self.use[DO_CNK] or self.use[DO_BVCK] or self.use[DO_LCV] or self.icv_method in [CNK_SIGNED, CNK_ABSOLUTE, BVCK_ABSOLUTE, LCV_ABSOLUTE]:
            # Save the neighbours constructors
            # Find k nearest neighbors
            nnc = Orange.classification.knn.FindNearestConstructor()
            nnc.distanceConstructor = Orange.distances.EuclideanConstructor()
            
            self.did = Orange.core.newmetaid()
            self.nn = nnc(self.examples, 0, self.did)
            
        # Variance of bagged model
        
        if self.use[DO_BAGV] or self.use[DO_BVCK] or self.icv_method in [BAGV_ABSOLUTE, BVCK_ABSOLUTE]:
        # Bagged classifiers
            self.classifiers = []
            
            # Create bagged classifiers using sampling with replacement
            for _ in xrange(m):
                selection = [random.randrange(len(examples)) for _ in xrange(len(examples))]
                data = examples.getitems(selection)
                self.classifiers.append(self.learner(data))
                
        if self.use[DO_MAHAL] or self.icv_method in [MAHAL_ABSOLUTE]:
            # Save the Mahalanobis distance constructor
            nnm = Orange.classification.knn.FindNearestConstructor()
            nnm.distanceConstructor = Orange.distances.MahalanobisConstructor()
            
            self.mid = Orange.core.newmetaid()
            self.nnm = nnm(self.examples, 0, self.mid)
        
    
    def __call__(self, example, result_type=Orange.core.GetValue):
        """
        Classifiy and estimate a new instance. When you chose 
        Orange.core.GetBoth or Orange.core.getProbabilities, you can access 
        the reliability estimates inside probabilities.reliability_estimate.
        
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """
        predicted, probabilities = self.classifier(example, Orange.core.GetBoth)
        
        # Create a placeholder for estimates
        if probabilities is None:
            probabilities = Orange.statistics.distribution.Continuous()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probabilities.reliability_estimate = []
        
        # Sensitivity analysis
        
        SAvar = 0
        SAbias = 0
        
        if self.use[DO_SA] or self.icv_method in [SAVAR_ABSOLUTE, SABIAS_ABSOLUTE, SABIAS_SIGNED]:
            
            # Create new dataset
            r_data = Orange.data.Table(self.examples)
            
            # Create new example
            modified_example = Orange.data.Instance(example)
                   
            # Append it to the data
            r_data.append(modified_example)
             
            # Calculate SAvar & SAbias
            SAvar = SAbias = 0
            
            for eps in self.e:
                # +epsilon
                r_data[-1].setclass(predicted.value + eps*(self.max_value - self.min_value))
                c = self.learner(r_data)
                k_plus = c(example, Orange.core.GetValue)
                
                # -epsilon
                r_data[-1].setclass(predicted.value - eps*(self.max_value - self.min_value))
                c = self.learner(r_data)
                k_minus = c(example, Orange.core.GetValue)
                #print len(r_data)
                #print eps*(self.max_value - self.min_value)
                #print k_plus
                #print k_minus
                # calculate part SAvar and SAbias
                SAvar += k_plus.value - k_minus.value
                SAbias += k_plus.value + k_minus.value - 2*predicted.value
            
            SAvar /= len(self.e)
            SAbias /= 2*len(self.e)
            probabilities.reliability_estimate.append( (SAvar, ABSOLUTE, SAVAR_ABSOLUTE) )
            probabilities.reliability_estimate.append( (SAbias, SIGNED, SABIAS_SIGNED) )
            probabilities.reliability_estimate.append( (SAbias, ABSOLUTE, SABIAS_ABSOLUTE) )
        
        
        #print SAvar
        #print SAbias
        
        # Bagging variance
        
        BAGV = 0
        
        if self.use[DO_BAGV] or self.use[DO_BVCK] or self.icv_method in [BAGV_ABSOLUTE, BVCK_ABSOLUTE]:
        
            # Calculate the bagging variance
            bagged_values = [c(example, Orange.core.GetValue).value for c in self.classifiers]
            
            k = sum(bagged_values) / len(bagged_values)
            
            BAGV = sum( (bagged_value - k)**2 for bagged_value in bagged_values) / len(bagged_values)
            
            if self.use[DO_BAGV]:
                probabilities.reliability_estimate.append( (BAGV, ABSOLUTE, BAGV_ABSOLUTE) )
        
        # For each of the classifiers
        #for c in self.classifiers:
        #    baggedPrediction = c(example, orange.GetValue).value
        #    BAGV += (baggedPrediction.value - predicted.value)**2
        #
        #BAGV /= self.m
        
        # C_neighbours -k
        
        CNK = 0
        
        if self.use[DO_CNK] or self.use[DO_BVCK] or self.icv_method in [CNK_SIGNED, CNK_ABSOLUTE, BVCK_ABSOLUTE]:
        
            CNK = 0
                    
            # Find k nearest neighbors
            
            knn = [ex for ex in self.nn(example, self.cnk_k)]
            
            # average label of neighbors
            for ex in knn:
                CNK += ex.getclass().value
            
            CNK /= self.cnk_k
            CNK -= predicted.value
            
            if self.use[DO_CNK]:
                probabilities.reliability_estimate.append( (CNK, SIGNED, CNK_SIGNED) )
                probabilities.reliability_estimate.append( (CNK, ABSOLUTE, CNK_ABSOLUTE) )
        
        # Calculate local cross-validation reliability estimate
        LCV = 0
        
        if self.use[DO_LCV] or self.icv_method == LCV_ABSOLUTE:
        
            LCVer = 0
            LCVdi = 0
            
            
            # Find k nearest neighbors
            
            knn = [ex for ex in self.nn(example, self.lcv_k)]
            
            # leave one out of prediction error
            for i in xrange(len(knn)):
                train = knn[:]
                del train[i]
                
                c = self.learner(Orange.data.Table(train))
                
                cc = c(knn[i], Orange.core.GetValue)
                
                e = abs(knn[i].getclass().value - cc.value)
                
                LCVer += e * math.exp(-knn[i][self.did])
                LCVdi += math.exp(-knn[i][self.did])
            
            
            LCV = LCVer / LCVdi if LCVdi != 0 else 0
            
            probabilities.reliability_estimate.append( (LCV, ABSOLUTE, LCV_ABSOLUTE) )
        
        # BVCK
        
        BVCK = 0
        
        if self.use[DO_BVCK] or self.icv_method == BVCK_ABSOLUTE:
            
            BVCK = (BAGV + abs(CNK)) / 2
            
            probabilities.reliability_estimate.append( (BVCK, ABSOLUTE, BVCK_ABSOLUTE) )
        
        # Mahalanobis distance to 3 closest neighbours
        
        mahalanobis_distance = 0
        
        if self.use[DO_MAHAL] or self.icv_method == MAHAL_ABSOLUTE:
        
            mahalanobis_distance = sum(ex[self.mid].value for ex in self.nnm(example, 3))
            
            probabilities.reliability_estimate.append( (mahalanobis_distance, ABSOLUTE, MAHAL_ABSOLUTE) )
            
        #probabilities.reliability_estimate = [SAvar, SAbias, BAGV, CNK, LCV, BVCK, mahalanobis_distance]
        
        if self.icv:
            method = [SAvar, SAbias, SAbias, BAGV, CNK, CNK, LCV, BVCK, mahalanobis_distance]
            ICV = method[self.icv_method]
            probabilities.reliability_estimate.append( (ICV, self.icv_signed_or_absolute, 10, self.icv_method))
        
        if result_type == Orange.core.GetValue:
            return predicted
        elif result_type == Orange.core.GetProbabilities:
            return probabilities
        else:
            return predicted, probabilities

def get_reliability_estimation_list(res, i):
    return [result.probabilities[0].reliability_estimate[i][0] for result in res.results], res.results[0].probabilities[0].reliability_estimate[i][1], res.results[0].probabilities[0].reliability_estimate[i][2]

def get_prediction_error_list(res):
    return [result.actualClass - result.classes[0] for result in res.results]

def get_pearson_r(res):
    """
    Returns Pearsons coefficient between the prediction error and each of the
    used reliability estimates. Function also return the p-value of each of
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

def get_pearson_r_by_iterations(res):
    """
    Returns average Pearsons coefficient over all folds between prediction error
    and each of the used estimates.
    """
    results_by_fold = Orange.evaluation.scoring.split_by_iterations(res)
    number_of_estimates = len(res.results[0].probabilities[0].reliability_estimate)
    number_of_examples = len(res.results)
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
    ps = [p_value_from_r(r, number_of_examples) for r in results]
    
    return zip(results, ps, sig, method_list)

def p_value_from_r(r, n):
    """
    Calculate p-value from the paerson coefficient and the sample size.
    """
    df = n - 2
    t = r * (df /((-r + 1.0 + 1e-30) * (r + 1.0 + 1e-30)) )**0.5
    return statc.betai (df * 0.5, 0.5, df/(df + t*t))

