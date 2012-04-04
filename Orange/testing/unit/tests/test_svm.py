try:
    import unittest2 as unittest
except:
    import unittest
    
import Orange
from Orange.classification.svm import SVMLearner, SVMLearnerSparse, \
                            ScoreSVMWeights, LinearSVMLearner, \
                            MultiClassSVMLearner, RFE, \
                            get_linear_svm_weights, \
                            example_weighted_sum
from Orange.classification import svm                            
from Orange.classification.svm.kernels import BagOfWords, RBFKernelWrapper
from Orange.testing import testing
from Orange.testing.testing import datasets_driven, test_on_datasets, test_on_data
import orange

import copy

import pickle
import numpy as np

def multiclass_from1vs1(dec_values, class_var):
    n_class = len(class_var.values)
    votes = [0] * n_class
    p = 0
    for i in range(n_class - 1):
        for j in range(i + 1, n_class):
            val = dec_values[p]
            if val > 0:
                votes[i] += 1
            else:
                votes[j] += 1
            p += 1
    max_i = np.argmax(votes)
    return class_var(int(max_i))
    
    
def svm_test_binary_classifier(self, data):
    if isinstance(data.domain.class_var, Orange.feature.Discrete):
        # Test binary classifiers equivalence.  
        classes = data.domain.class_var.values
        bin_cls = []
        # Collect all binary classifiers
        for i in range(len(classes) - 1):
            for j in range(i + 1, len(classes)):
                bin_cls.append(self.classifier.get_binary_classifier(i, j))
                
        pickled_bin_cls = pickle.loads(pickle.dumps(bin_cls))
        
        indices = Orange.data.sample.SubsetIndices2(p0=0.2)
        sample = data.select(indices(data), 0)
        
        learner = copy.copy(self.learner)
        learner.probability = False 
        classifier_no_prob = learner(data)
        
        for inst in sample:
            d_val = list(self.classifier.get_decision_values(inst))
            d_val_b = [bc.get_decision_values(inst)[0] for bc in bin_cls]
            d_val_b1 = [bc.get_decision_values(inst)[0] for bc in pickled_bin_cls]
            for v1, v2, v3 in zip(d_val, d_val_b, d_val_b1):
                self.assertAlmostEqual(v1, v2, places=3)
                self.assertAlmostEqual(v1, v3, places=3)
            
            prediction_1 = classifier_no_prob(inst)
            d_val = classifier_no_prob.get_decision_values(inst)
            prediciton_2 = multiclass_from1vs1(d_val, classifier_no_prob.class_var)
            self.assertEqual(prediction_1, prediciton_2)
            
datasets = testing.CLASSIFICATION_DATASETS + testing.REGRESSION_DATASETS
@datasets_driven(datasets=datasets)
class LinearSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-lin", kernel_type=SVMLearner.Linear)

    @test_on_data
    def test_learner_on(self, dataset):
        testing.LearnerTestCase.test_learner_on(self, dataset)
        svm_test_binary_classifier(self, dataset)
        
    
    # Don't test on "monks" the coefs are really large and
    @test_on_datasets(datasets=["iris", "brown-selected", "lenses", "zoo"])
    def test_linear_classifier_weights_on(self, dataset):
        # Test get_linear_svm_weights
        classifier = self.LEARNER(dataset)
        weights = get_linear_svm_weights(classifier, sum=True)
        
        weights = get_linear_svm_weights(classifier, sum=False)
        
        n_class = len(classifier.class_var.values)
        
        def class_pairs(n_class):
            for i in range(n_class - 1):
                for j in range(i + 1, n_class):
                    yield i, j
                    
        l_map = classifier._get_libsvm_labels_map()
    
        for inst in dataset[:20]:
            dec_values = classifier.get_decision_values(inst)
            
            for dec_v, weight, rho, pair in zip(dec_values, weights,
                                    classifier.rho, class_pairs(n_class)):
                t_inst = Orange.data.Instance(classifier.domain, inst)                    
                dec_v1 = example_weighted_sum(t_inst, weight) - rho
                self.assertAlmostEqual(dec_v, dec_v1, 4)
                    
    @test_on_datasets(datasets=testing.REGRESSION_DATASETS)
    def test_linear_regression_weights_on(self, dataset):
        predictor = self.LEARNER(dataset)
        weights = get_linear_svm_weights(predictor)
        
        for inst in dataset[:20]:
            t_inst = Orange.data.Instance(predictor.domain, inst)
            prediction = predictor(inst)
            w_sum = example_weighted_sum(t_inst, weights)
            self.assertAlmostEqual(float(prediction), 
                                   w_sum - predictor.rho[0],
                                   places=4)
        

@datasets_driven(datasets=datasets)
class PolySVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-poly", kernel_type=SVMLearner.Polynomial)
    
    @test_on_data
    def test_learner_on(self, dataset):
        testing.LearnerTestCase.test_learner_on(self, dataset)
        svm_test_binary_classifier(self, dataset)
        

@datasets_driven(datasets=datasets)
class RBFSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-RBF", kernel_type=SVMLearner.RBF)
    
    @test_on_data
    def test_learner_on(self, dataset):
        testing.LearnerTestCase.test_learner_on(self, dataset)
        svm_test_binary_classifier(self, dataset)

@datasets_driven(datasets=datasets)
class SigmoidSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-sig", kernel_type=SVMLearner.Sigmoid)
    
    @test_on_data
    def test_learner_on(self, dataset):
        testing.LearnerTestCase.test_learner_on(self, dataset)
        svm_test_binary_classifier(self, dataset)


def to_sparse(data):
    domain = Orange.data.Domain([], data.domain.class_var)
    domain.add_metas(dict([(Orange.core.newmetaid(), v) for v in data.domain.attributes]))
    return Orange.data.Table(domain, data)

def sparse_data_iter():
    for name, (data, ) in testing.datasets_iter(datasets):
        yield name, (to_sparse(data), )

@testing.data_driven(data_iter=sparse_data_iter())
class SparseSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearnerSparse(name="svm-sparse")
    
    @test_on_data
    def test_learner_on(self, dataset):
        testing.LearnerTestCase.test_learner_on(self, dataset)
        svm_test_binary_classifier(self, dataset)


@datasets_driven(datasets=datasets)
class CustomWrapperSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner

    @test_on_data
    def test_learner_on(self, data):
        """ Test custom kernel wrapper
        """
        if data.domain.has_continuous_attributes():
            dist = orange.ExamplesDistanceConstructor_Euclidean(data)
        else:
            dist = orange.ExamplesDistanceConstructor_Hamming(data)
        self.learner = self.LEARNER(kernel_type=SVMLearner.Custom,
                                    kernel_func=RBFKernelWrapper(dist, gamma=0.5))

        testing.LearnerTestCase.test_learner_on(self, data)
        svm_test_binary_classifier(self, data)

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinLearner(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner
    
    
@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestMCSVMLearner(testing.LearnerTestCase):
    LEARNER = MultiClassSVMLearner


@datasets_driven(datasets=datasets)
class TestScoreSVMWeights(testing.MeasureAttributeTestCase):
    MEASURE = ScoreSVMWeights()
    
@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestScoreSVMWeightsWithMCSVM(testing.MeasureAttributeTestCase):
    MEASURE = ScoreSVMWeights(learner=MultiClassSVMLearner())
    
@datasets_driven(datasets=["iris"])
class TestRFE(testing.DataTestCase):
    @test_on_data
    def test_rfe_on(self, data):
        rfe = RFE()
        num_selected = min(5, len(data.domain.attributes))
        reduced = rfe(data, num_selected)
        self.assertEqual(len(reduced.domain.attributes), num_selected)
        scores = rfe.get_attr_scores(data, stop_at=num_selected)
        self.assertEqual(len(data.domain.attributes) - num_selected, len(scores))
        self.assertTrue(set(reduced.domain.attributes).isdisjoint(scores.keys()))

    def test_pickle(self):
        import cPickle
        rfe = RFE()
        copy = cPickle.loads(cPickle.dumps(rfe))

def load_tests(loader, tests, ignore):
    import doctest
    tests.addTests(doctest.DocTestSuite(svm))
    return tests

if __name__ == "__main__":
    unittest.main()
