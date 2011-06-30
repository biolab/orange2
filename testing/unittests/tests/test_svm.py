from Orange.classification.svm import SVMLearner, MeasureAttribute_SVMWeights, LinearLearner, RFE
from Orange.classification.svm.kernels import BagOfWords, RBFKernelWrapper
from Orange.misc import testing
from Orange.misc.testing import datasets_driven, test_on_datasets
import orange


datasets = testing.CLASSIFICATION_DATASETS + testing.REGRESSION_DATASETS
@datasets_driven(datasets=datasets)
class LinearSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-lin", kernel_type=SVMLearner.Linear)
    
    
@datasets_driven(datasets=datasets)
class PolySVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-poly", kernel_type=SVMLearner.Polynomial)
    
    
@datasets_driven(datasets=datasets)
class RBFSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-RBF", kernel_type=SVMLearner.RBF)
    
    
@datasets_driven(datasets=datasets)
class SigmoidSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-sig", kernel_type=SVMLearner.Sigmoid)
    
    
@datasets_driven(datasets=datasets)
class BagOfWordsSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-bow", kernel_type=SVMLearner.Custom, kernelFunc=BagOfWords())
    
    
@datasets_driven(datasets=datasets)
class CustomWrapperSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner
    
    @test_on_datasets(datasets=datasets)
    def test_learner_on(self, data):
        """ Test custom kernel wrapper
        """
        # Need the data for ExamplesDistanceConstructor_Euclidean  
        self.learner = self.LEARNER(kernel_type=SVMLearner.Custom,
                                    kernelFunc=RBFKernelWrapper(orange.ExamplesDistanceConstructor_Euclidean(data), gamma=0.5))
        
        testing.LearnerTestCase.test_learner_on(self, data)
    
    
@datasets_driven(datasets=datasets)
class TestLinLearner(testing.LearnerTestCase):
    LEARNER = LinearLearner
    
    
@datasets_driven(datasets=datasets)
class TestMeasureAttr_LinWeights(testing.MeasureAttributeTestCase):
    MEASURE = MeasureAttribute_SVMWeights()


@datasets_driven(datasets=datasets)
class TestRFE(testing.DataTestCase):
    @datasets_driven
    def test_rfe_on(self, data):
        rfe = RFE()
        num_selected = min(5, len(data.domain.attributes))
        reduced = rfe(data, num_selected)
        self.assertEqual(len(reduced.domain.attributes), num_selected)
        scores = rfe.getAttrScores(data, stopAt=num_selected)
        self.assertEqual(len(data.domain.attributes) - num_selected, len(scores))
        self.assertTrue(set(reduced.domain.attributes).isdisjoint(scores.keys()))
        
    def test_pickle(self):
        import cPickle
        rfe = RFE()
        cPickle.loads(cPickle.dumps(rfe))

if __name__ == "__main__":
    import unittest
    unittest.main()