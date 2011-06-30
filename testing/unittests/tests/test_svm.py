from Orange.classification.svm import SVMLearner, MeasureAttribute_SVMWeights, LinearLearner, RFE
from Orange.classification.svm.kernels import BagOfWords, RBFKernelWrapper
import Orange.misc.testing as testing
import orange

#LinerSVMTestCase = test_case_learner(SVMLearner(kernel_type=SVMLearner.Linear))
#RBFSVMTestCase = test_case_learner(SVMLearner(kernel_type=SVMLearner.RBF))

@testing.expand_tests
class LinearSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-lin", kernel_type=SVMLearner.Linear)
    
@testing.expand_tests
class PolySVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-poly", kernel_type=SVMLearner.Polynomial)
    
@testing.expand_tests
class RBFSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-RBF", kernel_type=SVMLearner.RBF)
    
@testing.expand_tests
class SigmoidSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-sig", kernel_type=SVMLearner.Sigmoid)
    
@testing.expand_tests
class BagOfWordsSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner(name="svm-bow", kernel_type=SVMLearner.Custom, kernelFunc=BagOfWords())
    
@testing.expand_tests
class CustomWrapperSVMTestCase(testing.LearnerTestCase):
    LEARNER = SVMLearner
    
    @testing.test_on_data
    def test_learner_on(self, data):
        """ Test custom kernel wrapper
        """
        self.LEARNER = self.LEARNER(kernel_type=SVMLearner.Custom,
                                    kernelFunc=RBFKernelWrapper(orange.ExamplesDistanceConstructor_Euclidean(data), gamma=0.5))
        
        c = self.LEARNER(data)
    
@testing.expand_tests
class TestLinLearner(testing.LearnerTestCase):
    LEARNER = LinearLearner
    
    
@testing.expand_tests
class TestMeasureAttr_LinWeights(testing.TestMeasureAttribute):
    MEASURE = MeasureAttribute_SVMWeights()
    

@testing.expand_tests
class TestRFE(testing.BaseTestOnData):
    FLAGS = testing.TEST_ALL_CLASSIFICATION
    
    @testing.test_on_data
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