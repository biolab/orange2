from Orange.misc import testing

# TODO: test different prob estimators
@testing.expand_tests
class TestNaiveBayes(testing.LearnerTestCase):
    FLAGS = testing.TEST_ALL_CLASSIFICATION
    
    def setUp(self):
        import orngBayes
        self.learner = orngBayes.BayesLearner()
        
    @testing.test_on_data
    def test_learner_on(self, dataset):
        testing.LearnerTestCase.test_learner_on(self, dataset)
        # test __str__ method
        
        print_str = str(self.classifier)
        
        # test p method
        if dataset.domain.classVar:
            for ex in dataset:
                for cls in dataset.domain.classVar.values:
                    p = self.classifier.p(cls, ex)
                    
    