from Orange.misc import testing
      
from Orange.classification import rules

@testing.test_on_data
def test_learner_on(self, dataset):
    testing.LearnerTestCase.test_learner_on(self, dataset)
    for r in self.classifier.rules:
        str = rules.rule_to_string(r, True)
            
@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestCN2Learner(testing.LearnerTestCase):
    def setUp(self):
        self.learner = rules.CN2Learner()
        
    test_learner_on = test_learner_on
    
@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestCN2UnorderedLearner(testing.LearnerTestCase):
    def setUp(self):
        self.learner = rules.CN2UnorderedLearner()
        
    test_learner_on = test_learner_on
    

    
    
            
