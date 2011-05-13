from Orange.misc import testing

@testing.expand_tests
class TestBoosting(testing.LearnerTestCase):
    FLAGS = testing.TEST_ALL_CLASSIFICATION
    
    def setUp(self): 
        import orngEnsemble, orngTree
        self.learner = orngEnsemble.BoostedLearner(orngTree.TreeLearner)
        
        
@testing.expand_tests
class TestBagging(testing.LearnerTestCase):
    FLAGS = testing.TEST_ALL
    
    def setUp(self): 
        import orngEnsemble, orngTree
        self.learner = orngEnsemble.BaggedLearner(orngTree.TreeLearner)


@testing.expand_tests
class TestRandomForest(testing.LearnerTestCase):
    FLAGS = testing.TEST_ALL
    
    def setUp(self): 
        import orngEnsemble, orngTree
        self.learner = orngEnsemble.RandomForestLearner()
        
        
if __name__ == "__main__":
    import unittest
    unittest.run()
                