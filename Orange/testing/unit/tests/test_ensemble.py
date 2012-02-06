from Orange.misc import testing
from Orange.misc.testing import datasets_driven, test_on_datasets

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestBoosting(testing.LearnerTestCase):    
    def setUp(self): 
        import orngEnsemble, orngTree
        self.learner = orngEnsemble.BoostedLearner(orngTree.TreeLearner)
        
    @test_on_datasets(datasets=["iris"])
    def test_pickling_on(self, dataset):
        testing.LearnerTestCase.test_pickling_on(self, dataset)
        
@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS +\
                 testing.REGRESSION_DATASETS)
class TestBagging(testing.LearnerTestCase):
    def setUp(self): 
        import orngEnsemble, orngTree
        self.learner = orngEnsemble.BaggedLearner(orngTree.TreeLearner)

    @test_on_datasets(datasets=["iris"])
    def test_pickling_on(self, dataset):
        testing.LearnerTestCase.test_pickling_on(self, dataset)

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestRandomForest(testing.LearnerTestCase):
    def setUp(self): 
        import orngEnsemble, orngTree
        self.learner = orngEnsemble.RandomForestLearner()
        
    @test_on_datasets(datasets=["iris"])
    def test_pickling_on(self, dataset):
        testing.LearnerTestCase.test_pickling_on(self, dataset)
        
        
if __name__ == "__main__":
    import unittest
    unittest.main()
                