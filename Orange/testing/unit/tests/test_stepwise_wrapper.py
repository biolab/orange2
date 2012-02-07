import unittest

import Orange
from Orange.misc import testing

from Orange.classification import bayes
from Orange.classification import wrappers


@testing.datasets_driven(datasets=["iris", "lenses"])
class TestStepwise(testing.LearnerTestCase):
    LEARNER = wrappers.StepwiseLearner(learner=bayes.NaiveLearner())
    @testing.test_on_data
    def test_learner_on(self, dataset):
        if len(dataset) > 100:
            dataset = dataset.select(
                Orange.data.sample.SubsetIndices2(n=100)(dataset)
                )
        testing.LearnerTestCase.test_learner_on(self, dataset)
        
if __name__ == "__main__":
    unittest.main()
