from Orange.misc import testing
try:
    import unittest2 as unittest
except:
    import unittest

from orngLR import LogRegLearner, Univariate_LogRegLearner, StepWiseFSS, StepWiseFSS_Filter

def datasets_iter():
    for name, (data,) in testing.datasets_iter(testing.CLASSIFICATION_DATASETS):
        if len(data.domain.class_var.values) == 2:
            yield name, (data,)


@testing.data_driven(data_iter=datasets_iter())
class TestLogRegLearner(testing.LearnerTestCase):
    LEARNER = LogRegLearner
    @testing.test_on_data
    def test_learner_on(self, dataset):
        """ Test LogRegLearner.
        """
        if len(dataset) < len(dataset.domain):
            raise unittest.SkipTest("No enough examples")
        testing.LearnerTestCase.test_learner_on(self, dataset)


#@testing.data_driven(data_iter=datasets_iter())
#class TestUniLogRegLearner(testing.LearnerTestCase):
#    LEARNER = Univariate_LogRegLearner
#    @testing.test_on_data
#    def test_learner_on(self, dataset):
#        """ Test LogRegLearner.
#        """
#        if len(dataset) < len(dataset.domain.attributes):
#            raise unittest.SkipTest("No enough examples")
#        testing.LearnerTestCase.test_learner_on(self, dataset)

@testing.data_driven(data_iter=datasets_iter())
class TestStepWiseFSS(unittest.TestCase):
    @testing.test_on_data
    def test_stepwise_fss_on(self, dataset):
        """ Test StepWiseFSS.
        """
        if len(dataset) < len(dataset.domain):
            raise unittest.SkipTest("No enough examples")

        attrs = StepWiseFSS(dataset)
        new_dataset = StepWiseFSS_Filter(dataset)
        self.assertTrue([a1 == a2 for a1, a2 in zip(attrs, new_dataset.domain.attributes)])

if __name__ == "__main__":
    unittest.main()
