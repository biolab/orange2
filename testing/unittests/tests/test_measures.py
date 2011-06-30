from Orange.misc import testing
from Orange.feature import scoring
import unittest

datasets = testing.CLASSIFICATION_DATASETS + testing.REGRESSION_DATASETS    
    
@testing.datasets_driven(datasets=testing.REGRESSION_DATASETS,
                         preprocess=testing.DISCRETIZE_DOMAIN)
class TestMeasureAttr_MSE(testing.MeasureAttributeTestCase):
    MEASURE = scoring.MSE


@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS,
                         preprocess=testing.DISCRETIZE_DOMAIN)
class TestMeasureAttr_GainRatio(testing.MeasureAttributeTestCase):
    MEASURE = scoring.GainRatio
    
    
@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS,
                         preprocess=testing.DISCRETIZE_DOMAIN)
class TestMeasureAttr_InfoGain(testing.MeasureAttributeTestCase):
    MEASURE = scoring.InfoGain
    
    
@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestMeasureAttr_Relief(testing.MeasureAttributeTestCase):
    MEASURE = scoring.Relief
    
    
if __name__ == "__main__":
    unittest.main()