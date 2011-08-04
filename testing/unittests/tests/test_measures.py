from Orange.misc import testing
from Orange.misc.testing import datasets_driven, test_on_data
from Orange.feature import scoring
import unittest    
    

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS,
                 preprocess=testing.DISCRETIZE_DOMAIN)
class TestMeasureAttr_GainRatio(testing.MeasureAttributeTestCase):
    MEASURE = scoring.GainRatio()
    
    
@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS,
                 preprocess=testing.DISCRETIZE_DOMAIN)
class TestMeasureAttr_InfoGain(testing.MeasureAttributeTestCase):
    MEASURE = scoring.InfoGain()
    

# TODO: Relevance, Cost

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS,
                 preprocess=testing.DISCRETIZE_DOMAIN)
class TestMeasureAttr_Distance(testing.MeasureAttributeTestCase):
    MEASURE = scoring.Distance()

    
@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS,
                 preprocess=testing.DISCRETIZE_DOMAIN)
class TestMeasureAttr_MDL(testing.MeasureAttributeTestCase):
    MEASURE = scoring.MDL()


@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestMeasureAttr_Relief(testing.MeasureAttributeTestCase):
    MEASURE = scoring.Relief()


@datasets_driven(datasets=testing.REGRESSION_DATASETS,
                 preprocess=testing.DISCRETIZE_DOMAIN)
class TestMeasureAttr_MSE(testing.MeasureAttributeTestCase):
    MEASURE = scoring.MSE()
    

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestScoringUtils(testing.DataTestCase):
    @test_on_data
    def test_order_attrs(self, dataset):
        order = scoring.OrderAttributes(scoring.Relief())
        orderes_attrs = order(dataset, 0)
        
    @test_on_data
    def test_score_all(self, dataset):
        scoring.score_all(dataset, measure=scoring.Relief())
         
        
if __name__ == "__main__":
    unittest.main()