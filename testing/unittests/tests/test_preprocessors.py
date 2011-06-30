import unittest

from Orange.preprocess import (Preprocessor_addCensorWeight,
         Preprocessor_addClassNoise,
         Preprocessor_addClassWeight, 
         Preprocessor_addGaussianClassNoise,
         Preprocessor_addGaussianNoise,
         Preprocessor_addMissing,
         Preprocessor_addMissingClasses,
         Preprocessor_addNoise,
         Preprocessor_discretize,
         Preprocessor_drop,
         Preprocessor_dropMissing,
         Preprocessor_dropMissingClasses,
         Preprocessor_filter,
         Preprocessor_ignore,
         Preprocessor_imputeByLearner,
         Preprocessor_removeDuplicates,
         Preprocessor_select,
         Preprocessor_shuffle,
         Preprocessor_take,
         Preprocessor_takeMissing,
         Preprocessor_takeMissingClasses,
         Preprocessor_discretizeEntropy,
         Preprocessor_removeContinuous,
         Preprocessor_removeDiscrete,
         Preprocessor_continuize,
         Preprocessor_impute,
         Preprocessor_featureSelection,
         Preprocessor_RFE,
         Preprocessor_sample,
         Preprocessor_preprocessorList,
         )

import orange

import Orange.misc.testing as testing
    
@testing.expand_tests
class TestAddClassNoise(testing.TestPreprocessor):
    """ Test Preprocessor_addClassNoise 
    """
    PREPROCESSOR = Preprocessor_addClassNoise
    FLAGS = testing.TEST_CLASSIFICATION + testing.TEST_PICKLE


#class TestAddClassWeight(tesring.TestPreprocessor):
#    PREPROCESSOR = Preprocessor_addClassWeight

@testing.expand_tests
class TestAddGaussianClassNoise(testing.TestPreprocessor):
    """ Test Preprocessor_addGaussianClassNoise
    """
    PREPROCESSOR = Preprocessor_addGaussianClassNoise
    FLAGS = testing.TEST_REGRESSION | testing.TEST_PICKLE

@testing.expand_tests
class TestAddGaussianNoise(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_addGaussianNoise
    FLAGS = testing.TEST_ALL
    
@testing.expand_tests
class TestAddMissing(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_addMissing

@testing.expand_tests
class TestAddMissingClasses(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_addMissingClasses
    
@testing.expand_tests
class TestAddMissing(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_addMissing

@testing.expand_tests
class TestAddNoise(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_addNoise
    
@testing.expand_tests
class TestDiscretizeEquiN(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_discretize(method=orange.EquiNDiscretization())
    FLAGS = testing.TEST_CLASSIFICATION + testing.CONTINUIZE_DOMAIN

@testing.expand_tests 
class TestDiscretizeEquiDist(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_discretize(method=orange.EquiDistDiscretization())
    
@testing.expand_tests
class TestDiscretizeEntropy(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_discretize(method=orange.EntropyDiscretization())
    FLAGS = testing.TEST_CLASSIFICATION + testing.TEST_PICKLE

# This crashes with std::bad_cast
#@testing.expand_tests
#class TestDiscretizeBiModal(testing.TestPreprocessor):
#    PREPROCESSOR = Preprocessor_discretize(method=orange.BiModalDiscretization())
    
# Pickling throws a segfault
@testing.expand_tests
class TestFilter(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_filter(filter=orange.Filter_random(prob = 0.7, randomGenerator = 24))
    FLAGS = testing.TEST_ALL - testing.TEST_PICKLE
    
@testing.expand_tests
class TestDropMissingClasses(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_dropMissingClasses
    
@testing.expand_tests
class TestIgnore(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_ignore
    
@testing.expand_tests
class TestImputeByLearner(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_imputeByLearner(learner=orange.MajorityLearner())
    
@testing.expand_tests
class TestRemoveDuplicates(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_removeDuplicates
    
@testing.expand_tests
class TestSelect(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_select
    
@testing.expand_tests
class TestShuffle(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_shuffle
    
@testing.expand_tests
class TestTake(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_take
    
@testing.expand_tests
class TestTakeMissing(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_takeMissing
    
@testing.expand_tests
class TestTakeMissingClasses(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_takeMissingClasses
    
@testing.expand_tests
class TestDiscretizeEntropy(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_discretizeEntropy
    FLAGS = testing.TEST_CLASSIFICATION + testing.TEST_PICKLE
    
@testing.expand_tests
class TestRemoveContinuous(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_removeContinuous
    
@testing.expand_tests
class TestRemoveDiscrete(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_removeDiscrete
    
@testing.expand_tests
class TestContinuize(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_continuize
    
@testing.expand_tests
class TestImpute(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_impute
    
@testing.expand_tests
class TestFeatureSelection(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_featureSelection
    
@testing.expand_tests
class TestRFE(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_RFE
    FLAGS = testing.TEST_CLASSIFICATION + testing.TEST_PICKLE
    
@testing.expand_tests
class TestSample(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_sample
    
@testing.expand_tests
class TestSelect(testing.TestPreprocessor):
    PREPROCESSOR = Preprocessor_preprocessorList(preprocessors=[Preprocessor_sample, Preprocessor_takeMissing])
    
if __name__ == "__main__":
    unittest.main()

    
         