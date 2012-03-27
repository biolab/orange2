try:
    import unittest2 as unittest
except:
    import unittest

from Orange.data.preprocess import (AddCensorWeight as Preprocessor_addCensorWeight,
         AddClassNoise as  Preprocessor_addClassNoise,
         AddClassWeight as Preprocessor_addClassWeight,
         AddGaussianClassNoise as  Preprocessor_addGaussianClassNoise,
         AddGaussianNoise as Preprocessor_addGaussianNoise,
         AddMissing as Preprocessor_addMissing,
         AddMissingClasses as Preprocessor_addMissingClasses,
         AddNoise as Preprocessor_addNoise,
         Discretize as Preprocessor_discretize,
         Drop as Preprocessor_drop,
         DropMissing as Preprocessor_dropMissing,
         DropMissingClasses as Preprocessor_dropMissingClasses,
         Filter as Preprocessor_filter,
         Ignore as Preprocessor_ignore,
         ImputeByLearner as Preprocessor_imputeByLearner,
         RemoveDuplicates as Preprocessor_removeDuplicates,
         Select as Preprocessor_select,
         Shuffle as Preprocessor_shuffle,
         Take as Preprocessor_take,
         TakeMissing as Preprocessor_takeMissing,
         TakeMissingClasses as Preprocessor_takeMissingClasses,
         DiscretizeEntropy as Preprocessor_discretizeEntropy,
         RemoveContinuous as Preprocessor_removeContinuous,
         RemoveDiscrete as Preprocessor_removeDiscrete,
         Continuize as Preprocessor_continuize,
         Impute as Preprocessor_impute,
         FeatureSelection as Preprocessor_featureSelection,
         RFE as Preprocessor_RFE,
         Sample as Preprocessor_sample,
         PreprocessorList as Preprocessor_preprocessorList,
         )

import orange

import Orange.testing.testing as testing

@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestAddClassNoise(testing.PreprocessorTestCase):
    """ Test Preprocessor_addClassNoise 
    """
    PREPROCESSOR = Preprocessor_addClassNoise


@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestAddClassWeight(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_addClassWeight


@testing.datasets_driven(datasets=testing.REGRESSION_DATASETS)
class TestAddGaussianClassNoise(testing.PreprocessorTestCase):
    """ Test Preprocessor_addGaussianClassNoise
    """
    PREPROCESSOR = Preprocessor_addGaussianClassNoise


@testing.datasets_driven
class TestAddGaussianNoise(testing.PreprocessorTestCase):
    """ Test Preprocessor_addGaussianNoise
    """
    PREPROCESSOR = Preprocessor_addGaussianNoise


@testing.datasets_driven
class TestAddMissing(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_addMissing


@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS + \
                         testing.REGRESSION_DATASETS)
class TestAddMissingClasses(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_addMissingClasses


@testing.datasets_driven
class TestAddMissing(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_addMissing


@testing.datasets_driven
class TestAddNoise(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_addNoise


@testing.datasets_driven
class TestDiscretizeEquiN(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_discretize(method=orange.EquiNDiscretization())


@testing.datasets_driven
class TestDiscretizeEquiDist(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_discretize(method=orange.EquiDistDiscretization())


@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestDiscretizeEntropy(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_discretize(method=orange.EntropyDiscretization())

##########
# This crashes with std::bad_cast
##########
#@testing.datasets_driven
#class TestDiscretizeBiModal(testing.PreprocessorTestCase):
#    PREPROCESSOR = Preprocessor_discretize(method=orange.BiModalDiscretization())

##### 
# Pickling throws a segfault
#####
#@testing.datasets_driven
#class TestFilter(testing.PreprocessorTestCase):
#    PREPROCESSOR = Preprocessor_filter(filter=orange.Filter_random(prob = 0.7, randomGenerator = 24))

@testing.datasets_driven
class TestDropMissingClasses(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_dropMissingClasses

@testing.datasets_driven
class TestIgnore(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_ignore

@testing.datasets_driven
class TestImputeByLearner(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_imputeByLearner(learner=orange.MajorityLearner())

@testing.datasets_driven(datasets=testing.ALL_DATASETS + ["lenses"])
class TestRemoveDuplicates(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_removeDuplicates

@testing.datasets_driven
class TestSelect(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_select

@testing.datasets_driven
class TestShuffle(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_shuffle

@testing.datasets_driven
class TestTake(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_take

@testing.datasets_driven
class TestTakeMissing(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_takeMissing

@testing.datasets_driven
class TestTakeMissingClasses(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_takeMissingClasses

@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestDiscretizeEntropy(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_discretizeEntropy

@testing.datasets_driven
class TestRemoveContinuous(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_removeContinuous

@testing.datasets_driven
class TestRemoveDiscrete(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_removeDiscrete

@testing.datasets_driven
class TestContinuize(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_continuize

@testing.datasets_driven
class TestImpute(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_impute

@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS + \
                         testing.REGRESSION_DATASETS)
class TestFeatureSelection(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_featureSelection

@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestRFE(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_RFE

@testing.datasets_driven
class TestSample(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_sample

@testing.datasets_driven
class TestSelect(testing.PreprocessorTestCase):
    PREPROCESSOR = Preprocessor_preprocessorList(preprocessors=[Preprocessor_sample, Preprocessor_takeMissing])

if __name__ == "__main__":
    unittest.main()


