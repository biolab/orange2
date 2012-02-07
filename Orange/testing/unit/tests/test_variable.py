import unittest
import Orange
from Orange.misc import testing

class TestVariableMake(unittest.TestCase):
    def test_make(self):
        """ Test Variable.make
        """
        v1, s = Orange.data.variable.make("test_variable_a",
                                      Orange.feature.Type.Discrete,["a", "b"])
        v2, s = Orange.data.variable.make("test_variable_a",
                                      Orange.feature.Type.Discrete, ["a"], ["c"])
        self.assertIs(v2, v1)
        
        v3, s = Orange.data.variable.make("test_variable_a",
                          Orange.feature.Type.Discrete, ["a", "b", "c", "d"])
        self.assertIs(v3, v1)
        
        v4, s = Orange.data.variable.make("test_variable_a",
                                     Orange.feature.Type.Discrete, ["b"])
        self.assertIsNot(v4, v1)
        
        v5, s = Orange.data.variable.make("test_variable_a",
                             Orange.feature.Type.Discrete, None, ["c", "a"])
        self.assertIs(v5, v1)
        
        v6, s = Orange.data.variable.make("test_variable_a", 
                            Orange.feature.Type.Discrete, None, ["e"])
        self.assertIs(v6, v1)
        
        v7, s = Orange.data.variable.make("test_variable_a",
                                 Orange.feature.Type.Discrete, None, ["f"],
                                 Orange.feature.Descriptor.MakeStatus.NoRecognizedValues)
        self.assertIsNot(v7, v1)
        
        v8, s = Orange.data.variable.make("test_variable_a",
                                     Orange.feature.Type.Discrete,
                                     ["a", "b", "c", "d", "e"], None,
                                     Orange.feature.Descriptor.MakeStatus.OK)
        self.assertIsNot(v8, v1)
        
