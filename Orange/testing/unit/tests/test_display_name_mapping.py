try:
    import unittest2 as unittest
except:
    import unittest

import orange, Orange

class TestNameMapping(unittest.TestCase):
    def test_qualified_names(self):
        """ Test that qualified names of core C++ objects 
        map to the correct name in the Orange.* hierarchy.
          
        """
        for cls in orange.__dict__.values():
            if type(cls) == type:
                try:
                    cls2 = eval(cls.__module__ + "." + cls.__name__)
                except AttributeError as err:
                    self.assertTrue(False, cls.__module__ + "." + \
                                    cls.__name__ + " does not exist")

                self.assertEqual(cls2, cls)
#                if cls2 != cls:
#                    print cls.__module__+"."+cls.__name__

if __name__ == "__main__":
    unittest.main()