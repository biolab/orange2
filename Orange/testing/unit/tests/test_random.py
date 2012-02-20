try:
    import unittest2 as unittest
except ImportError:
    import unittest
    
class TestImportSanity(unittest.TestCase):
    def test_import_sanity(self):
        """Test that the act of importing orange does not change
        the global seed state.
         
        """
        
        # Needs to be tested in a clean python environment
        import subprocess
        import sys
        rval = subprocess.call([sys.executable, "-c", 
"import random; state = random.getstate(); import Orange; assert(state == random.getstate())"
])
        self.assertEqual(rval, 0, "'import Orange' changes the global random seed")

if __name__ == "__main__":
    unittest.main()