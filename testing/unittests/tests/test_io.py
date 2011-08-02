import unittest
import Orange
from Orange.misc import testing
import os, sys

datasets = ["iris", "housing", ]

def names_iter():
    for n in datasets:
        yield n, (n,)
        
@testing.data_driven(data_iter=names_iter())
class TestIO(unittest.TestCase):
#    def setUp(self):
#        Orange.data.io.set_search_paths()
    @testing.test_on_data
    def test_io_on(self, name):
        table = Orange.data.Table(name)
        for ext in ["tab", "svm", "arff"]: # TODO: add R, and C50
            filename = name + "." + ext
            try:
                table.save(filename)
                table_clone = Orange.data.Table(filename)
            finally:
                os.remove(filename)
                
if __name__ == "__main__":
    unittest.main()