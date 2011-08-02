""" Orange.data.Table related unit-tests
"""
import unittest
from Orange.misc import testing
import Orange
import cPickle

def native(table):
    table = table.native()
    for i in range(len(table)):
        table[i] = [v.native() for v in table[i].native()]
    return table
        
def names_iter():
    for name in testing.ALL_DATASETS:
        yield name.replace(" ", "_").replace("-", "_"), (name,)
        
@testing.data_driven(data_iter=names_iter())
class TestLoading(unittest.TestCase):
    
    @testing.test_on_data
    def test_load_on(self, name):
        """ Test the loading of the data set
        """
        table = Orange.data.Table(name)
        self.assertIsNotNone(getattr(table, "attributeLoadStatus"), "No attributeLoadStatus")
        
    @testing.test_on_data
    def test_pickling_on(self, name):
        """ Test data table pickling.
        """
        table = Orange.data.Table(name)
        s = cPickle.dumps(table)
        table_clone = cPickle.loads(s)
        self.assertEqual(list(table.domain), list(table_clone.domain))
        self.assertEqual(table.domain.class_var, table_clone.domain.class_var)
        self.assertEqual(native(table), native(table_clone), "Native representation does is not equal!")
        
if __name__ == "__main__":
    unittest.main()
    