""" Orange.data.Table related unit-tests
"""
try:
    import unittest2 as unittest
except:
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
#        self.assertEqual(table.domain, table_clone.domain)
#        self.assertEqual(table.domain.class_var, table_clone.domain.class_var)
        self.assertEqual(native(table), native(table_clone), "Native representation is not equal!")


import tempfile

@testing.datasets_driven
class TestSaving(unittest.TestCase):
    @testing.test_on_data
    def test_R_on(self, name):
        data = Orange.data.Table(name)
        with tempfile.NamedTemporaryFile(suffix=".R") as f:
            data.save(f.name)

#    @testing.test_on_data
#    def test_toC50(self, name):
#        data = Orange.data.Table(name)

    @testing.test_on_datasets(datasets=testing.CLASSIFICATION_DATASETS + \
                              testing.REGRESSION_DATASETS)
    def test_arff_on(self, data):
        with tempfile.NamedTemporaryFile(suffix=".arff") as f:
            data.save(f.name)
            f.flush()
            data_arff = Orange.data.Table(f.name)
    @testing.test_on_datasets(datasets=testing.CLASSIFICATION_DATASETS + \
                              testing.REGRESSION_DATASETS)
    def test_svm_on(self, data):
        with tempfile.NamedTemporaryFile(suffix=".svm") as f:
            data.save(f.name)
            f.flush()
            data_svm = Orange.data.Table(f.name)

    @testing.test_on_datasets
    def test_csv_on(self, data):
        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            Orange.data.io.save_csv(f, data, dialect="excel-tab")
            f.flush()
            f.seek(0)
            Orange.data.io.load_csv(f)




if __name__ == "__main__":
    unittest.main()
