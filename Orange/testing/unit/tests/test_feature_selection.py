import Orange.testing.testing as testing
from Orange.feature import selection, scoring
import Orange

from operator import itemgetter

try:
    import unittest2 as unittest
except ImportError:
    import unittest

class TestSelection(unittest.TestCase):
    def setUp(self):
        self.score = Orange.feature.scoring.Gini()
        self.data = Orange.data.Table("lenses")
        
        self.scores = scoring.score_all(self.data, self.score)
        
    def test_best_n(self):
        sorted_scores = sorted(self.scores, key=itemgetter(1),
                               reverse=True)
        # Test the descending order of scores
        self.assertEqual(self.scores, sorted_scores)
        
        # best 3 scores
        best_3 = map(itemgetter(0), sorted_scores[:3])
        
        # test best_n function
        self.assertEqual(selection.top_rated(self.scores, 3), best_3)
        
        self.assertTrue(len(selection.top_rated(self.scores, 3)) == 3)
        
        # all returned values should be strings.
        self.assertTrue(all(isinstance(item, basestring) for item in \
                            selection.top_rated(self.scores, 3)))
        
        new_data = selection.select(self.data, self.scores, 3)
        self.assertEqual(best_3, [a.name for a in new_data.domain.attributes])
        self.assertEqual(new_data.domain.class_var, self.data.domain.class_var)
        
    def test_above_threashold(self):
        threshold = self.scores[len(self.scores) / 2][1]
        above = [item[0] for item in self.scores if item[1] >= threshold]
        
        self.assertEqual(above, 
                         selection.above_threshold(self.scores, threshold)
                         )
        
        new_data = selection.select_above_threshold(self.data, 
                                                    self.scores, threshold)
        self.assertEqual(above, [a.name for a in new_data.domain.attributes])
        self.assertEqual(new_data.domain.class_var, self.data.domain.class_var)

    def test_select_features_subset(self):
        data = Orange.data.Table("lenses")

        d1 = selection._select_features_subset(data, [])
        self.assertSequenceEqual(d1.domain.features, [])
        self.assertIs(d1.domain.class_var, data.domain.class_var)

        d1 = selection._select_features_subset(data, [data.domain[0]])
        self.assertSequenceEqual(d1.domain.features, [data.domain[0]])
        self.assertIs(d1.domain.class_var, data.domain.class_var)

        domain = Orange.data.Domain(data.domain.features[:2],
                                    data.domain.class_var,
                                    class_vars=[data.domain.features[2]])
        domain.add_metas({-1: data.domain.features[3]})
        data = Orange.data.Table(domain, data)

        d1 = selection._select_features_subset(data, [data.domain[0]])
        self.assertSequenceEqual(d1.domain.features, [data.domain[0]])
        self.assertIs(d1.domain.class_var, data.domain.class_var)
        self.assertSequenceEqual(d1.domain.class_vars, data.domain.class_vars)
        self.assertEqual(d1.domain.get_metas(), data.domain.get_metas())

if __name__ == "__main__":
    unittest.main()
