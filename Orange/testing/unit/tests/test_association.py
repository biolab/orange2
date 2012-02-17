from __future__ import division
from Orange import associate
from Orange.misc import testing
from Orange.misc.testing import test_on_data, datasets_driven
try:
    import unittest2 as unittest
except:
    import unittest
import pickle

# helper functions

def test_inducer_on(self, table):
    rules = self.inducer(table)
    if self.inducer.store_examples and rules:
        self.assertIsNotNone(rules[0].examples)

    self.assertLessEqual(len(rules), self.inducer.max_item_sets)
    for r in rules:
        self.assertGreaterEqual(r.support, self.inducer.support)
        self.assertIsNotNone(r.left)
        self.assertIsNotNone(r.right)
        self.assertAlmostEqual(r.support, r.n_applies_both / r.n_examples, places=3)
        self.assertAlmostEqual(r.confidence, r.n_applies_both / r.n_applies_left, places=3)
        self.assertAlmostEqual(r.coverage, r.n_applies_left / r.n_examples, places=3)
        self.assertAlmostEqual(r.strength, r.n_applies_right / r.n_applies_left, places=3)
        self.assertAlmostEqual(r.lift, r.n_examples * r.n_applies_both / (r.n_applies_left * r.n_applies_right), places=3)
#        self.assertAlmostEqual(r.leverage, (r.n_examples * r.n_applies_both - r.n_applies_left * r.n_applies_right) / 100.0)

    itemsets = self.inducer.get_itemsets(table)
    self.rules = rules


def test_pickling_on(self, table):
    rules = self.inducer(table)
    rules_clone = pickle.loads(pickle.dumps(rules))
    inducer_clone = pickle.loads(pickle.dumps(self.inducer))
    rules_clone1 = inducer_clone(table)
    for r1, r2, r3 in zip(rules, rules_clone, rules_clone1):
        self.assertEqual(r1.support, r2.support)
        self.assertEqual(r2.support, r3.support)

        self.assertEqual(r1.confidence, r2.confidence)
        self.assertEqual(r2.confidence, r3.confidence)

        self.assertEqual(r1.coverage, r2.coverage)
        self.assertEqual(r2.coverage, r3.coverage)

        self.assertEqual(r1.strength, r2.strength)
        self.assertEqual(r2.strength, r3.strength)

        for inst in table:
            self.assertEqual(r1.applies_left(inst), r2.applies_left(inst))
            self.assertEqual(r2.applies_left(inst), r3.applies_left(inst))

            self.assertEqual(r1.applies_right(inst), r2.applies_right(inst))
            self.assertEqual(r2.applies_right(inst), r3.applies_right(inst))

            self.assertEqual(r1.applies_both(inst), r2.applies_both(inst))
            self.assertEqual(r2.applies_both(inst), r3.applies_both(inst))


@datasets_driven(datasets=["inquisition.basket"])
class TestSparseInducer(unittest.TestCase):
    def setUp(self):
        self.inducer = associate.AssociationRulesSparseInducer(support=0.5,
                                store_examples=True, max_item_sets=2000)

    @test_on_data
    def test_inducer_on(self, table):
        test_inducer_on(self, table)


    @test_on_data
    def test_pickling_on(self, table):
        test_pickling_on(self, table)


@datasets_driven(datasets=["lenses", "monks-1"])
class TestInducer(unittest.TestCase):
    def setUp(self):
        self.inducer = associate.AssociationRulesInducer(support=0.2,
                            confidence=0.5, store_examples=True,
                            max_item_sets=2000)

    @test_on_data
    def test_inducer_on(self, table):
        test_inducer_on(self, table)

    @test_on_data
    def test_pickling_on(self, table):
        test_pickling_on(self, table)


@datasets_driven(datasets=["lenses", "monks-1"])
class TestInducerClassification(unittest.TestCase):
    def setUp(self):
        self.inducer = associate.AssociationRulesInducer(support=0.2,
                            confidence=0.5, store_examples=True,
                            max_item_sets=2000,
                            classification_rules=True)

    @test_on_data
    def test_inducer_on(self, table):
        test_inducer_on(self, table)

    @test_on_data
    def test_pickling_on(self, table):
        test_pickling_on(self, table)


if __name__ is "__main__":
    unittest.main()


