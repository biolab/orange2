try:
    import unittest2 as unittest
except ImportError:
    import unittest

import Orange


class TestValue(unittest.TestCase):
    def test_discrete(self):
        var = Orange.feature.Discrete("A", values=["A", "B", "C"])
        v = Orange.data.Value(var, 1)

        self.assertEqual(v, var(1))
        self.assertEqual(int(v), 1)
        self.assertEqual(float(v), 1.0)
        self.assertEqual(str(v), "B")

        self.assertEqual(v, 1)

        self.assertTrue(bool(v))

        # From the docs: "A value is considered true if it is not undefined."
        self.assertTrue(bool(Orange.data.Value(var, 0)))
        self.assertFalse(bool(Orange.data.Value(var, "?")))

        v = Orange.data.Value(var, "?")
        self.assertTrue(v.is_special())

        with self.assertRaises(TypeError):
            int(v)

        with self.assertRaises(TypeError):
            float(v)

        self.assertEquals(str(v), "?")

    def test_ordinal_cmp(self):
        deg1 = Orange.feature.Discrete(
            "deg1", values=["little", "medium", "big"]
        )
        deg2 = Orange.feature.Discrete(
            "deg2", values=["tiny", "little", "big", "huge"]
        )
        val1 = Orange.data.Value(deg1, "medium")
        val2 = Orange.data.Value(deg2, "little")

        self.assertGreater(val1, val2)
        self.assertLess(val2, val1)

        val3 = Orange.data.Value(deg1, "medium")
        val4 = Orange.data.Value(deg2, "huge")

        with self.assertRaises(TypeError):
            _ = val3 > val4

        with self.assertRaises(TypeError):
            _ = val3 < val4

        val5 = Orange.data.Value(deg1, "big")
        val6 = Orange.data.Value(deg2, "big")

        self.assertEqual(val5, val6)
        self.assertEqual(val6, val5)

        # Unknowns always compare greater
        val7 = Orange.data.Value(deg1, "?")
        self.assertGreater(val7, val5)
        self.assertGreater(val7, val6)
        self.assertLess(val5, val7)
        self.assertLess(val6, val7)

        self.assertEqual(val7, deg1("?"))
        self.assertEqual(val7, deg2("?"))

    def test_continuous(self):
        var = Orange.feature.Continuous("X")
        v = Orange.data.Value(var, 0.0)

        self.assertEqual(v, var(0.0))
        self.assertEqual(int(v), 0)
        self.assertEqual(float(v), 0.0)

        self.assertEqual(v, 0.0)

        # From the docs: "A value is considered true if it is not undefined."
        self.assertTrue(bool(Orange.data.Value(var, 0)))

        v = Orange.data.Value(var, "?")
        self.assertTrue(v.is_special())

        with self.assertRaises(TypeError):
            int(v)

        with self.assertRaises(TypeError):
            float(v)

        self.assertEquals(str(v), "?")
