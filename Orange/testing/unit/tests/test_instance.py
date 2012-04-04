try:
    import unittest2 as unittest
except ImportError:
    import unittest

import Orange

class TestInstance(unittest.TestCase):
    def test_empty_domain(self):
        domain = Orange.data.Domain([])
        instance = Orange.data.Instance(domain)
        instance = Orange.data.Instance(domain, [])

    def test_constructors(self):
        features = [Orange.feature.Continuous("a"),
                    Orange.feature.Discrete("b", values=["1", "2"])
                    ]
        domain = Orange.data.Domain(features)
        instance = Orange.data.Instance(domain)
        self.assertTrue(all(v.is_special() for v in instance))

        copy_instance = Orange.data.Instance(instance)
        self.assertTrue(all(v1 == v2 for v1, v2 in zip(instance, copy_instance)))

        instance = Orange.data.Instance(domain, ["?", "?"])
        self.assertTrue(all(v.is_special() for v in instance))

        copy_instance = Orange.data.Instance(instance)
        self.assertTrue(all(v1 == v2 for v1, v2 in zip(instance, copy_instance)))

        instance = Orange.data.Instance(domain, ["1.0", "1"])
        self.assertTrue(all(not v.is_special() for v in instance))

        copy_instance = Orange.data.Instance(instance)
        self.assertTrue(all(v1 == v2 for v1, v2 in zip(instance, copy_instance)))

        self.assertRaises(TypeError, Orange.data.Instance, (domain, ["?"]))
        
        class_vars = [Orange.feature.Discrete("C1", values=["1", "2"]),
                      Orange.feature.Discrete("C2", values=["1", "2"])
                      ]

        domain = Orange.data.Domain(features, class_vars=class_vars)

        instance = Orange.data.Instance(domain)

        instance = Orange.data.Instance(domain, ["?", "?", "?", "?"])

        self.assertRaises(TypeError, Orange.data.Instance, (domain, ["?", "?", "?"]))
        self.assertRaises(TypeError, Orange.data.Instance, (domain, ["?", "?"]))


if __name__ == "__main__":
    unittest.main()