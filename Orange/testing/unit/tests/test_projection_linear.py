try:
    import unittest2 as unittest
except:
    import unittest

import numpy as np
import random

from Orange import data, feature
from Orange.projection import linear

def normalize(a):
    a = a if isinstance(a, np.ndarray) else np.array(a)
    return a / np.linalg.norm(a)

datasets = None

def prepare_dataset(components=((),), n=150):
    components = components if isinstance(components, np.ndarray) else np.array(components)

    ncomponents, m = components.shape
    coefficients = np.random.normal(0., 1., (n, ncomponents))

    d = np.dot(coefficients, components)

    domain = data.Domain([feature.Continuous("A%d" % i) for i in range(m)], False)
    return domain, d



class TestPca(unittest.TestCase):
    def create_normal_dataset(self):
        self.principal_component = normalize([random.randint(0, 5) for _ in range(10)])
        self.dataset = data.Table(*prepare_dataset(components=[self.principal_component]))

    def create_wide_dataset(self):
        self.principal_component = normalize([random.randint(0, 5) for _ in range(250)])
        self.dataset = data.Table(*prepare_dataset(components=[self.principal_component]))

    def create_empty_dataset(self):
        self.dataset = data.Table(*prepare_dataset(components=([0, 0, 0, 0, 0],), n=0))

    def create_constant_dataset(self):
        self.dataset = data.Table(*prepare_dataset(components=([0, 0, 0, 0, 0],)))

    def create_dataset_with_unknowns(self, percentage=0.05):
        self.principal_component = normalize([random.randint(0, 5) for _ in range(10)])
        self.dataset = data.Table(*prepare_dataset(components=[self.principal_component]))

        for ex in self.dataset:
            for i, _ in enumerate(ex):
                if random.random() < percentage:
                    ex[i] = "?"


    def test_pca_on_normal_data(self):
        self.create_normal_dataset()

        pca = linear.Pca(standardize=False)(self.dataset)
        self.assertIsInstance(pca, linear.PcaProjector)

        absolute_error = (np.abs(pca.projection[0]) - np.abs(self.principal_component)).sum()
        self.assertAlmostEqual(absolute_error, 0.)

    def test_pca_on_wide_data(self):
        self.create_wide_dataset()

        pca = linear.Pca(standardize=False)(self.dataset)
        self.assertIsInstance(pca, linear.PcaProjector)

        absolute_error = (np.abs(pca.projection[0]) - np.abs(self.principal_component)).sum()
        self.assertAlmostEqual(absolute_error, 0., 1)

    def test_pca_with_standardization(self):
        self.create_normal_dataset()

        pca = linear.Pca(standardize=True)(self.dataset)
        eigen_vector = pca.projection[0]
        non_zero_elements = eigen_vector[eigen_vector.nonzero()]

        # since values in all dimensions are normally distributed, dimensions should be treated as equally important
        self.assertAlmostEqual(non_zero_elements.min(), non_zero_elements.max())

    def test_pca_with_variance_covered(self):
        self.create_normal_dataset()

        pca = linear.Pca(variance_covered=.99)(self.dataset)
        # all data points lie in one dimension, one component should cover all the variance
        nvectors, vector_dimension = pca.projection.shape
        self.assertEqual(nvectors, 1)

    def test_pca_with_max_components(self):
        self.create_normal_dataset()
        max_components = 3

        pca = linear.Pca(max_components=max_components)(self.dataset)
        # all data points lie in one dimension, one component should cover all the variance
        nvectors, vector_dimension = pca.projection.shape
        self.assertEqual(nvectors, max_components)

    def test_pca_handles_unknowns(self):
        self.create_dataset_with_unknowns()

        pca = linear.Pca()(self.dataset)



    def test_pca_on_empty_data(self):
        self.create_empty_dataset()

        with self.assertRaises(ValueError):
            linear.Pca()(self.dataset)

    def test_pca_on_only_constant_features(self):
        self.create_constant_dataset()

        with self.assertRaises(ValueError):
            linear.Pca()(self.dataset)


class TestProjector(unittest.TestCase):
    def create_normal_dataset(self):
        self.principal_component = normalize([random.randint(0, 5) for _ in range(10)])
        self.dataset = data.Table(*prepare_dataset(components=[self.principal_component]))

    def create_dataset_with_classes(self):
        domain, features = prepare_dataset(components=[[random.randint(0, 5) for _ in range(10)]])
        domain = data.Domain(domain.features,
                             feature.Discrete("C", values=["F", "T"]),
                             class_vars=[feature.Discrete("MC%i" % i, values=["F", "T"]) for i in range(4)])

        self.dataset = data.Table(domain, np.hstack((features, np.random.random((len(features), 5)))))


    def test_projected_domain_can_convert_data_with_class(self):
        self.create_dataset_with_classes()
        projector = linear.Pca(variance_covered=.99)(self.dataset)

        projected_data = projector(self.dataset)
        converted_data = data.Table(projected_data.domain, self.dataset)

        self.assertItemsEqual(projected_data, converted_data)

    def test_projected_domain_can_convert_data_without_class(self):
        self.create_normal_dataset()
        projector = linear.Pca(variance_covered=.99)(self.dataset)

        projected_data = projector(self.dataset)
        converted_data = data.Table(projected_data.domain, self.dataset)

        self.assertItemsEqual(projected_data, converted_data)

    def test_projected_domain_contains_class_vars(self):
        self.create_dataset_with_classes()

        projector = linear.Pca(variance_covered=.99)(self.dataset)
        projected_data = projector(self.dataset)

        self.assertIn(self.dataset.domain.class_var, projected_data.domain)
        for class_ in self.dataset.domain.class_vars:
            self.assertIn(class_, projected_data.domain)
        for ex1, ex2 in zip(self.dataset, projected_data):
            self.assertEqual(ex1.get_class(), ex2.get_class())
            for v1, v2 in zip(ex1.get_classes(), ex2.get_classes()):
                self.assertEqual(v2, v2)


    def test_projects_example(self):
        self.create_normal_dataset()
        projector = linear.Pca(variance_covered=.99)(self.dataset)

        projector(self.dataset[0])

    def test_projects_data_table(self):
        self.create_normal_dataset()
        projector = linear.Pca(variance_covered=.99)(self.dataset)

        projector(self.dataset)

    def test_converts_input_domain_if_needed(self):
        self.create_normal_dataset()
        projector = linear.Pca(variance_covered=.99)(self.dataset)

        new_examples = data.Table(data.Domain(self.dataset.domain.features[:5]), [[1.,2.,3.,4.,5.]])

        projector(new_examples)


class TestFda(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()

