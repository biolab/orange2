try:
    import unittest2 as unittest
except:
    import unittest

import numpy as np
import pickle, random

from Orange import data, feature
from Orange.projection import linear

np.random.seed(0)
random.seed(0)

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

def create_base(ncomponents, m):
    vector = np.random.randint(0,5,m)
    while not vector.any():
        vector = np.random.randint(0,5,m)
    principal_components = np.zeros((ncomponents, m))
    k = float(m) / ncomponents
    to = 0
    for i in range(1,ncomponents):
        from_, to = int(k*(i-1)), int(k*i)
        principal_components[i-1,from_:to] = vector[from_:to]
    principal_components[ncomponents-1,to:m] = normalize(vector[to:m])
    return principal_components


class TestPca(unittest.TestCase):
    def create_dataset(self, ncomponents=3, m=10):
        self.principal_components = create_base(ncomponents, m)
        self.dataset = data.Table(*prepare_dataset(components=self.principal_components))

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


    def test_pca(self):
        for m in (10, 250):
            self.create_dataset(m=m)

            pca = linear.Pca(standardize=False)(self.dataset)

            self.assertInCorrectSpace(pca.projection[pca.variances > 0.01, :])
            for v in pca.projection:
                # projections vectors should be normalized
                self.assertAlmostEqual(np.linalg.norm(v), 1.)

            # Components should have decreasing variants
            self.assertListEqual(pca.variances.tolist(), sorted(pca.variances, reverse=True))

    def test_pca_with_standardization(self):
        self.create_dataset(ncomponents=1)

        pca = linear.Pca(standardize=True)(self.dataset)
        projection = pca.projection[0]
        non_zero_elements = projection[projection.nonzero()]

        # since values in all dimensions are normally distributed, dimensions should be treated as equally important
        self.assertAlmostEqual(non_zero_elements.min(), non_zero_elements.max())

    def test_pca_with_variance_covered(self):
        ncomponents = 3
        self.create_dataset(ncomponents=ncomponents)

        pca = linear.Pca(variance_covered=.99)(self.dataset)

        nvectors, vector_dimension = pca.projection.shape
        self.assertEqual(nvectors, ncomponents)

    def test_pca_with_max_components(self):
        max_components = 3
        self.create_dataset(ncomponents = max_components + 3)

        pca = linear.Pca(max_components=max_components)(self.dataset)

        nvectors, vector_dimension = pca.projection.shape
        self.assertEqual(nvectors, max_components)

    def test_pca_handles_unknowns(self):
        self.create_dataset_with_unknowns()

        linear.Pca()(self.dataset)

    def test_total_variance_remains_the_same(self):
        for m in (10, 250):
            self.create_dataset(m=m)

            pca = linear.Pca()(self.dataset)

            self.assertAlmostEqual(pca.variance_sum, pca.variances.sum())
            self.assertAlmostEqual(pca.variance_sum, (self.principal_components != 0).sum())

    def test_pca_on_empty_data(self):
        self.create_empty_dataset()

        with self.assertRaises(ValueError):
            linear.Pca()(self.dataset)

    def test_pca_on_only_constant_features(self):
        self.create_constant_dataset()

        with self.assertRaises(ValueError):
            linear.Pca()(self.dataset)

    def assertInCorrectSpace(self, vectors):
        vectors = vectors.copy()
        for component in self.principal_components:
            i = component.nonzero()[0][0]
            coef = vectors[:,i] / component[i]
            vectors -= np.dot(coef.reshape(-1, 1), component.reshape(1, -1))

        for vector in vectors:
            for value in vector:
                self.assertAlmostEqual(value, 0.)


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

    def test_can_pickle_and_unpickle(self):
        self.create_normal_dataset()
        projector = linear.PCA(variance_covered=.99)(self.dataset)

        pickled = pickle.dumps(projector)
        restored = pickle.loads(pickled)

        self.assertFalse((projector.projection - restored.projection).any())
        self.assertFalse((projector.center - restored.center).any())
        self.assertFalse((projector.scale - restored.scale).any())

        transformed, new_transformed = projector(self.dataset), restored(self.dataset)
        print transformed[0][0]
        for ex1, ex2 in zip(transformed, new_transformed):
            for v1, v2 in zip(ex1, ex2):
                self.assertEqual(v1, v2)


class TestFda(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()

