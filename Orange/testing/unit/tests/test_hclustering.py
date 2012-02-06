from Orange.clustering.hierarchical import clustering, \
    clustering_features,  cluster_to_list, \
    top_clusters, HierarchicalClustering, order_leaves_py, \
    order_leaves_cpp, instance_distance_matrix
                      
from Orange.clustering import hierarchical as hier     
from Orange.distance.instances import *
                           
import Orange.misc.testing as testing
import Orange
import unittest

import cPickle as pickle

@testing.datasets_driven
class TestHClustering(testing.DataTestCase):    
    @testing.test_on_data
    def test_example_clustering_on(self, data):
        constructors = [EuclideanConstructor, ManhattanConstructor]
        for distance_constructor in constructors:
            clust = clustering(data, distance_constructor, HierarchicalClustering.Single)
            clust = clustering(data, distance_constructor, HierarchicalClustering.Average)
            clust = clustering(data, distance_constructor, HierarchicalClustering.Complete)
            clust = clustering(data, distance_constructor, HierarchicalClustering.Ward)
            top_clust = top_clusters(clust, 5)
            cluster_list = cluster_to_list(clust, 5)
            
    @testing.test_on_data
    def test_attribute_clustering_on(self, data):
        clust = clustering_features(data)
        cluster_list = cluster_to_list(clust, 5)
            
    @testing.test_on_datasets(datasets=["iris"])
    def test_pickling_on(self, data):
        cluster = clustering(data, EuclideanConstructor, HierarchicalClustering.Single)
        s = pickle.dumps(cluster)
        cluster_clone = pickle.loads(s)
        self.assertEqual(len(cluster), len(cluster_clone))
        self.assertEqual(cluster.mapping, cluster_clone.mapping)
        
    @testing.test_on_data
    def test_ordering_on(self, data):
        def p(val, obj=None):
            self.assert_(val >= 0 and val <=100)
            self.assertIsInstance(val, float)
        matrix = instance_distance_matrix(data, EuclideanConstructor(), progress_callback=p)
        root1 = HierarchicalClustering(matrix, progress_callback=p)
        root2 = hier.clone(root1)
        
        order_leaves_py(root1, matrix, progressCallback=p)
        order_leaves_cpp(root2, matrix, progress_callback=p)
        
        def score(mapping):
            sum = 0.0
            for i in range(matrix.dim - 1):
               sum += matrix[mapping[i], mapping[i+1]]
            return sum
        
        # Slight differences are possible due to the float/double precision.
        self.assertAlmostEqual(score(root1.mapping), score(root2.mapping),
                               places=3)
        

class TestHClusteringUtility(unittest.TestCase):
    def setUp(self):
        m = [[],
             [ 3],
             [ 2,  4],
             [17,  5,  4],
             [ 2,  8,  3,  8],
             [ 7,  5, 10, 11, 2],
             [ 8,  4,  1,  5, 11, 13],
             [ 4,  7, 12,  8, 10,  1,  5],
             [13,  9, 14, 15,  7,  8,  4,  6],
             [12, 10, 11, 15,  2,  5,  7,  3,  1]]
        self.matrix = Orange.core.SymMatrix(m)
        self.matrix.setattr("objects", ["Ann", "Bob", "Curt", "Danny", "Eve", "Fred", "Greg", "Hue", "Ivy", "Jon"])
        self.cluster = hier.HierarchicalClustering(self.matrix)
        
    def test_objects_mapping(self):
        objects = self.cluster.mapping.objects
        self.assertEqual(list(self.cluster),
                         [objects[i] for i in self.cluster.mapping])
        
    def test_clone(self):
        cloned_cluster = hier.clone(self.cluster)
        self.assertTrue(self.cluster.mapping.objects is cloned_cluster.mapping.objects)
        self.assertEqual(self.cluster.mapping, cloned_cluster.mapping)
        self.assertEqual(list(self.cluster), list(cloned_cluster))
        
    def test_order(self):
        post = hier.postorder(self.cluster)
        pre = hier.preorder(self.cluster)
        
    def test_prunning(self):
        pruned1 = hier.pruned(self.cluster, level=2)
        depths = hier.cluster_depths(pruned1)
        self.assertTrue(all(d <= 2 for d in depths.values()))
        
        pruned2 = hier.pruned(self.cluster, height=10)
        self.assertTrue(c.height >= 10 for c in hier.preorder(pruned2))
        
        pruned3 = hier.pruned(self.cluster, condition=lambda cl: len(cl) <= 3)
        self.assertTrue(len(c) > 3 for c in hier.preorder(pruned3))
        
    def test_dendrogram_draw(self):
        from StringIO import StringIO
        file = StringIO()
        hier.dendrogram_draw(file, self.cluster, format="svg")
        self.assertTrue(len(file.getvalue()))
        file = StringIO()
        hier.dendrogram_draw(file, self.cluster, format="eps")
        self.assertTrue(len(file.getvalue()))
        file = StringIO()
        hier.dendrogram_draw(file, self.cluster, format="png")
        self.assertTrue(len(file.getvalue()))
        
    def test_dendrogram_layout(self):
        hier.dendrogram_layout(self.cluster)
        pruned1 = hier.pruned(self.cluster, level=2)
        hier.dendrogram_layout(pruned1, expand_leaves=True)
        hier.dendrogram_layout(pruned1, expand_leaves=False)
        pruned2 = hier.pruned(self.cluster, height=10)
        hier.dendrogram_layout(pruned2, expand_leaves=True)
        hier.dendrogram_layout(pruned2, expand_leaves=False)
        
    def test_cophenetic(self):
        cmatrix = hier.cophenetic_distances(self.cluster)
        self.assertEqual(cmatrix.dim, self.matrix.dim)
        corr = hier.cophenetic_correlation(self.cluster, self.matrix)
        

if __name__ == "__main__":
    unittest.main()
        
            