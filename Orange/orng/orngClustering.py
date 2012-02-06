from Orange.clustering.kmeans import * 
# from Orange.cluster.hierarchical import * 

from Orange.core import \
     DistanceMap, \
     DistanceMapConstructor, \
     ExamplesDistance, \
     ExamplesDistance_Normalized, \
     ExamplesDistanceConstructor

from Orange.distance import HammingDistance as ExamplesDistance_Hamming, \
    DTWDistance as ExamplesDistance_DTW, \
    EuclideanDistance as ExamplesDistance_Euclidean, \
    ManhattanDistance as ExamplesDistance_Manhattan, \
    MaximalDistance as ExamplesDistance_Maximal, \
    ReliefDistance as ExamplesDistance_Relief, \
    DTW as ExamplesDistanceConstructor_DTW, \
    Euclidean as ExamplesDistanceConstructor_Euclidean, \
    Hamming as ExamplesDistanceConstructor_Hamming, \
    Manhattan as ExamplesDistanceConstructor_Manhattan, \
    Maximal as ExamplesDistanceConstructor_Maximal, \
    Relief as ExamplesDistanceConstructor_Relief, \
    PearsonR as ExamplesDistanceConstructor_PearsonR, \
    PearsonRDistance as ExamplesDistance_PearsonR, \
    SpearmanR as ExamplesDistanceConstructor_SpearmanR, \
    SpearmanRDistance as ExamplesDistance_SpearmanR

from Orange.clustering.kmeans import Clustering as KMeans
from Orange.clustering.kmeans import init_random as kmeans_init_random
from Orange.clustering.kmeans import init_diversity as kmeans_init_diversity
from Orange.clustering.kmeans import init_hclustering as KMeans_init_hierarchicalClustering
from Orange.clustering.kmeans import data_center as data_center
from Orange.clustering.kmeans import plot_silhouette as plot_silhouette
from Orange.clustering.kmeans import score_distance_to_centroids as score_distance_to_centroids
from Orange.clustering.kmeans import score_silhouette as score_silhouette
from Orange.clustering.kmeans import score_fast_silhouette as score_fastsilhouette

from Orange.clustering.hierarchical import clustering as hierarchicalClustering
from Orange.clustering.hierarchical import clustering_features as hierarchicalClustering_attributes
from Orange.clustering.hierarchical import cluster_to_list as hierarchicalClustering_clusterList
from Orange.clustering.hierarchical import top_clusters as hierarchicalClustering_topClusters
from Orange.clustering.hierarchical import top_cluster_membership as hierarhicalClustering_topClustersMembership
from Orange.clustering.hierarchical import order_leaves as orderLeaves

from Orange.clustering.hierarchical import dendrogram_draw, DendrogramPlotPylab, DendrogramPlot
#left for backward compatibility
hierarchicalClustering_attributes
