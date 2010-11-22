import orange
import Orange.cluster
import random

data_names = ["iris", "housing", "vehicle"]
data_sets = [orange.ExampleTable(name) for name in data_names]

print "%10s %3s %3s %3s" % ("", "Rnd", "Div", "HC")
for data, name in zip(data_sets, data_names):
    random.seed(42)
    km_random = Orange.cluster.KMeans(data, centroids = 3)
    km_diversity = Orange.cluster.KMeans(data, centroids = 3, \
        initialization=Orange.cluster.kmeans_init_diversity)
    km_hc = Orange.cluster.KMeans(data, centroids = 3, \
        initialization=Orange.cluster.KMeans_init_hierarchicalClustering(n=100))
    print "%10s %3d %3d %3d" % (name, km_random.iteration, km_diversity.iteration, km_hc.iteration)
