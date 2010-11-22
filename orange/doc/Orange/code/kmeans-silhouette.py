import orange
import Orange.cluster

data = orange.ExampleTable("voting")
# data = orange.ExampleTable("iris")
for k in range(2,5):
    km = Orange.cluster.KMeans(data, k, initialization=Orange.cluster.kmeans_init_diversity)
    score = Orange.cluster.score_silhouette(km)
    print k, score

km = Orange.cluster.KMeans(data, 3, initialization=Orange.cluster.kmeans_init_diversity)
Orange.cluster.plot_silhouette(km, "kmeans-silhouette.png")
