import orange
import orngClustering

data = orange.ExampleTable("voting")
# data = orange.ExampleTable("iris")
for k in range(2,5):
    km = orngClustering.KMeans(data, k, initialization=orngClustering.kmeans_init_diversity)
    score = orngClustering.score_silhouette(km)
    print k, score

km = orngClustering.KMeans(data, 3, initialization=orngClustering.kmeans_init_diversity)
orngClustering.plot_silhouette(km, "kmeans-silhouette.png")
