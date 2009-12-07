import orange
import orngClustering
import random

data = orange.ExampleTable("iris")
# data = orange.ExampleTable("lung-cancer")

bestscore = 0
for k in range(2,10):
    random.seed(42)
    km = orngClustering.KMeans(data, k, 
            initialization=orngClustering.KMeans_init_hierarchicalClustering(n=50), 
            nstart=10)
    score = orngClustering.score_silhouette(km)
    print "%d: %.3f" % (k, score)
    if score > bestscore:
        best_km = km
        bestscore = score

orngClustering.plot_silhouette(best_km, filename='tmp.png')
