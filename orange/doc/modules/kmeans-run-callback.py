import orange
import orngClustering

def callback(km):
    print "Iteration: %d, changes: %d, score: %.4f" % (km.iteration, km.nchanges, km.score)
    
data = orange.ExampleTable("iris")
km = orngClustering.KMeans(data, 3, inner_callback = callback)
