import orange
import orngClustering

import random
random.seed(42)

def callback(km):
    print "Iteration: %d, changes: %d, score: %.4f" % (km.iteration, km.nchanges, km.score)
    
data = orange.ExampleTable("iris")
km = orngClustering.KMeans(data, 3, minscorechange=0, inner_callback=callback)
