import Orange

import random
random.seed(42)

def callback(km):
    print "Iteration: %d, changes: %d, score: %.4f" % (km.iteration, km.nchanges, km.score)
    
iris = Orange.data.Table("iris")
km = Orange.clustering.kmeans.Clustering(iris, 3, minscorechange=0, inner_callback=callback)
