import orange
import Orange.cluster

def callback(km):
    print "Iteration: %d, changes: %d, score: %.4f" % (km.iteration, km.nchanges, km.score)
    
data = orange.ExampleTable("iris")
km = Orange.cluster.KMeans(data, 3, minscorechange=0, inner_callback=callback)
