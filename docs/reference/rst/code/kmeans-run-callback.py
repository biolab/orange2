import Orange

def callback(km):
    print "Iteration: %d, changes: %d, score: %.4f" % (km.iteration, km.nchanges, km.score)
    
table = Orange.data.Table("iris")
km = Orange.clustering.kmeans.Clustering(table, 3, minscorechange=0, inner_callback=callback)
