import orange
import orngClustering
import pylab
import random

def plot_scatter(data, km, attx, atty, filename="kmeans-scatter", title=None):
    """plot a data scatter plot with the position of centeroids"""
    pylab.rcParams.update({'font.size': 8, 'figure.figsize': [4,3]})
    x = [float(d[attx]) for d in data]
    y = [float(d[atty]) for d in data]
    colors = ["c", "w", "b"]
    cs = "".join([colors[c] for c in km.clusters])
    pylab.scatter(x, y, c=cs, s=10)
    
    xc = [float(d[attx]) for d in km.centroids]
    yc = [float(d[atty]) for d in km.centroids]
    pylab.scatter(xc, yc, marker="x", c="k", s=200)
    
    pylab.xlabel(attx)
    pylab.ylabel(atty)
    if title:
        pylab.title(title)
    pylab.savefig("%s-%03d.png" % (filename, km.iteration))
    pylab.close()

def in_callback(km):
    print "Iteration: %d, changes: %d, score: %8.6f" % (km.iteration, km.nchanges, km.score)
    plot_scatter(data, km, "petal width", "petal length", title="Iteration %d" % km.iteration)
    
data = orange.ExampleTable("iris")
random.seed(42)
km = orngClustering.KMeans(data, 3, maxiters=10, inner_callback=in_callback)
