import orange
import orngClustering
import pylab

def plot_scatter(data, cls, attx, atty, filename="hclust-scatter", title=None):
    """plot a data scatter plot with the position of centeroids"""
    pylab.rcParams.update({'font.size': 8, 'figure.figsize': [4,3]})
    x = [float(d[attx]) for d in data]
    y = [float(d[atty]) for d in data]
    colors = ["c", "w", "b"]
    cs = "".join([colors[c] for c in cls])
    pylab.scatter(x, y, c=cs, s=10)
    
    pylab.xlabel(attx)
    pylab.ylabel(atty)
    if title:
        pylab.title(title)
    pylab.savefig("%s.png" % filename)
    pylab.close()

data = orange.ExampleTable("iris")
root = orngClustering.hierarchicalClustering(data)
n = 3
cls = orngClustering.hierarhicalClustering_topClustersMembership(root, n)
plot_scatter(data, cls, "sepal width", "sepal length", title="Hiearchical clustering (%d clusters)" % n)
