import orange
import orngClustering
data = orange.ExampleTable("../datasets/brown-selected.tab")
root = orngClustering.hierarchicalClustering(data, order=False)
d = orngClustering.DendrogramPlotPylab(root, data=data, labels=[str(ex.getclass()) for ex in data])
d.plot(filename="tmp.png")
