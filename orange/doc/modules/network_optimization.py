import orangeom
from orngNetwork import NetworkOptimization
from pylab import *

# create graph object of type GraphAsList
graph = orangeom.Network(5, 0)

# set edges
for i in range(4):
    for j in range(i + 1, 5):
        graph[i,j] = 1

# vertices are placed randomly in NetworkOptimization constructor
network = NetworkOptimization(graph)

# optimize verices layout with one of included algorithms
network.fruchtermanReingold(100, 1000)

# read all edges and plot a line
for u, v in graph.getEdges():
    x1, y1 = network.coors[0][u], network.coors[1][u]
    x2, y2 = network.coors[0][v], network.coors[1][v]
    plot([x1, x2], [y1, y2], 'b-')

# read x and y coordinates to Python list
x = network.coors[0]
y = network.coors[1]

# plot vertices
plot(x, y, 'ro')
show()
