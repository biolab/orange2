import orngNetwork
from pylab import *

# create graph object of type GraphAsList
net = orngNetwork.Network(5, 0)

# set edges
for i in range(4):
    for j in range(i + 1, 5):
        net[i,j] = 1

# vertices are placed randomly in NetworkOptimization constructor
networkOptimization = orngNetwork.NetworkOptimization(net)

# optimize verices layout with one of included algorithms
networkOptimization.fruchtermanReingold(100, 1000)

# read all edges and plot a line
for u, v in net.getEdges():
    x1, y1 = net.coors[0][u], net.coors[1][u]
    x2, y2 = net.coors[0][v], net.coors[1][v]
    plot([x1, x2], [y1, y2], 'b-')

# read x and y coordinates to Python list
x = net.coors[0]
y = net.coors[1]

# plot vertices
plot(x, y, 'ro')
show()
