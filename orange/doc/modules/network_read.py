import orange
from orngNetwork import NetworkOptimization
from pylab import *

        
# vertices are placed randomly in NetworkOptimization constructor
network = NetworkOptimization()

# read network from file
network.readNetwork("K5.net")

# read all edges and plot a line
for u, v in network.graph.getEdges():
    x1, y1 = network.coors[0][u], network.coors[1][u]
    x2, y2 = network.coors[0][v], network.coors[1][v]
    plot([x1, x2], [y1, y2], 'b-')

# read x and y coordinates to Python list
x = network.coors[0]
y = network.coors[1]

# plot vertices
plot(x, y, 'ro')
show()
