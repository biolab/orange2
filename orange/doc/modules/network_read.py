import orange
from orngNetwork import NetworkOptimization
from pylab import *

        
# vertices are placed randomly in NetworkOptimization constructor
network = NetworkOptimization()

# read network from file
network.readNetwork("K5.net")

# read all edges and plot a line
for u, v in network.graph.getEdges():
    x1, y1 = network.coors[u][0], network.coors[u][1]
    x2, y2 = network.coors[v][0], network.coors[v][1]
    plot([x1, x2], [y1, y2], 'b-')        
        
# read x and y coordinates to Python list
x = [coordinate[0] for coordinate in network.coors]
y = [coordinate[1] for coordinate in network.coors]

# plot vertices
plot(x, y, 'ro')
show()
