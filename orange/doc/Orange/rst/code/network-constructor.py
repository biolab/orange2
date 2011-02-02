import Orange.network
from pylab import *

# create graph object of type GraphAsList
# vertices are placed randomly in Network constructor
net = Orange.network.Network(5, 0)

# set edges
for i in range(4):
    for j in range(i + 1, 5):
        net[i,j] = 1

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
