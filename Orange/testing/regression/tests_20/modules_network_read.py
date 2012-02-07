import orngNetwork
from pylab import *

# read network from file
net = orngNetwork.Network.read("K5.net")

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
