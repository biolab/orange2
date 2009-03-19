import orange
from orangeom import Network
from orngNetwork import NetworkOptimization
from pylab import *

        
# vertices are placed randomly in NetworkOptimization constructor
network = NetworkOptimization()

# read network from file
net = network.readNetwork("combination.net")

components = net.getConnectedComponents()
print "Connected components"
print components
print

distribution = net.getDegreeDistribution()
print "Degree distribution"
print distribution
print

degrees = net.getDegrees()
print "Degrees"
print degrees
print

hubs = net.getHubs(3)
print "Hubs"
print hubs
print

path = net.getShortestPaths(0, 2)
print "Shortest path"
print path
print

distance = net.getDistance(0, 2)
print "Distance"
print distance
print

diameter = net.getDiameter()
print "Diameter"
print diameter
print

subnet = Network(net.getSubGraph([0, 1, 2, 3]))
subNetOptimization = NetworkOptimization(subnet)
subNetOptimization.fruchtermanReingold(100, 1000)

# read all edges in subnetwork and plot a line
for u, v in subnet.getEdges():
    x1, y1 = subnet.coors[0][u], subnet.coors[1][u]
    x2, y2 = subnet.coors[0][v], subnet.coors[1][v]
    plot([x1, x2], [y1, y2], 'b-')        
        
# read x and y coordinates to Python list
x = [coordinate for coordinate in subnet.coors[0]]
y = [coordinate for coordinate in subnet.coors[1]]

# plot vertices of subnetwork
plot(x, y, 'ro')
show()
