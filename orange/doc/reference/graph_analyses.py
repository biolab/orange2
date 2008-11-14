import orange
from orngNetwork import NetworkOptimization
from pylab import *

        
# vertices are placed randomly in NetworkOptimization constructor
network = NetworkOptimization()

# read network from file
graph, table = network.readNetwork("combination.net")

components = graph.getConnectedComponents()
print "Connected components"
print components
print

distribution = graph.getDegreeDistribution()
print "Degree distribution"
print distribution
print

degrees = graph.getDegrees()
print "Degrees"
print degrees
print

hubs = graph.getHubs(3)
print "Hubs"
print hubs
print

path = graph.getShortestPaths(0, 2)
print "Shortest path"
print path
print

distance = graph.getDistance(0, 2)
print "Distance"
print distance
print

diameter = graph.getDiameter()
print "Diameter"
print diameter
print

subgraph = graph.getSubGraph([0, 1, 2, 3])
subNetwork = NetworkOptimization(subgraph)
subNetwork.fruchtermanReingold(100, 1000)

# read all edges in subnetwork and plot a line
for u, v in subNetwork.graph.getEdges():
    x1, y1 = subNetwork.coors[u][0], subNetwork.coors[u][1]
    x2, y2 = subNetwork.coors[v][0], subNetwork.coors[v][1]
    plot([x1, x2], [y1, y2], 'b-')        
        
# read x and y coordinates to Python list
x = [coordinate[0] for coordinate in subNetwork.coors]
y = [coordinate[1] for coordinate in subNetwork.coors]

# plot vertices of subnetwork
plot(x, y, 'ro')
show()
