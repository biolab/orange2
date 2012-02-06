import Orange.network

from matplotlib import pyplot as plt

# vertices are placed randomly in NetworkOptimization constructor
network = Orange.network.NetworkOptimization()

# read network from file
net = Orange.network.Network.read("combination.net")

components = net.get_connected_components()
print "Connected components"
print components
print

distribution = net.get_degree_distribution()
print "Degree distribution"
print distribution
print

degrees = net.get_degrees()
print "Degrees"
print degrees
print

hubs = net.get_hubs(3)
print "Hubs"
print hubs
print

path = net.get_shortest_paths(0, 2)
print "Shortest path"
print path
print

distance = net.get_distance(0, 2)
print "Distance"
print distance
print

diameter = net.get_diameter()
print "Diameter"
print diameter
print

subnet = Orange.network.Network(net.get_sub_graph([0, 1, 2, 3]))
subNetOptimization = Orange.network.NetworkOptimization(subnet)
subNetOptimization.fruchterman_reingold(100, 1000)

# read all edges in subnetwork and plot a line
for u, v in subnet.getEdges():
    x1, y1 = subnet.coors[0][u], subnet.coors[1][u]
    x2, y2 = subnet.coors[0][v], subnet.coors[1][v]
    plt.plot([x1, x2], [y1, y2], 'b-')        
        
# read x and y coordinates to Python list
x = [coordinate for coordinate in subnet.coors[0]]
y = [coordinate for coordinate in subnet.coors[1]]

# plot vertices of subnetwork
plt.plot(x, y, 'ro')
plt.savefig("network-graph-analysis.png")
