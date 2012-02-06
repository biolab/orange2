import Orange.network

from matplotlib import pyplot as plt

# create graph object of type GraphAsList
net = Orange.network.Graph()
net.add_nodes_from(range(5))

# set edges
for i in range(4):
    for j in range(i + 1, 5):
        net.add_edge(i, j)

# vertices are placed randomly in NetworkOptimization constructor
net_layout = Orange.network.GraphLayout()
net_layout.set_graph(net)

# optimize verices layout with one of included algorithms
net_layout.fr(100, 1000)

# read all edges and plot a line
for u, v in net.edges():
    x1, y1 = net_layout.coors[0][u], net_layout.coors[1][u]
    x2, y2 = net_layout.coors[0][v], net_layout.coors[1][v]
    plt.plot([x1, x2], [y1, y2], 'b-')

# read x and y coordinates to Python list
x = net_layout.coors[0]
y = net_layout.coors[1]

# plot vertices
plt.plot(x, y, 'ro')
plt.savefig("network-optimization-nx.png")
