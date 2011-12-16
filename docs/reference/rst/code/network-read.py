import Orange.network

from matplotlib import pyplot as plt

# read network from file
net = Orange.network.Network.read("K5.net")

# read all edges and plot a line
for u, v in net.get_edges():
    x1, y1 = net.coors[0][u], net.coors[1][u]
    x2, y2 = net.coors[0][v], net.coors[1][v]
    plt.plot([x1, x2], [y1, y2], 'b-')

# read x and y coordinates to Python list
x = net.coors[0]
y = net.coors[1]

# plot vertices
plt.plot(x, y, 'ro')
plt.savefig("network-read.py.png")
