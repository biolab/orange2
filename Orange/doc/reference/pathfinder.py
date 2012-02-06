import orngNetwork
from orangeom import Pathfinder
from pylab import *

def myPlot(net, titleTxt=''):
    """
    Displays the given network.
    """
    figure()
    title(titleTxt)
    # Plot the edges
    for u, v in net.getEdges():
        x1, y1 = net.coors[0][u], net.coors[1][u]
        x2, y2 = net.coors[0][v], net.coors[1][v]
        plot([x1, x2], [y1, y2], 'b-')
    # Plot the nodes
    for u in range(net.nVertices):
        x, y = net.coors[0][u], net.coors[1][u]
        plot(x, y, 'ro')
        # Label
        text(x, y + 100, net.items[u][1])
        
# Read a demo network from a file
net = orngNetwork.Network.read('demo.net')

# Compute a layout for plotting
netOp = orngNetwork.NetworkOptimization(net)
netOp.fruchtermanReingold(100, 1000)

# Plot the original
myPlot(net, 'Original network')

# Choose some parameters
r, q = 1, 6

# Create a pathfinder instance
pf = Pathfinder()

# Simplify the network
pf.simplify(r, q, net)

# Plot the simplified network
myPlot(net, 'Simplified network')
show()