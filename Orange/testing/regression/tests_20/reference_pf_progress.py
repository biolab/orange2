import orngNetwork
from orangeom import Pathfinder

def myCb(complete):
    """
    The callback function.
    """
    print 'The procedure is %d%% complete.' % int(complete * 100)

# Read a demo network from a file
net = orngNetwork.Network.read('demo.net')

# Choose some parameters
r, q = 1, 6

# Create a pathfinder instance
pf = Pathfinder()

# Pass the reference to the desired function
pf.setProgressCallback(myCb)

# Simplify the network
pf.simplify(r, q, net)