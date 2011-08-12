"""
<name>Pubmed Network View</name>
<description></description>
<icon>icons/Network.png</icon>
<contact></contact> 
<priority>6450</priority>
"""

import Orange
import OWGUI

from OWWidget import *

class PubmedNetworkView(Orange.network.NxView):
    """Network Inside View
    
    """
    
    def __init__(self, nhops):
        Orange.network.NxView.__init__(self)
        
        self._nhops = nhops
        self._center_node = None
        
    def init_network(self, graph):
        self._network = graph
        
        if graph is None:
            return None
        
        self._center_node = graph.nodes_iter().next()
        nodes = self._get_neighbors()
        return Orange.network.nx.Graph.subgraph(self._network, nodes)
    
    def update_network(self):
        nodes = self._get_neighbors()
        subnet = Orange.network.nx.Graph.subgraph(self._network, nodes)

        if self._nx_explorer is not None:
            self._nx_explorer.change_graph(subnet)
        
    def set_nhops(self, nhops):
        self._nhops = nhops
        
    def node_selection_changed(self):
        selection = self._nx_explorer.networkCanvas.selected_nodes()
        if len(selection) == 1:
            self._center_node = selection[0]
            self.update_network()
        
    def _get_neighbors(self):
        nodes = set([self._center_node])
        for n in range(self._nhops):
            neighbors = set()
            for node in nodes:
                neighbors.update(self._network.neighbors(node))
            nodes.update(neighbors)
        return nodes
        
class OWPubmedView(OWWidget):
    
    settingsList = ['_nhops']
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Pubmed Network View')
        
        self.inputs = []
        self.outputs = [("Nx View", Orange.network.NxView)]
        
        self._nhops = 2
        
        self.loadSettings()
        
        ib = OWGUI.widgetBox(self.controlArea, "Preferences", orientation="vertical")
        OWGUI.spin(ib, self, "_nhops", 1, 6, 1, label="Number of hops: ", callback=self.update_view)

        self.inside_view = PubmedNetworkView(self._nhops)
        self.send("Nx View", self.inside_view)
    
    def update_view(self):
        self.inside_view.set_nhops(self._nhops)
        
        self.inside_view.update_network()
    