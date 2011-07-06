"""
<name>Nx Inside View</name>
<description>Orange widget for community detection in networks</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>6440</priority>
"""

import Orange
import OWGUI

from OWWidget import *

class NxInsideView(Orange.network.NxView):
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
        return self._network.subgraph(nodes)
    
    def update_network(self):
        nodes = self._get_neighbors()
        subnet = self._network.subgraph(nodes)
        
        if self._nx_explorer is not None:
            self._nx_explorer.change_graph(subnet)
        
    def set_nhops(self, nhops):
        self._nhops = nhops
        
    def nodes_selected(self):
        selection = self._nx_explorer.networkCanvas.get_selected_nodes()
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
        
class OWNxInsideView(OWWidget):
    
    settingsList = ['_nhops']
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Nx Inside View')
        
        self.inputs = []
        self.outputs = [("Nx View", Orange.network.NxView)]
        
        self._nhops = 2
        
        self.loadSettings()
        
        ib = OWGUI.widgetBox(self.controlArea, "Preferences", orientation="vertical")
        OWGUI.spin(ib, self, "_nhops", 1, 6, 1, label="Number of hops: ", callback=self.update_view)

        self.inside_view = NxInsideView(self._nhops)
        self.send("Nx View", self.inside_view)
    
    def update_view(self):
        self.inside_view.set_nhops(self._nhops)
        
        self.inside_view.update_network()
    