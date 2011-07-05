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
        self._network = None
        
    def init_network(self, graph):
        self._network = graph
        
        if graph is None:
            return None
        
        for node in graph.nodes_iter():
            self._center_node = node
            break
         
        nodes = set([self._center_node])
        for n in range(self._nhops):
            neighbors = set()
            for node in nodes:
                neighbors.update(graph.neighbors(node))
            nodes.update(neighbors)
            
        return graph.subgraph(nodes)
    
    def update_network(self):
        pass
    
    def set_nhops(self, nhops):
        self._nhops = nhops
        
        
class OWNxInsideView(OWWidget):
    
    settingsList = ['_nhops']
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Nx Inside View')
        
        self.inputs = []
        self.outputs = [("Nx View", Orange.network.NxView)]
        
        self._nhops = 2
        
        self.loadSettings()
        
        ib = OWGUI.widgetBox(self.controlArea, "Preferences", orientation="vertical")
        OWGUI.spin(ib, self, "_nhops", 1, 6, 1, label="Number of hops: ", callback = self._update_view)

        self.inside_view = NxInsideView(self._nhops)
        self.send("Nx View", self.inside_view)
    
    def _update_view(self):
        self.inside_view.set_nhops(self._nhops)
        
        self.inside_view.update_network()
    