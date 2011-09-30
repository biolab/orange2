"""
<name>Net Explorer 3D</name>
<description>Orange widget for network exploration in 3D.</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>411</priority>
"""

from OWNxExplorerQt import OWNxExplorerQt
from OWNxCanvas3D import OWNxCanvas3D
import Orange

from OWNxCanvasQt import Default, AttributeList

class OWNxExplorer3D(OWNxExplorerQt):
    def __init__(self, parent=None, signalManager=None, name='Net Explorer 3D', network_canvas=OWNxCanvas3D):
        OWNxExplorerQt.__init__(self, parent, signalManager, name, network_canvas)

        self.inputs = [("Nx View", Orange.network.NxView, self.set_network_view),
                       ("Network", Orange.network.Graph, self.set_graph, Default),
                       ("Items", Orange.data.Table, self.setItems),
                       ("Items to Mark", Orange.data.Table, self.markItems), 
                       ("Items Subset", Orange.data.Table, self.setExampleSubset), 
                       ("Items Distance Matrix", Orange.core.SymMatrix, self.set_items_distance_matrix)]
        
        self.outputs = [("Selected Network", Orange.network.Graph),
                        ("Selected Items Distance Matrix", Orange.core.SymMatrix),
                        ("Selected Items", Orange.data.Table), 
                        ("Unselected Items", Orange.data.Table), 
                        ("Marked Items", Orange.data.Table),
                        ("Attribute Selection List", AttributeList)]


