"""
<name>Net Explorer 3D</name>
<description>Orange widget for network exploration in 3D.</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>411</priority>
"""

from OWNxExplorerQt import OWNxExplorerQt
from OWNxCanvas3D import OWNxCanvas3D

class OWNxExplorer3D(OWNxExplorerQt):
    def __init__(self, parent=None, signalManager=None, name='Net Explorer 3D', network_canvas=OWNxCanvas3D):
        OWNxExplorerQt.__init__(self, parent, signalManager, name, network_canvas)

