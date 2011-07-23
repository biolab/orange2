"""
<name>Nx Test (Qt)</name>
<description>Orange widget for network exploration.</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>7607</priority>
"""

from OWWidget import *
from OWNxCanvasQt import *
import random

class OWNxTestQt(OWWidget):
    settingsList = []
    
    def __init__(self, parent=None, signalManager=None, name="Nx Test"):
        OWWidget.__init__(self, parent, signalManager, name)
        self.inputs = []
        self.outputs = []
        
        self.canvas = OWNxCanvas(self, self.mainArea, "Nx Test Canvas")
        self.mainArea.layout().addWidget(self.canvas)
       
        curve = self.canvas.networkCurve
        
        
        nodes_to_add = dict((i, NodeItem(i, random.random(), random.random(), parent=curve)) for i in range(30))
        curve.set_nodes(nodes_to_add)
        
        edges_to_add = [ EdgeItem(nodes_to_add[2*i], nodes_to_add[3*i], parent=curve) for i in range(10) ]
        curve.set_edges(edges_to_add)

        
        curve.updateProperties()
        self.canvas.replot()
            
        