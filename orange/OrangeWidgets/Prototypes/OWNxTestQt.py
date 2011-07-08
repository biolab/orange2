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
        
        for i in range(100):
            n = NodeItem(i)
            n.x = random.random()
            n.y = random.random()
            qDebug('Adding node ' + str(n.x) + ' ' + str(n.y))
            curve.nodes[i] = n
            
        qDebug(str(len(curve.nodes)))
            
        for i in range(0, 100, 5):
            for j in range(0, 100, 7):
                if j == i:
                    continue
                else:
                    e = EdgeItem()
                    e.u = curve.nodes[i]
                    e.v = curve.nodes[j]
                    curve.edges.append(e)
        
        curve.updateProperties()
        self.canvas.replot()
            
        