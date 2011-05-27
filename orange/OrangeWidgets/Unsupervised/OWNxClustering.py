"""
<name>Nx Clustering</name>
<description>Orange widget for community detection in networks</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>6440</priority>
"""

import Orange
import OWGUI

from OWWidget import *

class OWNxClustering(OWWidget):
    
    settingsList = ['method', 'iterationHistory', "autoApply"]
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Nx Clustering')
        
        self.inputs = [("Network", Orange.network.Graph, self.setNetwork, Default)]
        self.outputs = [("Network", Orange.network.Graph)]
        
        self.net = None
        self.method = 0
        self.iterationHistory = 0
        self.autoApply = 0
        
        self.loadSettings()
        
        ribg = OWGUI.radioButtonsInBox(self.controlArea, self, "method", [], "Method", callback = self.cluster)
        OWGUI.appendRadioButton(ribg, self, "method", "Label propagation clustering (Raghavan et al., 2007)", callback = self.cluster)
        OWGUI.checkBox(OWGUI.indentedBox(ribg), self, "iterationHistory", "Append clustering data on each iteration", callback = self.cluster)
        self.info = OWGUI.widgetLabel(self.controlArea, ' ')
        autoApplyCB = OWGUI.checkBox(self.controlArea, self, "autoApply", "Commit automatically")
        OWGUI.button(self.controlArea, self, "Commit", callback=self.cluster)
        
    def setNetwork(self, net):
        self.net = net
        if self.autoApply:
            self.cluster()
        
    def cluster(self):
        self.info.setText(' ')
        
        if self.net == None:
            self.send("Network", None)
            return
        
        labels = Orange.network.community.label_propagation(self.net, results2items=1, resultHistory2items=self.iterationHistory)
        self.info.setText('%d clusters found' % len(set(labels)))        
        self.send("Network", self.net)