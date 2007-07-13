"""
<name>Distance</name>
<description>Computes distances between documents.</description>
<icon>icons/TextDistance.png</icon>
<contact></contact> 
<priority>2200</priority>
"""

from OWWidget import *
import OWGUI, orngText

class OWTextDistance(OWWidget):
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self,parent,signalManager,"Preprocess")
        self.inputs = [("Example Table", ExampleTable, self.dataset)]
        self.outputs = [("Distance Matrix", orange.SymMatrix)]
        self.data = None
        self.loadSettings()
        OWGUI.button(self.controlArea, self, "Apply", self.apply)
        self.adjustSize()        

    def dataset(self, data):
        self.data = data
        self.apply()

    def apply(self):
        if self.data:
            dist = orngText.cos(self.data)
            dist.setattr("items", self.data)
            self.send("Distance Matrix", dist)
        else:
            self.send("Distance Matrix", None)
