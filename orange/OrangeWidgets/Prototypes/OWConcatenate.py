"""
<name>Concatenate</name>
<description>Concatenates Example Tables.</description>
<icon>icons/Concatenate.png</icon>
<priority>12</priority>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
"""

from OWWidget import *
import OWGUI

class OWConcatenate(OWWidget):
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "FeatureConstructor")

        self.inputs = [("Primary Table", orange.ExampleTable, self.setData), ("Additional Tables", orange.ExampleTable, self.setMoreData, Multiple)]
        self.outputs = [("Examples", ExampleTable)]

        self.primary = None
        self.additional = {}

        OWGUI.widgetLabel(self.controlArea, "This widget has no parameters.")
        self.adjustSize()

    def setData(self, data):
        self.primary = data
        self.apply()
        
    def setMoreData(self, data, id):
        if not data:
            if id in self.additional:
                del self.additional[id]
        else:
            self.additional[id] = data
        self.apply()
        
    def apply(self):
        self.warning(1)
        if not self.primary:
            if self.additional:
                self.warning(1, "Primary data table is missing")
            self.send("Examples", None)
            return
            
        if not self.additional:
            self.send("Examples", self.primary)
            
        newTable = orange.ExampleTable(self.primary)
        for additional in self.additional.values():
            newTable.extend(additional)
        self.send("Examples", newTable)
