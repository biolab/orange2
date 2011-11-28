"""
<name>Shuffle</name>
<description>Shuffle the instances in a data table</description>

"""

from OWWidget import *
import OWGUI

import random
import Orange

class OWShuffle(OWWidget):
    def __init__(self, parent=None, signalManager=None, title="Suffle"):
        OWWidget.__init__(self, parent, signalManager, title,
                          wantMainArea=False)
        
        self.inputs = [("Data Table", Orange.data.Table, self.set_data)]
        self.outputs = [("Shuffled Data Table", Orange.data.Table)]
        
        self.seed = 0
        
        OWGUI.lineEdit(self.controlArea, self, "seed", box="Seed",
                       tooltip="Random seed",
                       callback=self.run,
                       valueType=int,
                       validator=QIntValidator()
                       )
        
    def set_data(self, data=None):
        self.data = data
        self.run()
        
    def run(self):
        shuffled = None
        if self.data is not None:
            rand = random.Random(self.seed)
            shuffled = list(self.data)
            rand.shuffle(shuffled)
            shuffled = Orange.data.Table(shuffled) if shuffled else \
                       Orange.data.Table(self.data.domain) # In case of empty table
        self.send("Shuffled Data Table", shuffled)
    