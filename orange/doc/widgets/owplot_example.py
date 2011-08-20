



from OWWidget import *
import OWGUI
from plot.owplot import *

class BasicPlot(OWPlot):
    pass

class BasicWidget(OWWidget):
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Basic')
        self.inputs = [("Examples", ExampleTable, self.set_data)]
        
        self.plot = BasicPlot(self, self.mainArea, "Example plot")
            
    def set_data(self, data):        
        if data is not None and (len(data) == 0 or len(data.domain) == 0):
            data = None
            
        self.data = data
        n = len(data) # The number of attributes in data
        
        x_index = 0
        y_index = 1 if n > 1 else 0
        
        if data.domain[x_index].varType = Orange.VarType.Discrete:
            self.plot.set_axis_labels(OWPlot.xBottom, get_variable_names_sorted(self.data.domain[x_index]))
        if data.domain[y_index].varType = Orange.VarType.Discrete:
            self.plot.set_axis_labels(OWPlot.yLeft, get_variable_names_sorted(self.data.domain[y_index]))
            
        color_data = data[2] if n > 2 else [self.plot.color(OWPalette.Data)]
        size = data[3] if n > 3 else [10]
        
        self.plot.set_main_curve_data()
        