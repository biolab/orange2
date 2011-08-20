
from OWWidget import *
import OWGUI
from plot.owplot import *
import random
import orange
from Orange.preprocess.scaling import get_variable_values_sorted

class BasicPlot(OWPlot):
    pass

class BasicWidget(OWWidget):
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Basic')
        self.inputs = [("Examples", ExampleTable, self.set_data)]
        
        self.plot = BasicPlot(self.mainArea, "Example plot", widget = self)
        self.mainArea.layout().addWidget(self.plot)
        random.seed(7)
        self.time_id = self.startTimer(5000)
            
    def set_data(self, data):        
        if data is None or len(data) == 0 or len(data.domain) == 0:
            return
            
        self.data = data
        domain = data.domain
        n = len(data.domain) # The number of attributes in data
        
        y_i, x_i, c_i, s_i = [int(random.random() * n) for i in range(4)]
        qDebug(' '.join(str(i) for i in [y_i, x_i, c_i, s_i]))
        
        self.plot.set_axis_title(xBottom, domain[x_i].name)
        self.plot.set_show_axis_title(xBottom, True)
        self.plot.set_axis_title(yLeft, domain[y_i].name)
        self.plot.set_show_axis_title(yLeft, True)
        
        if data.domain[x_i].varType == orange.VarTypes.Discrete:
            self.plot.set_axis_labels(xBottom, get_variable_values_sorted(domain[x_i]))
        else:
            self.plot.set_axis_autoscale(xBottom)
        if data.domain[y_i].varType == orange.VarTypes.Discrete:
            self.plot.set_axis_labels(yLeft, get_variable_values_sorted(domain[y_i]))
        else:
            self.plot.set_axis_autoscale(yLeft)
            
        x_data = []
        y_data = []
        c_data = []
        s_data = []
        
        color_cont = False
        
        if domain[c_i].varType == orange.VarTypes.Discrete:
            palette = self.plot.discPalette
        else:
            palette = self.plot.contPalette
            color_cont = True
        
        legend_colors = set()
        legend_sizes = set()
        
        for e in data:
            x_data.append(e[x_i])
            y_data.append(e[y_i])
            color = QColor(*palette.getRGB(e[c_i]))
            c_data.append(color)
            size = 5 + round(e[s_i])
            s_data.append(size)
            
            if color_cont:
                legend_colors.add(float(e[c_i]))
            else:   
                legend_colors.add( (color, int(e[c_i])) ) 
            legend_sizes.add( (size, float(e[s_i])) )
            
        self.plot.legend().clear()
            
        if domain[s_i].varType == orange.VarTypes.Discrete:
            for size, value in legend_sizes:
                self.plot.legend().add_item( domain[s_i].name, "%.1f" % value, OWPoint(OWPoint.Diamond, self.plot.color(OWPalette.Data), size) )
            
        if color_cont:
            self.plot.legend().add_color_gradient(domain[c_i].name, ("%.1f" % min(legend_colors), "%.1f" % max(legend_colors)))
        else:
            for color, value in legend_colors:
                self.plot.legend().add_item( domain[c_i].name, "%.1f" % value, OWPoint(OWPoint.Diamond, color, 5) )
                   
        self.plot.set_main_curve_data(x_data, y_data, color_data=c_data, label_data = [], size_data=s_data, shape_data = [OWPoint.Diamond])
        
    def timerEvent(self, event):
        self.set_data(self.data)
    
    
#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=BasicWidget()
    ow.show()
    data = orange.ExampleTable(r"../../doc/datasets/iris.tab")
    ow.set_data(data)
    ow.handleNewSignals()
    a.exec_()
    #save settings
    ow.saveSettings()
        