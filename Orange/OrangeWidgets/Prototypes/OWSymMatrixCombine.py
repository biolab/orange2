"""
<name>Combine Matrices</name>
<description>Combine multiple matrices</description>

"""

from OWWidget import *
import OWGUI
import Orange
import numpy
import math

from Orange.misc import progress_bar_milestones

def mul(values):
    return reduce(float.__mul__, values, 1.0)

def geometric_mean(vals):
    prod = mul(vals)
    return math.pow(abs(prod), 1.0/len(vals))

def harmonic_mean(vals):
    hsum = sum(map(lambda v: 1.0/(v or 1e-7), vals))
    return len(vals) / hsum

class OWSymMatrixCombine(OWWidget):
    METHODS = [("Add", sum),
               ("Multiply", mul),
               ("Mean", numpy.mean),
               ("Median", numpy.median),
               ("Geometric Mean", geometric_mean),
               ("Harmonic Mean", harmonic_mean)
               ]
    
    settingsList = ["selected_method"]
    def __init__(self, parent=None, signalManager=None, title="Combine Matrices"):
        OWWidget.__init__(self, parent, signalManager, title, wantMainArea=False)
        
        self.inputs = [("Matrices", Orange.misc.SymMatrix, self.set_matrix, Multiple)]
        self.outputs = [("Combined Matrix", Orange.misc.SymMatrix, Multiple)]
        
        self.selected_method = 0
        
        box = OWGUI.widgetBox(self.controlArea, "Method")
        OWGUI.comboBox(box, self, "selected_method", 
                       items=[t[0] for t in self.METHODS],
                       tooltip="Select the method for combining the matrices",
                       callback=self.method_changed)
        
        OWGUI.rubber(self.controlArea)
        self.matrices = {}
        self.resize(150,30)
        
    def set_matrix(self, matrix=None, id=None):
        if id in self.matrices:
            # A new value on the signal
            if matrix is None:
                del self.matrices[id]
            else:
                self.matrices[id] = matrix
        elif matrix is not None:
            # New signal
            self.matrices[id] = matrix
            
    def handleNewSignals(self):
        self.compute_combined()
        
    def method_changed(self):
        self.compute_combined()
        
    def compute_combined(self):
        self.error(0)
        dim = 0
        matrices = self.matrices.values()
        new = None
        if matrices and len(set(m.dim for m in matrices)) != 1:
            # Do dimensions match
            self.error(0, "Matrices are of different dimensions.")
        elif matrices:
            dim = matrices[0].dim
            method = self.METHODS[self.selected_method][1]
            new = Orange.misc.SymMatrix(dim)
            milestones = progress_bar_milestones(100, dim*(dim + 1) / 2)
            iter = 0
            for i in range(dim):
                for j in range(i, dim):
                    vals = [m[i, j] for m in matrices]
                    new[i, j] = method(vals)
                    
            items = [getattr(m, "items") for m in matrices]
            items = filter(None, items)
            if items:
                new.setattr("items", items[0])
            
        self.send("Combined Matrix", new)
        
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = OWSymMatrixCombine()
    w.show()
    app.exec_()
        
