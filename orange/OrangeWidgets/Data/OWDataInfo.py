"""<name>Info</name>
<description>Shows data information.</description>
<icon>icons/DataInfo.png</icon>
<priority>11</priority>
<contact>Ales Erjavec (ales.erjavec@fri.uni-lj.si)</contact>"""

from OWWidget import *
import OWGUI

import orange
import sys, os

class OWDataInfo(OWWidget):
    def __init__(self, parent=None, signalManager=None, name="Info"):
        OWWidget.__init__(self, parent, signalManager, name)
        
        self.inputs = [("Data Table", ExampleTable, self.data)]
        self.rowcount = 0
        self.columncount = 0
        self.discattrcount = 0
        self.contattrcount = 0
        self.stringattrcount = 0
        self.metaattrcount = 0
        self.classattr = "no"
        
        box = OWGUI.widgetBox(self.controlArea, "Data Set Size")
        OWGUI.label(box, self, "Samples (rows): %(rowcount)i\nAttributes (columns): %(columncount)i")
        
        box = OWGUI.widgetBox(self.controlArea, "Attributes")
        OWGUI.label(box, self, "Discrete attributes: %(discattrcount)i\nContinuous attributes: %(contattrcount)i\nString attributes: %(stringattrcount)i")
        OWGUI.separator(box)
        OWGUI.label(box, self, "Meta attributes: %(metaattrcount)i\nClass attribute: %(classattr)s")
        
        OWGUI.rubber(self.controlArea)
        
        
    def data(self, data):
        if data:
            self.rowcount = len(data)
            self.columncount = len(list(data.domain.attributes) + data.domain.getmetas().keys())
            self.discattrcount = len([attr for attr in data.domain.attributes if attr.varType == orange.VarTypes.Discrete])
            self.contattrcount = len([attr for attr in data.domain.attributes if attr.varType == orange.VarTypes.Continuous])
            self.stringattrcount = len([attr for attr in data.domain.attributes if attr.varType == orange.VarTypes.String])
            self.metaattrcount = len(data.domain.getmetas())
            self.classattr = "yes" if data.domain.classVar else "No"
        else:
            self.rowcount = 0
            self.columncount = 0
            self.discattroutn = 0
            self.contattrcount = 0
            self.stringattrcount = 0
            self.metaattrcount = 0
            self.classattr = "no"
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWDataInfo()
    data = orange.ExampleTable("../../doc/datasets/iris.tab")
    w.data(data)
    w.show()
    app.exec_()