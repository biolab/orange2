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
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea=0)
        
        self.inputs = [("Data Table", ExampleTable, self.data)]
        self.rowcount = 0
        self.columncount = 0
        self.discattrcount = 0
        self.contattrcount = 0
        self.stringattrcount = 0
        self.metaattrcount = 0
        self.classattr = "no"
        
        box = OWGUI.widgetBox(self.controlArea, "Data Set Size", addSpace=True)
        OWGUI.label(box, self, '<table><tr><td width="150">Samples (rows):</td><td align="right" width="60">%(rowcount)7i</td></tr>\
                                <tr><td>Attributes (columns):</td><td align="right">%(columncount)7i</td></tr></table>')
        
        box = OWGUI.widgetBox(self.controlArea, "Attributes")
        OWGUI.label(box, self, '<table><tr><td width="150">Discrete attributes:</td><td align="right" width="60">%(discattrcount)7i</td></tr>\
                                <tr><td>Continuous attributes:</td><td align="right">%(contattrcount)7i</td></tr>\
                                <tr><td>String attributes:</td><td align="right">%(stringattrcount)7i</td></tr>\
                                <tr><td> </td></tr>\
                                <tr><td>Meta attributes:</td><td align="right">%(metaattrcount)7i</td></tr>\
                                <tr><td>Class attribute:</td><td align="right">%(classattr)7s</td></tr></table>')
#        OWGUI.separator(box)
#        OWGUI.label(box, self, '<table><tr><td width="100">Meta attributes:</td><td align="right" width="50">%(metaattrcount)7i</td></tr>\
#                                <tr><td>Class attribute:</td><td align="right">%(classattr)7s</td></tr></table>')
#        
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