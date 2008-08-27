"""
<name>Distance Matrix Filter</name>
<description>Filters distance matrix</description>
<contact>Miha Stajdohar</contact>
<icon>icons/DistanceFilter.png</icon>
<priority>1160</priority>
"""

import orngOrangeFoldersQt4
import orange
import OWGUI
import exceptions
from OWWidget import *
import os.path
import pickle

class OWDistanceFilter(OWWidget):
    
    def __init__(self, parent=None, signalManager = None, name='Distance Matrix Filter'):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)
        
        self.inputs = [("Distance Matrix", orange.SymMatrix, self.setSymMatrix, Default), ("Example Subset", ExampleTable, self.setExampleSubset)]
        self.outputs = [("Distance Matrix", orange.SymMatrix)]
        
        self.matrix = None
        self.subset = None
        self.subsetAttr = 0
        self.icons = self.createAttributeIconDict()
        
        subsetBox = OWGUI.widgetBox(self.controlArea, box='Filter by Subset', orientation='vertical')
        
        self.subsetAttrCombo = OWGUI.comboBox(subsetBox, self, "subsetAttr", callback=self.filter)
        self.subsetAttrCombo.addItem("(none)")
        
        self.resize(200, 50)
        
    def setSymMatrix(self, sm):
        self.matrix = sm
        self.newInput()
    
    def setExampleSubset(self, et):
        self.subset = et
        self.newInput()
    
    def newInput(self):
        self.warning()
        self.error()
        
        if self.matrix == None or self.subset == None:
            return
        
        if not (hasattr(self.matrix, 'items') and self.matrix.items != None and type(self.matrix.items) == type(self.subset)):
            self.error('Distance Matrix has no attribute items of type ExampleTable')
            return
        
        self.subsetAttrCombo.clear()
        self.subsetAttrCombo.addItem("(none)")
        
        intemsVar = [var.name for var in self.matrix.items.domain]
        for var in self.subset.domain:
            if var.name in intemsVar:
                print var.name
                self.subsetAttrCombo.addItem(self.icons[var.varType], unicode(var.name))
        
        self.send("Distance Matrix", self.matrix)
        
    def filter(self):
        if self.subsetAttr > 0:
            print self.subsetAttrCombo.currentText()
            col = str(self.subsetAttrCombo.currentText())
            print self.subset.domain
            
            filter = [str(x[col]) for x in self.subset]
            filter = set(filter)
            
            nodes = [x for x in range(len(self.matrix.items)) if str(self.matrix.items[x][col]) in filter]
            print nodes
            
            nNodes = len(nodes)
            print "nNodes:", nNodes
            matrix = orange.SymMatrix(nNodes)
            
            for i in range(nNodes):
                for j in range(i):
                    matrix[i,j] = self.matrix[nodes[i], nodes[j]]
                    
            matrix.items = self.matrix.items.getitems(nodes)
                    
            self.send("Distance Matrix", matrix)

if __name__=="__main__":
    import orange
    a = QApplication(sys.argv)
    ow = OWDistanceFilter()
    ow.show()
    a.exec_()
    ow.saveSettings()