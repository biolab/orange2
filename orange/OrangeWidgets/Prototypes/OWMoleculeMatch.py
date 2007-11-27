"""<name>Molecule Match</name>
<description>Selection of molecules based on SMILES fragments</description>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact> 
<priority>2020</priority>"""

import orange
import orngChem
from OWWidget import *
from qt import *
import OWGUI
import sys
import os
#import vis
import pybel

class OWMoleculeMatch(OWWidget):
    settingsList = ["recentFragments"]
    def __init__(self, parent=None, signalManager=None, name="Molecule Match"):
        apply(OWWidget.__init__, (self, parent, signalManager, name))
        self.inputs = [("Molecules", ExampleTable, self.SetMoleculeTable)]
        self.outputs = [("Selected molecules", ExampleTable), ("All molecules", ExampleTable)]
        self.fragment = ""
        self.recentFragments = []
        self.comboSelection = 0 
        self.smilesAttr = 0
        self.smilesAttrList = []
        self.data = None
        self.loadSettings()

        ##GUI        
        self.lineEdit = OWGUI.lineEdit(self.controlArea, self, "fragment", box="Fragment", callback=self.LineEditSelect)
        OWGUI.separator(self.controlArea)
        self.fragmentComboBox = OWGUI.comboBox(self.controlArea, self, "comboSelection", "Recent Fragments", items=self.recentFragments, callback=self.RecentSelect)
        OWGUI.separator(self.controlArea)
        self.smilesAttrCombo = OWGUI.comboBox(self.controlArea, self, "smilesAttr", "Fragment SMILES attribute", callback=self.Process)
        OWGUI.separator(self.controlArea)
        self.resize(100, 100)

    def SetMoleculeTable(self, data=None):
        self.data = data
        if data:
            self.SetSmilesAttrCombo()
            self.Process()

    def SetSmilesAttrCombo(self):
        fragmenter = orngChem.Fragmenter()
        attr = [fragmenter.FindSmilesAttr(self.data)]
        self.smilesAttrList=attr
        if self.smilesAttr > len(attr)-1:
            self.smilesAttr = 0
        self.smilesAttrCombo.clear()
        self.smilesAttrCombo.insertStrList(map(str, attr))
            
    def LineEditSelect(self):
        self.Process()
        self.recentFragments.insert(0, self.fragment)
        self.fragmentComboBox.insertItem(self.fragment, 0)
        if len(self.recentFragments) > 10:
            self.recentFragments = self.recentFragments[:10]

    def RecentSelect(self):
        self.fragment = self.recentFragments[self.comboSelection]
        self.Process()

    def Process(self):
        from functools import partial
        newVar = orange.FloatVariable(str(self.fragment), numberOfDecimals=0)
        def getVal(var, smilesAttr, smarts ,example, returnWhat):
            mol = pybel.readstring("smi", str(example[self.smilesAttrList[smilesAttr]]))
            if smarts.findall(mol):
                return var(1)
            else:
                return var(0)
        newVar.getValueFrom  = partial(getVal, newVar, self.smilesAttr, pybel.Smarts(str(self.fragment)))
        vars = list(self.data.domain.attributes) + [newVar] + (self.data.domain.classVar and [self.data.domain.classVar] or [])
        domain = orange.Domain(vars)
        domain.addmetas(self.data.domain.getmetas())
        selected = []
        all = []
        for example in self.data:
            ex = orange.Example(domain, example)
            if ex[newVar]==newVar(1):
                selected.append(example)
            all.append(ex)
        all = orange.ExampleTable(domain, all)
        if selected:
            selected = orange.ExampleTable(selected)
        else:
            selected = orange.ExampleTable(self.data.domain)
        self.send("Selected molecules", selected)
        self.send("All molecules", all)

if __name__=="__main__":
    app = QApplication(sys.argv)
    w = OWMoleculeMatch()
    w.show()
    app.setMainWidget(w)
    data = orange.ExampleTable("E:/orangecvs/Chemo/smiles.tab")
    w.SetMoleculeTable(data)
    app.exec_loop()