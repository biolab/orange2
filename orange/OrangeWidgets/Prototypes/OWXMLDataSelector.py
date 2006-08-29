"""
<name>XML Data Selector</name>
<description></description>
<icon>icons/ca.png</icon>
<priority>3600</priority>
"""

from qt import *
from OWWidget import *
import OWGUI, OWToolbars, OWDlgs
from orngXMLData import orngXMLData

class OWXMLDataSelector(OWWidget):
    settingsList = []                    
    contextHandlers = {}
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'XML Data selector')
        
        self.inputs = [("XML Data file", orngXMLData, self.dataset)]
        self.outputs = [("XML Data file", orngXMLData)]
        
        self.data = None
        
        self.mainArea.setFixedWidth(0)
        ca = QFrame(self.controlArea)
        ca.adjustSize()
        gl=QGridLayout(ca,2,4,5)      
        
        wb1 = OWGUI.widgetBox(ca, "All categories")
        self.allCategories = []
        self.allCategoriesSel = []
        self.allCategoriesLB = OWGUI.listBox(wb1, self, "allCategoriesSel", "allCategories",  selectionMode = QListBox.Multi)
        gl.addMultiCellWidget(wb1, 0, 1, 0, 0)
        
        vbox = QVBox(ca)
        vbox.adjustSize()
        OWGUI.button(vbox, self, '>', callback = self.loadCategories)
        OWGUI.button(vbox, self, '<', callback = self.unloadCategories)
        gl.addMultiCellWidget(vbox, 0, 1, 1, 1)
        
        wb2 = OWGUI.widgetBox(ca, "Selected categories")        
        self.selCategories = []
        self.selCategoriesSel = []
        self.selCategoriesLB = OWGUI.listBox(wb2, self, "selCategoriesSel", "selCategories",  selectionMode = QListBox.Multi)
        gl.addMultiCellWidget(wb2, 0, 1, 2, 2)        
        
        
        vbox1_ = QVBox(ca)
        vbox1 = QVGroupBox("Selected documents", vbox1_)
        self.allDocs = []
        self.allDocsSel = []
        self.allDocsLB = OWGUI.listBox(vbox1, self, "allDocsSel", "allDocs",  selectionMode = QListBox.NoSelection)
##        OWGUI.button(vbox1, self, 'Show docs', callback = self.clicked)
        OWGUI.button(vbox1_, self, 'Apply', callback = self.apply)
        gl.addMultiCellWidget(vbox1_, 0, 1, 3, 3) 
        
        self.resize(800, 1000)
        
    def dataset(self, data):
        self.data = data
        if data:
            self.allCategoriesLB.clear()
            self.selCategoriesLB.clear()
            self.allDocsLB.clear()
            self.allCategories = self.data.getCategories()
        else:
            self.allCategoriesLB.clear()
            self.selCategoriesLB.clear()
            self.allDocsLB.clear()            
            
    def loadCategories(self):
        if not self.allCategoriesSel: return
        for cat in self.allCategoriesSel:
            self.selCategories.append(self.allCategories.pop(cat))
        self.allCategoriesSel = []
        self.allCategories = self.allCategories[:]
        self.selCategories = self.selCategories[:]
        self.updateDocs()
    def unloadCategories(self):
        if not self.selCategoriesSel: return
        for cat in self.selCategoriesSel:
            self.allCategories.append(self.selCategories.pop(cat))
        self.selCategoriesSel = []
        self.allCategories = self.allCategories[:]
        self.selCategories = self.selCategories[:]
        self.updateDocs()
    def apply(self):
        pass
    def updateDocs(self):
        if not self.selCategories: 
            self.allDocs = []
        else:
            self.allDocs = self.data.getDocumentInCategories(self.selCategories)
        
        
if __name__ == "__main__":
    a = orngXMLData('reuters-exchanges.xml')
    appl = QApplication(sys.argv) 
    ow = OWXMLDataSelector() 
    appl.setMainWidget(ow) 
    ow.show() 
    ow.dataset(a)
    appl.exec_loop() 
