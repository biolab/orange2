from qt import *
from qwt import *
import sys
import cPickle
import os

class InterestingProjections(QWidget):
    def __init__(self,parent=None):
        QWidget.__init__(self, parent)
        self.space=QVBox(self)
        self.grid=QGridLayout(self)
        self.grid.addWidget(self.space,0,0)
        self.interestingGroup = QVGroupBox(self.space)
        self.interestingGroup.setTitle("Interesting visualizations")
        self.interestingList = QListBox(self.interestingGroup)
        self.setCaption("Qt Interesting visualizations")
        self.resize(200, 500)
        self.filterButton = QPushButton("Remove attribute", self.space)
        self.removeSelectedButton = QPushButton("Remove selected projections", self.space)
        self.saveButton = QPushButton("Save", self.space)
        self.loadButton = QPushButton("Load", self.space)
        self.closeButton = QPushButton("Close", self.space)
        self.connect(self.filterButton, SIGNAL("clicked()"), self.filter)
        self.connect(self.removeSelectedButton, SIGNAL("clicked()"), self.removeSelected)
        self.connect(self.saveButton, SIGNAL("clicked()"), self.save)
        self.connect(self.loadButton, SIGNAL("clicked()"), self.load)
        self.connect(self.closeButton, SIGNAL("clicked()"), self.hide)

        self.interestingList.setSelectionMode(QListBox.Extended)
        self.widgetDir = sys.prefix + "/lib/site-packages/Orange/OrangeWidgets/"
        self.parentName = "Projection"
        #self.domainName = "Unknown"
        self.kValue = 1
        self.optimizedList = []

    def clear(self):
        self.optimizedList = []
        self.interestingList.clear()

    def filter(self):
        (Qstring,ok) = QInputDialog.getText("Remove attribute", "Remove projections with attribute:")
        if ok:
            string = str(Qstring)
            i = 0
            while i < self.interestingList.count():
                list = str(self.interestingList.text(i))
                if list.find(string) >= 0:
                    self.interestingList.removeItem(i)
                    self.optimizedList.remove(self.optimizedList[i])
                else:
                    i += 1

    def removeSelected(self):
        for i in range(self.interestingList.count()-1, -1, -1):
            if self.interestingList.isSelected(i):
                self.interestingList.removeItem(i)

    def save(self):
        # get file name
        filename = "%s (k - %2d)" % (self.parentName, self.kValue )
        qname = QFileDialog.getSaveFileName( os.getcwd() + "/" + filename, "Interesting projections (*.proj)", self, "", "Save Projections")
        if qname.isEmpty():
            return
        name = str(qname)
        if name[-5] != ".":
            name = name + ".proj"

        # create array of strings from list box
        lista = []
        for i in range(self.interestingList.count()):
            lista.append(str(self.interestingList.text(i)))
            
        # open, write and save file
        file = open(name, "wt")
        cPickle.dump(self.optimizedList, file)
        cPickle.dump(lista, file)        
        file.flush()
        file.close()

    def load(self):
        self.optimizedList = []
        self.interestingList.clear()
        
        name = QFileDialog.getOpenFileName( os.getcwd(), "Interesting projections (*.proj)", self, "", "Open Projections")
        if name.isEmpty():
            return

        file = open(str(name), "rt")
        self.optimizedList = cPickle.load(file)
        lista = cPickle.load(file)
        file.close()

        for item in lista:
            self.interestingList.insertItem(item)
       