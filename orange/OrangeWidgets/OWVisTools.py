from OWBaseWidget import *
from qt import *
from qwt import *
import sys
import cPickle
import os

class OptimizationDialog(OWBaseWidget):
    settingsList = ["resultListLen"]
    resultsListLenList = ['10', '20', '50', '100', '150', '200', '250', '300', '400', '500', '700', '1000', '2000']
    resultsListLenNums = [ 10 ,  20 ,  50 ,  100 ,  150 ,  200 ,  250 ,  300 ,  400 ,  500 ,  700 ,  1000 ,  2000 ]

    def __init__(self,parent=None):
        #QWidget.__init__(self, parent)
        OWBaseWidget.__init__(self, parent, "Optimization Dialog", "optimize visualization impression and manage result", TRUE, FALSE, FALSE)

        self.setCaption("Qt Optimization Dialog")
        self.topLayout = QVBoxLayout( self, 10 ) 
        self.grid=QGridLayout(3,2)
        self.topLayout.addLayout( self.grid, 10 )

        self.kValue = 1
        self.resultListLen = 100
        self.widgetDir = sys.prefix + "/lib/site-packages/Orange/OrangeWidgets/"
        self.parentName = "Projection"
        #self.domainName = "Unknown"
        
        self.optimizedListFull = []
        self.optimizedListFiltered = []
        self.attrLenDict = {}

        self.loadSettings()
        
        self.optimizeButtonBox =QVGroupBox(self, "Optimize toolbox")
        self.optimizeButtonBox.setTitle("Optimize toolbox")
        
        self.manageResultsBox = QVGroupBox (self, "Manage results")
        self.manageResultsBox.setTitle("Manage results")

        self.infoBox =QVGroupBox(self, "Selected projection information")
        self.infoBox.setTitle("Information")
        
        self.resultsBox = QVGroupBox (self, "Results")
        self.resultsBox.setTitle("Results")

        self.grid.addWidget(self.optimizeButtonBox,0,0)
        self.grid.addWidget(self.manageResultsBox,1,0)
        self.grid.addWidget(self.infoBox, 2,0)
        self.grid.addMultiCellWidget (self.resultsBox,0,2, 1, 1)
        self.grid.setColStretch(0, 0)
        self.grid.setColStretch(1, 100)
        self.grid.setRowStretch(0, 0)
        self.grid.setRowStretch(1, 100)
        self.grid.setRowStretch(2, 0)
                
        self.interestingList = QListBox(self.resultsBox)
        #self.interestingList.setSelectionMode(QListBox.Extended)   # this would be nice if could be enabled, but it has a bug - currentItem doesn't return the correct value if this is on
        self.interestingList.setMinimumSize(200,200)

        self.hbox1 = QHBox(self.optimizeButtonBox)
        self.attrOrdLabel = QLabel('Number of neighbours (k):', self.hbox1)
        self.attrKNeighbour = QComboBox(self.hbox1)

        self.hbox2 = QHBox(self.optimizeButtonBox)
        self.resultListLabel = QLabel('Length of results list:', self.hbox2)
        self.resultListCombo = QComboBox(self.hbox2)
        for item in self.resultsListLenList:
            self.resultListCombo.insertItem(item)
        self.resultListCombo.setCurrentItem(self.resultsListLenNums.index(self.resultListLen))
        self.connect(self.resultListCombo, SIGNAL("activated(int)"), self.setResultListLen)
    
        self.optimizeSeparationButton = QPushButton('Optimize class separation', self.optimizeButtonBox)
        self.hbox3 = QHBox(self.optimizeButtonBox)
        self.optimizeAllSubsetSeparationButton = QPushButton('Optimize separation for subsets', self.hbox3)
        self.maxLenCombo = QComboBox(self.hbox3)    # maximum number of attributes in subset
        self.maxLenCombo.insertItem("ALL")
        for i in range(3, 15):
            self.maxLenCombo.insertItem(str(i))
        self.maxLenCombo.setCurrentItem(0)
        
        #self.resize(200, 500)
        self.attrLenCaption = QLabel('Select attribute count', self.manageResultsBox)
        self.attrLenList = QListBox(self.manageResultsBox)
        self.attrLenList.setSelectionMode(QListBox.Multi)
        self.attrLenList.setMinimumSize(60,60)
        self.connect(self.attrLenList, SIGNAL("selectionChanged()"), self.attrLenListChanged)
        self.filterButton = QPushButton("Remove attribute", self.manageResultsBox)
        self.removeSelectedButton = QPushButton("Remove selected projections", self.manageResultsBox)
        self.saveButton = QPushButton("Save", self.manageResultsBox)
        self.loadButton = QPushButton("Load", self.manageResultsBox)
        self.clearButton = QPushButton("Clear results", self.manageResultsBox)
        self.closeButton = QPushButton("Close", self.manageResultsBox)
        self.connect(self.filterButton, SIGNAL("clicked()"), self.filter)
        self.connect(self.removeSelectedButton, SIGNAL("clicked()"), self.removeSelected)
        self.connect(self.saveButton, SIGNAL("clicked()"), self.save)
        self.connect(self.loadButton, SIGNAL("clicked()"), self.load)
        self.connect(self.clearButton, SIGNAL("clicked()"), self.clear)
        self.connect(self.closeButton, SIGNAL("clicked()"), self.hide)

        #self.optimizeButtonBox.setMinimumSize(180,150)
        #self.manageResultsBox.setMinimumSize(180,150)

    def attrLenListChanged(self):
        self.interestingList.clear()
        self.optimizedListFiltered = []

        # check which attribute lengths do we want to show
        self.attrLenDict = {}
        for i in range(self.attrLenList.count()):
            intVal = int(str(self.attrLenList.text(i)))
            selected = self.attrLenList.isSelected(i)
            self.attrLenDict[intVal] = selected

        # show in results list only those results, that are the correct length        
        for i in range(len(self.optimizedListFull)):
            (accuracy, itemCount, list, strList) = self.optimizedListFull[i]
            if self.attrLenDict[len(list)] == 1:
                self.interestingList.insertItem("(%.2f, %d) - %s"%(accuracy, itemCount, strList))
                self.optimizedListFiltered.append((accuracy, itemCount, list, strList))
        self.interestingList.setCurrentItem(0)        

    def insertItem(self, accuracy, tableLen, list, strList = None):
        if strList == None:
            strList = list[0]
            for item in list[1:]:
                strList = strList + ", " + item

        for i in range(len(self.optimizedListFull)):
            (acc, iC, list2, strList2) = self.optimizedListFull[i]
            if acc < accuracy:
                self.optimizedListFull.insert(i, (accuracy, tableLen, list, strList))
                return
        
        self.optimizedListFull.append((accuracy, tableLen, list, strList))

    def updateNewResults(self):
        # update list of attribute lenghts
        self.attrLenList.clear()
        self.attrLenDict = {}
        found = []
        for i in range(len(self.optimizedListFull)):
            (acc, tableLen, list, strList) = self.optimizedListFull[i]
            if len(list) not in found:
                found.append(len(list))
        found.sort()
        for val in found:
            self.attrLenList.insertItem(str(val))
            self.attrLenDict[val] = 1
        self.attrLenList.selectAll(1)
        self.interestingList.setCurrentItem(0)

    def setResultListLen(self, n):
        self.resultListLen = self.resultsListLenNums[n]
        self.saveSettings()

    def clear(self):
        self.optimizedListFull = []
        self.optimizedListFiltered = []
        self.attrLenDict = {}        
        self.interestingList.clear()

    def filter(self):
        (Qstring,ok) = QInputDialog.getText("Remove attribute", "Remove projections with attribute:")
        if ok:
            attributeName = str(Qstring)
            for i in range(len(self.optimizedListFiltered)-1, -1, -1):
                (accuracy, itemCount, list, strList) = self.optimizedListFiltered[i]
                found = 0
                for val in list:
                    if val == attributeName: found = 1
                if found:
                    # remove from  listbox and original list of results
                    self.interestingList.removeItem(i)
                    self.optimizedListFull.remove((accuracy, itemCount, list, strList))

    def removeSelected(self):
        for i in range(self.interestingList.count()-1, -1, -1):
            if self.interestingList.isSelected(i):
                # remove from listbox and original list of results
                self.interestingList.removeItem(i)
                (accuracy, itemCount, list, strList) = self.optimizedListFiltered[i]
                self.optimizedListFiltered.remove((accuracy, itemCount, list, strList))
                self.optimizedListFull.remove((accuracy, itemCount, list, strList))

    def save(self):
        # get file name
        filename = "%s (k - %2d)" % (self.parentName, self.kValue )
        qname = QFileDialog.getSaveFileName( os.getcwd() + "/" + filename, "Interesting projections (*.proj)", self, "", "Save Projections")
        if qname.isEmpty():
            return
        name = str(qname)
        if name[-5] != ".":
            name = name + ".proj"

        # open, write and save file
        file = open(name, "wt")
        cPickle.dump(self.optimizedListFiltered, file)
        file.flush()
        file.close()

    def load(self):
        self.clear()
                
        name = QFileDialog.getOpenFileName( os.getcwd(), "Interesting projections (*.proj)", self, "", "Open Projections")
        if name.isEmpty():
            return

        file = open(str(name), "rt")
        self.optimizedListFull = cPickle.load(file)
        file.close()

        self.updateNewResults()