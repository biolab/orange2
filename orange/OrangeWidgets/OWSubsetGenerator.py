"""
<name>Subset Generator</name>
<description>Generate a subset of input example table.</description>
<category>Miscelaneous</category>
<icon>icons/SubsetGen.png</icon>
<priority>30</priority>
"""

#
# OWSubsetGenerator.py
#

from OWWidget import *
from OData import *
from random import *

class OWSubsetGenerator(OWWidget):
    settingsList=[]
    def __init__(self,parent=None):
        OWWidget.__init__(self,parent,"&SubsetGenerator",
        "Create exaple table subset",
        FALSE)
        "Constructor"
        #get settings from the ini file, if they exist
        self.loadSettings()

        self.data = None
        self.addInput("data")
        self.addInput("cdata")
        self.addOutput("data")
        self.addOutput("cdata")
        
        #GUI
        self.space.hide()
        self.controlArea.hide()
        #self.grid.deleteAllItems()
        #self.box = QVGroupBox(self)
        self.PercentBox = QVGroupBox("Percent of data", self)
        self.hbox1 = QHBox(self.PercentBox, "percent")
        self.percentSlider = QSlider(1, 100, 10, 50, QSlider.Horizontal, self.hbox1)
        self.percentSlider.setTickmarks(QSlider.Below)
        self.percentLCD = QLCDNumber(3, self.hbox1)
        self.connect(self.percentSlider, SIGNAL("valueChanged(int)"), self.percentLCD, SLOT("display(int)"))
        self.percentLCD.display(50)
        self.generatePercentButton = QPushButton('Generate', self.hbox1)
        self.connect(self.generatePercentButton, SIGNAL("clicked()"), self.generatePercent)

        
        self.countBox = QVGroupBox("Exact table length", self)
        self.hbox2 = QHBox(self.countBox, "exact")
        self.exactCaption = QLabel('Number of examples: ', self.hbox2)
        self.exactEdit = QLineEdit(self.hbox2)
        self.generateExactButton = QPushButton('Generate', self.hbox2)
        self.exactEdit.setText("100")
        self.connect(self.generateExactButton, SIGNAL("clicked()"), self.generateExact)

        self.grid.addWidget(self.PercentBox, 0,0)
        self.grid.addWidget(self.countBox, 1,0)
        self.resize(300,100)



    def generateExact(self):
        if self.data == None: return
        dataLen = float(str(self.exactEdit.text()))
        
        selection = orange.MakeRandomIndices2(self.data, 1.0-float(dataLen/len(self.data)))
        table = orange.ExampleTable(self.data.domain)
        for i in range(len(self.data)):
            if selection[i] == 0: continue
            table.append(self.data[i])

        odata = OrangeData(table)
        self.send("cdata", odata)
        self.send("data", odata)

    def generatePercent(self):
        if self.data == None: return
        dataLen = float(self.percentLCD.intValue())
        
        selection = orange.MakeRandomIndices2(self.data, 1.0-float(dataLen/100.0))
        table = orange.ExampleTable(self.data.domain)
        for i in range(len(self.data)):
            if selection[i] == 0: continue
            table.append(self.data[i])

        odata = OrangeData(table)
        self.send("cdata", odata)
        self.send("data", odata)

    def cdata(self, data):
        self.data = data.data


    def saveSettings(self, file = None):
        pass
    
    def activateLoadedSettings(self):
        pass
        
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWSubsetGenerator()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()
