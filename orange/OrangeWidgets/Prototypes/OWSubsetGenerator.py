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
    settingsList=["applyGenerateExact"]
    def __init__(self,parent=None):
        OWWidget.__init__(self,parent,"&SubsetGenerator",
        "Create exaple table subset",
        wantSettings = FALSE, wantGraph = FALSE)
        "Constructor"

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1)]
        self.outputs = [("Classified Examples", ExampleTableWithClass)] 

        #get settings from the ini file, if they exist
        self.applyGenerateExact = 1 # we remember the last kind of generate
        self.data = None
        self.loadSettings()        
        
        #GUI
        self.space.hide()
        self.controlArea.hide()
        self.BigHbox1 = QHBox(self)
        self.dummyLabel1 = QLabel("  ", self.BigHbox1)
        self.percentRB = QRadioButton("  ", self.BigHbox1)
        self.PercentBox = QVGroupBox("Percent of data", self.BigHbox1)
        self.hbox1 = QHBox(self.PercentBox, "percent")
        self.percentSlider = QSlider(1, 100, 10, 50, QSlider.Horizontal, self.hbox1)
        self.percentSlider.setTickmarks(QSlider.Below)
        self.percentLCD = QLCDNumber(3, self.hbox1)
        QObject.connect(self.percentSlider, SIGNAL("valueChanged(int)"), self.percentLCD, SLOT("display(int)"))
        self.percentLCD.display(50)
                
        self.BigHbox2 = QHBox(self)
        self.dummyLabel2 = QLabel("  ", self.BigHbox2)
        self.exactRB = QRadioButton("  ", self.BigHbox2)
        self.countBox = QVGroupBox("Exact table length", self.BigHbox2)
        self.hbox2 = QHBox(self.countBox, "exact")
        self.exactCaption = QLabel('Number of examples: ', self.hbox2)
        self.exactEdit = QLineEdit(self.hbox2)

        self.exactEdit.setText("100")
        self.generateButton = QPushButton('Generate', self)

        self.connect(self.percentRB, SIGNAL("toggled(bool)"), self.percentRBActivated)
        self.connect(self.exactRB, SIGNAL("toggled(bool)"), self.exactRBActivated)
        self.connect(self.generateButton, SIGNAL("clicked()"), self.generate)


        #self.grid.addWidget(self.PercentBox, 0,0)
        self.grid.addWidget(self.BigHbox1, 0,0)
        self.grid.addWidget(self.BigHbox2, 1,0)
        self.grid.addWidget(self.generateButton, 2,0)
        
        self.resize(200,200)
        self.activateLoadedSettings()

    def percentRBActivated(self, b):
        self.percentRB.setChecked(b)
        self.exactRB.setChecked(not b)
        #self.generate()

    def exactRBActivated(self, b):
        self.percentRB.setChecked(not b)
        self.exactRB.setChecked(b)
        #self.generate()

    def generate(self):
        self.applyGenerateExact = self.exactRB.isChecked()
        if self.data == None: return

        if self.applyGenerateExact: self.generateExact()
        else:                       self.generatePercent()


    def generateExact(self):
        dataLen = float(str(self.exactEdit.text()))
        
        selection = orange.MakeRandomIndices2(self.data, 1.0-float(dataLen/len(self.data)))
        table = orange.ExampleTable(self.data.domain)
        for i in range(len(self.data)):
            if selection[i] == 0: continue
            table.append(self.data[i])

        self.send("Classified Examples", table)

    def generatePercent(self):
        dataLen = float(self.percentLCD.intValue())
        
        selection = orange.MakeRandomIndices2(self.data, 1.0-float(dataLen/100.0))
        table = orange.ExampleTable(self.data.domain)
        for i in range(len(self.data)):
            if selection[i] == 0: continue
            table.append(self.data[i])

        self.send("Classified Examples", table)

    def cdata(self, data):
        self.data = data
        self.generate()


    def activateLoadedSettings(self):
        self.exactRB.setChecked(self.applyGenerateExact)
        self.percentRB.setChecked(not self.applyGenerateExact)
        
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWSubsetGenerator()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()
