"""
<name>Example Builder</name>
<description>Create example table.</description>
<category>Other</category>
<icon>icons/EBuilder.png</icon>
<priority>2000</priority>
"""

#
# OWExampleBuilder.py
# The Example builder
#

from OWWidget import *
from random import *

class OWExampleBuilder(OWWidget):
    settingsList=["codeStr", "domainStr", "numStr"]
    
    def __init__(self,parent=None):
        OWWidget.__init__(self,parent,"&ExampleBuilder",
        "Builds example table",
        FALSE)
        "Constructor"
        #get settings from the ini file, if they exist

        self.inputs = []
        self.outputs = [("Classified Examples", ExampleTableWithClass)]
    
        self.codeStr = ""
        self.domainStr = ""
        self.numStr = ""
        self.loadSettings()

        self.data = None
                
        #GUI
        self.space.hide()
        self.controlArea.hide()
        #self.grid.deleteAllItems()
        #self.box = QVGroupBox(self)
        self.num = QLineEdit(self)
        self.domain = QMultiLineEdit(self)
        self.domain.setFont(QFont( "courier", 12, QFont.Bold ))
        self.textBox = QMultiLineEdit(self)
        self.textBox.setFont(QFont( "courier", 12, QFont.Bold ))
        self.buildButtion = QPushButton("Generate", self)
        self.saveButton = QPushButton("Save", self)
        self.closeButton = QPushButton("Close", self)
        self.grid.addMultiCellWidget(self.num, 0,0,0,1)
        self.grid.addMultiCellWidget(self.domain, 1,6,0,1)
        self.grid.addMultiCellWidget(self.textBox, 7,13,0,1)
        self.grid.addMultiCellWidget(self.buildButtion, 14,14,0,1)
        self.grid.addMultiCellWidget(self.saveButton, 15,15,0,1)
        self.grid.addMultiCellWidget(self.closeButton, 16,16,0,1)
        self.grid.setRowStretch(0,20)
        self.grid.setColStretch(0,20)
        self.resize(500,300)
        self.connect(self.buildButtion,SIGNAL("clicked()"),self.build)
        self.connect(self.saveButton,SIGNAL("clicked()"),self.save)
        self.connect(self.closeButton,SIGNAL("clicked()"),self.close)

        self.activateLoadedSettings()

    def build(self):
        domainList = str(self.domain.text()).split("\n")
        varList = []
        execList = "["
        randData = []
        for item in domainList:
            list = item.split(",")
            name = list[0]
            execList = execList + name + ","
            type = list[1]
            if list[2].find("..") >= 0:
                domain = []
                dom = list[2].split("..")
                if type == "d":
                    for i in range(int(dom[0]), int(dom[1])+1):
                        domain.append(str(i))
                else:
                    domain = dom
                randData.append((name, domain, type))
            else:
                print "error in text"
                return

            if type == "d":
                var = orange.EnumVariable(name = str(name), values = domain)
            elif type == "c":
                var = orange.FloatVariable(name = str(name), startValue = int(domain[0]), endValue= int(domain[1]))
            else:
                continue
            varList.append(var)
        domain = orange.Domain(varList)
        self.data = orange.ExampleTable(domain)
        execList = execList[:-1] + "]"

        randCode = []
        for (name, dom, type) in randData:
            if type == "d":
                code = compile("%s = randint(%i, %i)" %(name, int(dom[0]), int(dom[len(dom)-1])), ".", "single")
            else:
                diff = float(dom[len(dom)-1]) - float(dom[0])
                code = compile("%s = random()* %f + %f" %(name, diff, int(dom[0])), ".", "single")
            randCode.append(code)

        code = compile(str(self.textBox.text()), ".", "exec")
        count = int(str(self.num.text()))
        while len(self.data) < count:
            # get new random data values
            for c in randCode: exec(c)

            validExample = 1
            exec(code)
            exampleCode = compile("example = orange.Example(domain," + execList + ")", ".", "single")
            exec(exampleCode)
            if validExample:
                self.data.append(example)

        self.send("Classified Examples", self.data)
        
    def save(self):
        qname = QFileDialog.getSaveFileName( os.getcwd() + "/" + "data.tab", "Tabulated data (*.tab)", self, "", "Save Data Table")
        if qname.isEmpty():
            return
        name = str(qname)
        if name[-4] != ".":
            name = name + ".tab"
        orange.saveTabDelimited(name, self.data)

    def saveSettingsStr(self):
        self.numStr = str(self.num.text())
        self.domainStr = str(self.domain.text())
        self.codeStr = str(self.textBox.text())
        return OWWidget.saveSettingsStr(self)
        
    def saveSettings(self, file = None):
        self.numStr = str(self.num.text())
        self.domainStr = str(self.domain.text())
        self.codeStr = str(self.textBox.text())
        return OWWidget.saveSettings(self, file)

    def activateLoadedSettings(self):
        self.num.setText(self.numStr)
        self.domain.setText(self.domainStr)
        self.textBox.setText(self.codeStr)
        
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWExampleBuilder()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()
