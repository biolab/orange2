"""
<name>Python Script</name>
<description>Executes python script.</description>
<icon>icons/PythonScript.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>3011</priority>
"""
from OWWidget import *

import sys, traceback
import OWGUI, orngNetwork

class OWPythonScript(OWWidget):
    
    settingsList = ["codeFile"] 
                    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Python Script')
        
        self.inputs = [("inExampleTable", ExampleTable, self.setExampleTable), ("inDistanceMatrix", orange.SymMatrix, self.setDistanceMatrix), ("inNetwork", orngNetwork.Network, self.setNetwork)]
        self.outputs = [("outExampleTable", ExampleTable), ("outDistanceMatrix", orange.SymMatrix), ("outNetwork", orngNetwork.Network)]
        
        self.inNetwork = None
        self.inExampleTable = None
        self.inDistanceMatrix = None
        self.codeFile = ''
        
        self.loadSettings()
        
        self.infoBox = OWGUI.widgetBox(self.controlArea, 'Info')
        OWGUI.label(self.infoBox, self, "Execute python script.\n\nInput variables:\n - inExampleTable\n - inDistanceMatrix\n - inNetwork\n\nOutput variables:\n - outExampleTable\n - outDistanceMatrix\n - outNetwork")
        
        self.controlBox = OWGUI.widgetBox(self.controlArea, 'File')
        OWGUI.button(self.controlBox, self, "Open...", callback=self.openScript)
        OWGUI.button(self.controlBox, self, "Save...", callback=self.saveScript)
        
        self.runBox = OWGUI.widgetBox(self.controlArea, 'Run')
        OWGUI.button(self.runBox, self, "Execute", callback=self.execute)
        
        self.splitCanvas = QSplitter(Qt.Vertical, self.mainArea)
        self.mainArea.layout().addWidget(self.splitCanvas)
        
        self.textBox = OWGUI.widgetBox(self, 'Python script')
        self.splitCanvas.addWidget(self.textBox)
        self.text = QPlainTextEdit(self)
        self.textBox.layout().addWidget(self.text)
        self.text.setFont(QFont("Monospace"))
        self.textBox.setAlignment(Qt.AlignVCenter)
        self.text.setTabStopWidth(4)
        
        self.consoleBox = OWGUI.widgetBox(self, 'Console')
        self.splitCanvas.addWidget(self.consoleBox)
        self.console = QPlainTextEdit(self)
        self.consoleBox.layout().addWidget(self.console)
        self.console.setFont(QFont("Monospace"))
        self.consoleBox.setAlignment(Qt.AlignBottom)
        self.console.setTabStopWidth(4)
        
        self.openScript(self.codeFile)
        
        self.controlArea.layout().addStretch(1)
        self.resize(800,600)
        
    def setExampleTable(self, et):
        self.inExampleTable = et
        
    def setDistanceMatrix(self, dm):
        self.inDistanceMatrix = dm
        
    def setNetwork(self, net):
        self.inNetwork = net
    
    def openScript(self, filename=None):
        if filename == None:
            self.codeFile = str(QFileDialog.getOpenFileName(self, 'Open Python Script', self.codeFile, 'Python files (*.py)\nAll files(*.*)'))    
        else:
            self.codeFile = filename
            
        if self.codeFile == "": return
            
        f = open(self.codeFile, 'r')
        self.text.setPlainText(f.read())
        f.close()
    
    def saveScript(self):
        self.codeFile = QFileDialog.getSaveFileName(self, 'Save Python Script', self.codeFile, 'Python files (*.py)\nAll files(*.*)')
        
        if self.codeFile:
            fn = ""
            head, tail = os.path.splitext(str(self.codeFile))
            if not tail:
                fn = head + ".py"
            else:
                fn = str(self.codeFile)
            
            f = open(fn, 'w')
            f.write(self.text.toPlainText())
            f.close()
    
    def execute(self):
        self.console.setPlainText('')

        try:
            code = self.text.toPlainText()
            inExampleTable = self.inExampleTable
            inDistanceMatrix = self.inDistanceMatrix
            inNetwork = self.inNetwork
            
            outExampleTable = None
            outDistanceMatrix = None
            outNetwork = None
            
            exec(str(code))
            
            self.send("outExampleTable", outExampleTable)
            self.send("outDistanceMatrix", outDistanceMatrix)
            self.send("outNetwork", outNetwork)

        except:
            message = str(sys.exc_info()[0]) + "\n"
            message += str(sys.exc_info()[1]) + "\n"
            message += "LINE=" + str(traceback.tb_lineno(sys.exc_info()[2])) + "\n"
            self.console.setPlainText(message)

if __name__=="__main__":    
    appl = QApplication(sys.argv)
    ow = OWPythonScript()
    ow.show()
    appl.exec_()