import sys
import os
from OWFile import *
from OWSieveDiagram import *


class sieve(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Qt Orange Widgets Panes")
        self.bottom=QHBox(self)

        # create widget instances
        self.owFile = OWFile()
        self.owSieve_Diagram = OWSieveDiagram()
        

        # create widget buttons
        owButtonFile = QPushButton("File", self)
        owButtonSieve_Diagram = QPushButton("Sieve Diagram", self)
        exitButton = QPushButton("E&xit",self)
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))
        

        #connect GUI buttons to show widgets
        self.connect( owButtonFile,SIGNAL("clicked()"), self.owFile.show)
        self.connect( owButtonSieve_Diagram,SIGNAL("clicked()"), self.owSieve_Diagram.show)
        

        # add widget signals
        self.owSieve_Diagram.link(self.owFile, "cdata")
        

    def exit(self):
        self.owFile.saveSettings()
        self.owSieve_Diagram.saveSettings()
        


a=QApplication(sys.argv)
ow=sieve()
a.setMainWidget(ow)
QObject.connect(a, SIGNAL('aboutToQuit()'),ow.exit) 
ow.show()
a.exec_loop()