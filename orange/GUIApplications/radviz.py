import sys
import os
from OWFile import *
from OWRadviz import *


class radviz(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Orange Widgets Panes")
        self.setIcon(QPixmap("OrangeWidgetsIcon.gif"))
        self.bottom=QHBox(self)

        # create widget instances
        self.owFile = OWFile()
        self.owRadviz = OWRadviz()
        

        # create widget buttons
        owButtonFile = QPushButton("File", self)
        owButtonRadviz = QPushButton("Radviz", self)
        exitButton = QPushButton("E&xit",self)
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))
        

        #connect GUI buttons to show widgets
        self.connect( owButtonFile,SIGNAL("clicked()"), self.owFile.show)
        self.connect( owButtonRadviz,SIGNAL("clicked()"), self.owRadviz.show)
        

        # add widget signals
        self.owRadviz.link(self.owFile, "cdata")
        

    def exit(self):
        self.owFile.saveSettings()
        self.owRadviz.saveSettings()
        


a=QApplication(sys.argv)
ow=radviz()
a.setMainWidget(ow)
QObject.connect(a, SIGNAL('aboutToQuit()'),ow.exit) 
ow.show()
a.exec_loop()