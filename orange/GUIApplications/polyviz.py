import sys
import os
from OWFile import *
from OWPolyviz import *
import win32process, win32api

class polyviz(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Qt Orange Widgets Panes")
        self.bottom=QHBox(self)

        # create widget instances
        self.owFile = OWFile()
        self.owPolyviz = OWPolyviz()
        

        # create widget buttons
        owButtonFile = QPushButton("File", self)
        owButtonPolyviz = QPushButton("Polyviz", self)
        exitButton = QPushButton("E&xit",self)
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))
        

        #connect GUI buttons to show widgets
        self.connect( owButtonFile,SIGNAL("clicked()"), self.owFile.show)
        self.connect( owButtonPolyviz,SIGNAL("clicked()"), self.owPolyviz.show)
        

        # add widget signals
        self.owPolyviz.link(self.owFile, "cdata")
        

    def exit(self):
        self.owFile.saveSettings()
        self.owPolyviz.saveSettings()
        

win32process.SetPriorityClass(win32api.GetCurrentProcess(), 64)
a=QApplication(sys.argv)
ow=polyviz()
a.setMainWidget(ow)
QObject.connect(a, SIGNAL('aboutToQuit()'),ow.exit) 
ow.show()
a.exec_loop()