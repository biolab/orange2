import sys
import os
from orngSignalManager import *
from OWFile import *
from OWPolyviz import *


class polyviz(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Qt polyviz")

        # create widget instances
        self.owFile = OWFile()
        self.owPolyviz = OWPolyviz()
        self.owFile.progressBarSetHandler(self.progressHandler)
        self.owPolyviz.progressBarSetHandler(self.progressHandler)
        
        signalManager.addWidget(self.owFile)
        signalManager.addWidget(self.owPolyviz)
        
        # create widget buttons
        owButtonFile = QPushButton("File", self)
        owButtonPolyviz = QPushButton("Polyviz", self)
        exitButton = QPushButton("E&xit",self)
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))
        

        statusBar = QStatusBar(self)
        self.caption = QLabel('', statusBar)
        self.caption.setMaximumWidth(200)
        self.caption.hide()
        self.progress = QProgressBar(100, statusBar)
        self.progress.setMaximumWidth(100)
        self.progress.hide()
        self.progress.setCenterIndicator(1)
        statusBar.addWidget(self.caption, 1)
        statusBar.addWidget(self.progress, 1)
        #connect GUI buttons to show widgets
        self.connect( owButtonFile,SIGNAL("clicked()"), self.owFile.reshow)
        self.connect( owButtonPolyviz,SIGNAL("clicked()"), self.owPolyviz.reshow)
        
        # add widget signals
        signalManager.setFreeze(1)
        signalManager.addLink( self.owFile, self.owPolyviz, 'Classified Examples', 'Classified Examples', 1)
        signalManager.setFreeze(0)
        


    def progressHandler(self, widget, val):
        if val < 0:
            self.caption.setText("<nobr>Processing: <b>" + str(widget.caption()) + "</b></nobr>")
            self.caption.show()
            self.progress.setProgress(0)
            self.progress.show()
        elif val >100:
            self.caption.hide()
            self.progress.hide()
        else:
            self.progress.setProgress(val)
            self.update()

    def exit(self):
        self.owFile.saveSettings()
        self.owPolyviz.saveSettings()
        


a=QApplication(sys.argv)
ow=polyviz()
a.setMainWidget(ow)
QObject.connect(a, SIGNAL('aboutToQuit()'),ow.exit) 
ow.show()
a.exec_loop()