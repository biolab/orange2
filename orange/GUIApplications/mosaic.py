import sys
import os
from orngSignalManager import *
from OWFile import *
from OWMosaicDisplay import *


class mosaic(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Qt mosaic")

        # create widget instances
        self.owFile = OWFile()
        self.owMosaic_Display = OWMosaicDisplay()
        self.owFile.progressBarSetHandler(self.progressHandler)
        self.owMosaic_Display.progressBarSetHandler(self.progressHandler)
        
        signalManager.addWidget(self.owFile)
        signalManager.addWidget(self.owMosaic_Display)
        
        # create widget buttons
        owButtonFile = QPushButton("File", self)
        owButtonMosaic_Display = QPushButton("Mosaic Display", self)
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
        self.connect( owButtonMosaic_Display,SIGNAL("clicked()"), self.owMosaic_Display.reshow)
        
        # add widget signals
        signalManager.setFreeze(1)
        signalManager.addLink( self.owFile, self.owMosaic_Display, 'Classified Examples', 'Classified Examples', 1)
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
        self.owMosaic_Display.saveSettings()
        


a=QApplication(sys.argv)
ow=mosaic()
a.setMainWidget(ow)
QObject.connect(a, SIGNAL('aboutToQuit()'),ow.exit) 
ow.show()
a.exec_loop()