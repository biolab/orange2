import sys
import os
from OWFile import *
from OWMosaicDisplay import *


class mosaic(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Qt Orange Widgets Panes")
        self.bottom=QHBox(self)

        # create widget instances
        self.owFile = OWFile()
        self.owMosaic_Display = OWMosaicDisplay()
        

        # create widget buttons
        owButtonFile = QPushButton("File", self)
        owButtonMosaic_Display = QPushButton("Mosaic Display", self)
        exitButton = QPushButton("E&xit",self)
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))
        

        #connect GUI buttons to show widgets
        self.connect( owButtonFile,SIGNAL("clicked()"), self.owFile.show)
        self.connect( owButtonMosaic_Display,SIGNAL("clicked()"), self.owMosaic_Display.show)
        

        # add widget signals
        self.owMosaic_Display.link(self.owFile, "cdata")
        

    def exit(self):
        self.owFile.saveSettings()
        self.owMosaic_Display.saveSettings()
        


a=QApplication(sys.argv)
ow=mosaic()
a.setMainWidget(ow)
QObject.connect(a, SIGNAL('aboutToQuit()'),ow.exit) 
ow.show()
a.exec_loop()