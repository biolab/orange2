import sys
import os
from OWFile import *
from OW2DInteractions import *
from OWParallelCoordinates import *


class parallel(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Orange Widgets Panes")
        self.setIcon(QPixmap("OrangeWidgetsIcon.gif"))
        self.tabs = QTabWidget(self, 'tabWidget')
        self.bottom=QHBox(self)
        self.resize(640,480)
        exitButton=QPushButton("E&xit",self.bottom)

        # create widget instances
        self.owFile = OWFile(self.tabs)
        #self.ow2D_Interactions = OW2DInteractions(self.tabs)
        self.owParallel_coordinates = OWParallelCoordinates(self.tabs)
        

        # add tabs
        self.tabs.insertTab (self.owFile,"File")
        #self.tabs.insertTab (self.ow2D_Interactions,"2D Interactions")
        self.tabs.insertTab (self.owParallel_coordinates,"Parallel coordinates")
        

        #self.ow2D_Interactions.link(self.owFile, "cdata")
        self.owParallel_coordinates.link(self.owFile, "cdata")
        

    def exit(self):
        self.owParallel_coordinates.saveSettings()
        


a=QApplication(sys.argv)
ow=parallel()
a.setMainWidget(ow)
QObject.connect(a, SIGNAL('aboutToQuit()'),ow.exit)
ow.show()
a.exec_loop()