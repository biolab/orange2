import sys
import os
from OWFile import *
from OWScatterPlot import *


class scatter(QVBox):
    def __init__(self, app, parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Qt Orange Widgets Panes")
        self.bottom=QHBox(self)
        self.application = app

        # create widget instances
        self.owFile = OWFile()
        self.owScatterplot = OWScatterPlot(app = self.application)
        

        # create widget buttons
        owButtonFile = QPushButton("File", self)
        owButtonScatterplot = QPushButton("Scatterplot", self)
        exitButton = QPushButton("E&xit",self)
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))
        

        #connect GUI buttons to show widgets
        self.connect( owButtonFile,SIGNAL("clicked()"), self.owFile.show)
        self.connect( owButtonScatterplot,SIGNAL("clicked()"), self.owScatterplot.show)
        

        # add widget signals
        self.owScatterplot.link(self.owFile, "cdata")
        

    def exit(self):
        self.owFile.saveSettings()
        self.owScatterplot.saveSettings()
        


a=QApplication(sys.argv)
ow=scatter(a)
a.setMainWidget(ow)
QObject.connect(a, SIGNAL('aboutToQuit()'),ow.exit) 
ow.show()
a.exec_loop()