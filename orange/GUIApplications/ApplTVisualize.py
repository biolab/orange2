import sys
import os
from OWFile import *
from OWScatterPlot import *
from OWParallelCoordinates import *
from OWSurveyPlot import *
from OWRadviz import *


class TVisualize(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Orange Widgets Panes")
        self.bottom=QHBox(self)
        self.tabs = QTabWidget(self, 'tabWidget')
        self.resize(640,480)

        # create widget instances
        self.owFile = OWFile(self.tabs)
        self.owScatterplot = OWScatterPlot(self.tabs)
        self.owParallel_coordinates = OWParallelCoordinates(self.tabs)
        self.owSurvey_Plot = OWSurveyPlot(self.tabs)
        self.owRadviz = OWRadviz(self.tabs)
        

        # add tabs
        self.tabs.insertTab (self.owFile,"File")
        self.tabs.insertTab (self.owScatterplot,"Scatterplot")
        self.tabs.insertTab (self.owParallel_coordinates,"Parallel coordinates")
        self.tabs.insertTab (self.owSurvey_Plot,"Survey Plot")
        self.tabs.insertTab (self.owRadviz,"Radviz")
        

        # add widget signals
        self.owParallel_coordinates.link(self.owFile, "cdata")
        self.owSurvey_Plot.link(self.owFile, "cdata")
        self.owRadviz.link(self.owFile, "cdata")
        self.owScatterplot.link(self.owFile, "cdata")
        
        

    def exit(self):
        self.owFile.saveSettings()
        self.owScatterplot.saveSettings()
        self.owParallel_coordinates.saveSettings()
        self.owSurvey_Plot.saveSettings()
        self.owRadviz.saveSettings()
        


a=QApplication(sys.argv)
ow=TVisualize()
a.setMainWidget(ow)
QObject.connect(a, SIGNAL('aboutToQuit()'),ow.exit) 
ow.show()
a.exec_loop()