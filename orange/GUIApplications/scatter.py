import sys, os, cPickle, orange
from orngSignalManager import *

widgetDir = os.path.join(os.path.split(orange.__file__)[0], "OrangeWidgets")
if os.path.exists(widgetDir):
        for name in os.listdir(widgetDir):
            fullName = os.path.join(widgetDir, name)
            if os.path.isdir(fullName): sys.path.append(fullName)

from OWFile import *
from OWScatterPlot import *


class Schema_1(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Qt scatter")
        self.tabs = QTabWidget(self, 'tabWidget')
        self.resize(800,600)

        # create widget instances
        self.owFile = OWFile (self.tabs)
        self.owScatterplot = OWScatterPlot (self.tabs)
        
        # create instances of hidden widgets
        self.owFile.progressBarSetHandler(self.progressHandler)
        self.owScatterplot.progressBarSetHandler(self.progressHandler)
        
        
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
        signalManager.addWidget(self.owFile)
        signalManager.addWidget(self.owScatterplot)
        
        # add tabs
        self.tabs.insertTab (self.owFile, "File")
        self.tabs.insertTab (self.owScatterplot, "Scatterplot")
        
        #load settings before we connect widgets
        self.loadSettings()

        # add widget signals
        signalManager.setFreeze(1)
        signalManager.addLink( self.owFile, self.owScatterplot, 'Examples', 'Examples', 1)
        signalManager.setFreeze(0)
        

    def progressHandler(self, widget, val):
        if val < 0:
            self.caption.setText("<nobr>Processing: <b>" + str(widget.captionTitle) + "</b></nobr>")
            self.caption.show()
            self.progress.setProgress(0)
            self.progress.show()
        elif val >100:
            self.caption.hide()
            self.progress.hide()
        else:
            self.progress.setProgress(val)
            self.update()


        
    def loadSettings(self):
        try:
            file = open("scatter.sav", "r")
        except:
            return
        strSettings = cPickle.load(file)
        file.close()
        self.owFile.loadSettingsStr(strSettings["File"])
        self.owFile.activateLoadedSettings()
        self.owScatterplot.loadSettingsStr(strSettings["Scatterplot"])
        self.owScatterplot.activateLoadedSettings()
        
        
    def saveSettings(self):
        strSettings = {}
        strSettings["File"] = self.owFile.saveSettingsStr()
        strSettings["Scatterplot"] = self.owScatterplot.saveSettingsStr()
        
        file = open("scatter.sav", "w")
        cPickle.dump(strSettings, file)
        file.close()
        


application = QApplication(sys.argv)
ow = Schema_1()
application.setMainWidget(ow)
ow.loadSettings()
ow.show()
application.exec_loop()
ow.saveSettings()
