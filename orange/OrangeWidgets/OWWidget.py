#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#

from OWBaseWidget import *

class OWWidget(OWBaseWidget):
    def __init__( self, parent = None, signalManager = None, title = "Qt Orange Widget", wantGraph = FALSE, wantStatusBar = FALSE):
        """
        Initialization
        Parameters: 
            title - The title of the\ widget, including a "&" (for shortcut in about box)
            wantGraph - displays a save graph button or not
        """

        apply(OWBaseWidget.__init__, (self, parent, signalManager, title))

        self.mainArea=QWidget(self)
        self.controlArea=QVBox(self)
        self.buttonBackground=QVBox(self)
        self.space = self.controlArea
        #self.controlArea.setMaximumWidth(250)
        #self.space=QVBox(self)
        #self.grid=QGridLayout(self,2,2,5)
        self.grid=QGridLayout(self,3,2,5)
        self.grid.addWidget(self.controlArea,0,0)
        #self.grid.addWidget(self.space,1,0)
        self.grid.addWidget(self.buttonBackground,1,0)
        self.grid.setRowStretch(0,10)
        self.grid.setColStretch(0,10)
        self.grid.setColStretch(1,50)
        self.grid.addMultiCellWidget(self.mainArea,0,2,1,1)
        
        if wantGraph:    self.graphButton=QPushButton("&Save Graph",self.buttonBackground)

        if wantStatusBar:
            self.widgetStatusArea = QHBox(self)
            self.grid.addMultiCellWidget(self.widgetStatusArea, 3, 3, 0, 1)
            self.statusBar = QStatusBar(self.widgetStatusArea, )
            self.statusBar.setSizeGripEnabled(1)
            #self.statusBar.setWFlags( Qt.WStyle_DialogBorder)
            #self.statusBar.hide()
        
        self.resize(640,480)


    
if __name__ == "__main__":  
    a=QApplication(sys.argv)
    oww=OWWidget()
    a.setMainWidget(oww)
    oww.show()
    a.exec_loop()
    oww.saveSettings()
