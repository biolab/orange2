#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#

import sys
from OWBaseWidget import *
import ConfigParser,os
from string import *
import cPickle
from OWTools import *

class OWWidget(OWBaseWidget):
    def __init__(
    self,
    parent=None,
    title="Qt Orange Widget",
    description="This a general Orange Widget\n from which all the other Orange Widgets are derived.",
    wantSettings=FALSE,
    wantGraph=FALSE,
    wantAbout=FALSE,
    icon="OrangeWidgetsIcon.png",
    logo="OrangeWidgetsLogo.png",
    ):
        """
        Initialization
        Parameters: 
            title - The title of the\ widget, including a "&" (for shortcut in about box)
            description - self explanatory
            wantSettings - display settings button or not
            wantGraph - displays a save graph button or not
            icon - the icon file
            logo - the logo file
            parent - parent of the widget if needed
        """

        apply(OWBaseWidget.__init__, (self, parent, title, description, wantSettings, wantGraph, wantAbout, icon, logo))

        self.mainArea=QWidget(self)
        self.controlArea=QVBox(self)
        self.space = self.controlArea
        #self.controlArea.setMaximumWidth(250)
        #self.space=QVBox(self)
        self.grid=QGridLayout(self,2,2,5)
        self.grid.addWidget(self.controlArea,0,0)
        #self.grid.addWidget(self.space,1,0)
        self.grid.addWidget(self.buttonBackground,1,0)
        self.grid.setRowStretch(0,10)
        self.grid.setColStretch(0,10)
        self.grid.setColStretch(1,30)
        self.grid.addMultiCellWidget(self.mainArea,0,2,1,1)
        self.resize(640,480)


    
if __name__ == "__main__":  
    a=QApplication(sys.argv)
    oww=OWWidget()
    a.setMainWidget(oww)
    oww.show()
    a.exec_loop()
    oww.saveSettings()
