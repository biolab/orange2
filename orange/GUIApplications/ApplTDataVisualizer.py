#
# OWDemo.py
#

import sys
from OWTools import *
from OWAboutX import *
from OWFile import *
from OWOutcome import *
from OWDistributions import *
from OW2DInteractions import *
from OWRank import *

class OWDemo(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setSpacing(0)
        self.setCaption("Orange Widgets Demo")
        self.setIcon(QPixmap("OrangeWidgetsIcon.gif"))
        #GUI
        owfButton=QPushButton("&File",self)
        owwButton=QPushButton("&Outcome",self)
        owdButton=QPushButton("&Distributions",self)
        owiButton=QPushButton("2D&Interactions",self)
        owrButton=QPushButton("Rank",self)
        owaButton=QPushButton("&About",self)
        exitButton=QPushButton("E&xit",self)
        #Widgets
        self.owf=OWFile()
        self.owo=OWOutcome()
        self.owd=OWDistributions()
        self.owi=OW2DInteractions()
        self.owr=OWRank()
        self.owa=OWAboutX("OW &Demo",
        """
Orange Widgets Demo
is a simple demonstration
of how easy it is to use of Orange Widgets
        """
        )
        #make links between widgets
        self.owo.link(self.owf,"data")
        self.owd.link(self.owo,"cdata")
        self.owi.link(self.owo,"cdata")
        self.owr.link(self.owo,"cdata")
        
        #connect GUI buttons to show widgets
        self.connect(owfButton,SIGNAL("clicked()"),self.owf.show)
        self.connect(owwButton,SIGNAL("clicked()"),self.owo.show)
        self.connect(owdButton,SIGNAL("clicked()"),self.owd.show)
        self.connect(owiButton,SIGNAL("clicked()"),self.owi.show)
        self.connect(owrButton,SIGNAL("clicked()"),self.owr.show)
        self.connect(owaButton,SIGNAL("clicked()"),self.owa.show)        

        #connect exit button to exit
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))

    def exit(self):
        self.owf.saveSettings()
        self.owd.saveSettings()
        self.owi.saveSettings()


a=QApplication(sys.argv)
owd=OWDemo()
a.setMainWidget(owd)
QObject.connect(a, SIGNAL('aboutToQuit()'),owd.exit) 
owd.show()
a.exec_loop()
