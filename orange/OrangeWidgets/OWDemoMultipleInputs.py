#
# OWDemo.py
#

import sys
from OWTools import *
from OWAboutX import *
from OWFile import *
from OWTestMultipleInput import *

class OWDemo(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setSpacing(0)
        self.setCaption("Qt Orange Widgets Demo")
        self.setIcon(QPixmap("OrangeWidgetsIcon.gif"))
        #GUI
        owf1Button=QPushButton("&File",self)
        owf2Button=QPushButton("&File",self)
        owmButton=QPushButton("MultiInputs",self)
        owaButton=QPushButton("&About",self)
        exitButton=QPushButton("E&xit",self)
        #Widgets
        self.owf1=OWFile()
        self.owf2=OWFile()
        
        self.owm=OWTestMultipleInput()
        self.owa=OWAboutX("OW &Demo",
        """
Orange Widgets Demo
is a simple demonstration
of how easy it is to use of Orange Widgets
        """
        )
        #make links between widgets
        self.owm.link(self.owf1,"data")
        self.owm.link(self.owf2,"data")
        
        #connect GUI buttons to show widgets
        self.connect(owf1Button,SIGNAL("clicked()"),self.owf1.show)
        self.connect(owf2Button,SIGNAL("clicked()"),self.owf2.show)
        self.connect(owmButton,SIGNAL("clicked()"),self.owm.show)
        
        #connect exit button to exit
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))
        

a=QApplication(sys.argv)
owd=OWDemo()
a.setMainWidget(owd)
#QObject.connect(a, SIGNAL('lastWindowClosed()'),owd.exit) 
owd.show()
a.exec_loop()
        