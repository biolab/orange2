#
# OWPanes.py
#

import sys
from OWTools import *
from OWAboutX import *
from OWFile import *
from OWOutcome import *
from OWDistributions import *
from OW2DInteractions import *
from OWRank import *

class OWPanes(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Qt Orange Widgets Panes")
        self.setIcon(QPixmap("OrangeWidgetsIcon.gif"))
        self.tabs = QTabWidget(self, 'tabWidget')
        self.bottom=QHBox(self)
        #GUI
        owaButton=QPushButton("&About Orange Panes",self.bottom)
        exitButton=QPushButton("E&xit",self.bottom)
        #Widgets
        self.owf=OWFile(self.tabs)
        self.owo=OWOutcome(self.tabs)
        self.owd=OWDistributions(self.tabs)
        self.owi=OW2DInteractions(self.tabs)
        self.owr=OWRank(self.tabs)
 
        #the tabs
        self.tabs.insertTab (self.owf,'&File')
        self.tabs.insertTab (self.owo,'&Outcome')
        self.tabs.insertTab (self.owd,'&Distributions')
        self.tabs.insertTab (self.owi,'2D &Interactions')
        self.tabs.insertTab (self.owr,'&Rank')

        self.resize(640,480)
        
        self.owa=OWAboutX("OW &Panes",
        """
Orange Widgets Panes 
is an application based on Orange Widgets
that shows widgets in a tabbed form.
        """
        )
        #make links between widgets
        self.owo.link(self.owf,"data")
        self.owd.link(self.owo,"cdata")
        self.owi.link(self.owo,"cdata")
        self.owr.link(self.owo,"cdata")
        
        #connect GUI buttons to show widgets
        self.connect(owaButton,SIGNAL("clicked()"),self.owa.show)        
        
        #connect exit button to save options and to exit
        self.connect(exitButton,SIGNAL("clicked()"),self.exit)
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))
        
    def exit(self):
        self.owf.saveSettings()
        self.owd.saveSettings()
        self.owi.saveSettings()        
    

a=QApplication(sys.argv)
owp=OWPanes()
a.setMainWidget(owp)
QObject.connect(a, SIGNAL('lastWindowClosed()'),owp.exit) 
owp.show()
a.exec_loop()
        