import sys
from OWTools import *
from OWAboutX import *
from OWFile import *
from OWOutcome import *
from OWRank import *
from OWClassificationTree import *
from OWClassificationTreeViewer import *

class OWDemo(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setSpacing(0)
        self.setCaption("Orange Widgets Demo")
        self.setIcon(QPixmap("OrangeWidgetsIcon.gif"))
        #GUI
        owfButton=QPushButton("&File",self)
        owwButton=QPushButton("&Outcome",self)
        owrButton=QPushButton("Rank",self)
        owtButton=QPushButton("Tree",self)
        owvButton=QPushButton("View Tree",self)
        owaButton=QPushButton("&About",self)
        exitButton=QPushButton("E&xit",self)
        #Widgets
        self.owf=OWFile()
        self.owo=OWOutcome()
        self.owr=OWRank()
        self.owt=OWClassificationTree()
        self.owv=OWClassificationTreeViewer()
        self.owa=OWAboutX("OW &Demo",
        """
Orange Widgets Demo
is a simple demonstration
of how easy it is to use of Orange Widgets
        """
        )
        #make links between widgets
        self.owo.link(self.owf, "data")
        self.owt.link(self.owo, "cdata")
        self.owv.link(self.owt, "ctree")
        self.owv.link(self.owo, "target")
        self.owr.link(self.owo, "cdata")
        
        #connect GUI buttons to show widgets
        self.connect(owfButton,SIGNAL("clicked()"),self.owf.show)
        self.connect(owwButton,SIGNAL("clicked()"),self.owo.show)
        self.connect(owrButton,SIGNAL("clicked()"),self.owr.show)
        self.connect(owtButton,SIGNAL("clicked()"),self.owt.show)
        self.connect(owvButton,SIGNAL("clicked()"),self.owv.show)
        self.connect(owaButton,SIGNAL("clicked()"),self.owa.show)        
        
        #connect exit button to exit
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))
#        self.connect(self,SIGNAL("exit"),self.exit)
 
    def exit(self):
        print "exit"
        self.owf.saveSettings()
        self.owo.saveSettings()
        self.owt.saveSettings()
        self.owv.saveSettings()
        self.owr.saveSettings()
    

a = QApplication(sys.argv)
owd = OWDemo()
a.setMainWidget(owd)
#QObject.connect(a, SIGNAL('lastWindowClosed()'),owd.exit) 
owd.show()
a.exec_loop()
owd.exit()