#
# OWPanes.py
#

import sys
from OWTools import *
from OWAboutX import *
from OWFile import *
from OWOutcome import *
from OWDistributions import *
from OWRank import *
from OWNaiveBayes import *
from OW2DMisclassified import *

class OWPanes(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
        self.setCaption("Orange Widgets Panes")
        self.setIcon(QPixmap("OrangeWidgetsIcon.gif"))
        self.tabs = QTabWidget(self, 'tabWidget')
        self.bottom=QHBox(self)
        #GUI
        owaButton=QPushButton("&About Orange Panes",self.bottom)
        exitButton=QPushButton("E&xit",self.bottom)
        #Widgets
        self.owf=OWFile(self.tabs)
        self.owo=OWOutcome(self.tabs)
        self.owNB=OWNaiveBayes(self.tabs)
        self.owm=OW2DMisclassified(self.tabs)
        
        #the tabs
        self.tabs.insertTab (self.owf,'&File')
        self.tabs.insertTab (self.owo,'&Outcome')
        self.tabs.insertTab (self.owNB,'&Naive Bayes')
        self.tabs.insertTab (self.owm,'2D &Misclassified (Naive Bayes)')

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
        self.owm.link(self.owo,"cdata")
        self.owNB.link(self.owo,"cdata")
        self.owm.link(self.owNB,"classifier")
        
        #connect exit button to save options and to exit
        self.connect(exitButton,SIGNAL("clicked()"),self.exit)
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))

    def exit(self):
        self.owf.saveSettings()
        self.owo.saveSettings()        
        self.owNB.saveSettings()
        self.owm.saveSettings()        


a=QApplication(sys.argv)
owp=OWPanes()
a.setMainWidget(owp)
QObject.connect(a, SIGNAL('lastWindowClosed()'),owp.exit) 
owp.show()
a.exec_loop()
