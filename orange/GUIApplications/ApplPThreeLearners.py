#
# OWPanes.py
#

import sys
from OWTools import *
from OWAboutX import *
from OWFile import *
from OWOutcome import *
from OWCategorize import *
from OWNaiveBayes import *
from OWTestLearners import *
from OWClassificationTree import *


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
        self.owf = OWFile(self.tabs)
        self.owo = OWOutcome(self.tabs)
        self.owc = OWCategorize(self.tabs)
        self.owb1 = OWNaiveBayes(self.tabs, 'Bayes1')
        self.owb2 = OWNaiveBayes(self.tabs, 'Bayes2')
        self.owtree = OWClassificationTree(self.tabs, 'Tree')
        self.owtl = OWTestLearners(self.tabs)
 
        #the tabs
        self.tabs.insertTab(self.owf,'&File')
        self.tabs.insertTab(self.owo,'&Outcome')
        self.tabs.insertTab(self.owc,'Categorize')
        self.tabs.insertTab(self.owb1,'Bayes1')
        self.tabs.insertTab(self.owb2,'Bayes2')
        self.tabs.insertTab(self.owtree,'Tree')
        self.tabs.insertTab(self.owtl,'Test')

        self.resize(640,600)
        
        self.owa=OWAboutX("OW &Panes",
        """
Orange Widgets Panes 
is an application based on Orange Widgets
that shows widgets in a tabbed form.
        """
        )
        
        #make links between widgets
        self.owo.link(self.owf, "data")
        self.owc.link(self.owo, "cdata")
        self.owtl.link(self.owb1, "learner")
        self.owtl.link(self.owb2, "learner")
        self.owtl.link(self.owtree, "learner")
        self.owtl.link(self.owc, "cdata")
        
        #connect GUI buttons to show widgets
        self.connect(owaButton,SIGNAL("clicked()"),self.owa.show)        
        
        #connect exit button to save options and to exit
        self.connect(exitButton,SIGNAL("clicked()"),self.exit)
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))
        
    def exit(self):
        self.owf.saveSettings()
        #self.owd.saveSettings()
        #self.owi.saveSettings()
        self.owc.saveSettings()
        self.owb1.saveSettings()
        self.owb2.saveSettings()
        self.owtl.saveSettings()
    

a=QApplication(sys.argv)
owp=OWPanes()
a.setMainWidget(owp)
QObject.connect(a, SIGNAL('lastWindowClosed()'),owp.exit) 
owp.show()
a.exec_loop()
