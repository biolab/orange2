#
import sys
from OWTools import *
from OWAboutX import *
from OWFile import *
from OWOutcome import *
from OWClassificationTree import *
from OWClassificationTreeViewer import *

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
        self.owtree = OWClassificationTree(self.tabs, 'Tree')
        self.owv = OWClassificationTreeViewer(self.tabs)
 
        #the tabs
        self.tabs.insertTab(self.owf,'&File')
        self.tabs.insertTab(self.owo,'&Outcome')
        self.tabs.insertTab(self.owtree,'Tree Learner')
        self.tabs.insertTab(self.owv,'Tree Vieweer')

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
        self.owtree.link(self.owo, "cdata")
        self.owv.link(self.owtree, "ctree")
        
        #connect GUI buttons to show widgets
        self.connect(owaButton,SIGNAL("clicked()"),self.owa.show)        
        
        #connect exit button to save options and to exit
        self.connect(exitButton,SIGNAL("clicked()"),self.exit)
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))
        
    def exit(self):
        self.owf.saveSettings()
        self.owo.saveSettings()
        self.owtree.saveSettings()
        self.owv.saveSettings()
    

a=QApplication(sys.argv)
owp=OWPanes()
a.setMainWidget(owp)
QObject.connect(a, SIGNAL('lastWindowClosed()'),owp.exit) 
owp.show()
a.exec_loop()
