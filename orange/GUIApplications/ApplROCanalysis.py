#
# OWPanes.py
#

import sys
from OWTools import *
from OWAboutX import *
from OWFile import *
from OWOutcome import *
from OWNaiveBayes import *
from OWTestLearners import *
from OWClassificationTree import *
from OWROC import *

class OWDemo(QVBox):
    def __init__(self,parent=None):
        QVBox.__init__(self,parent)
##        self.setSpacing(0)
        self.setCaption("Orange Widgets Demo")
##        self.setIcon(QPixmap("OrangeWidgetsIcon.gif"))
        #GUI
        owfButton=QPushButton("&File",self)
        owoButton=QPushButton("&Outcome",self)
        ownbButton=QPushButton("&Bayes",self)
        ownb2Button=QPushButton("&Bayes2",self)
        owtreeButton=QPushButton("&Tree",self)
        owtlButton=QPushButton("Test",self)
        owrocButton=QPushButton("ROC",self)
        exitButton=QPushButton("E&xit",self)

        #Widgets
        self.owf = OWFile()
        self.owo = OWOutcome()
        self.ownb = OWNaiveBayes()
        self.ownb2 = OWNaiveBayes()
        self.owtree = OWClassificationTree()
        self.owtl = OWTestLearners()
        self.owroc = OWROC()

        #make links between widgets
        self.owo.link(self.owf, "data")
        self.owtl.link(self.owo, "cdata")
        self.owtl.link(self.ownb, "learner")
        self.owtl.link(self.ownb2, "learner")
        self.owtl.link(self.owtree, "learner")
        self.owroc.link(self.owtl, "results")
        self.owroc.link(self.owo, "target")

        #connect GUI buttons to show widgets
        self.connect(owfButton,SIGNAL("clicked()"),self.owf.show)
        self.connect(owoButton,SIGNAL("clicked()"),self.owo.show)
        self.connect(ownbButton,SIGNAL("clicked()"),self.ownb.show)
        self.connect(ownb2Button,SIGNAL("clicked()"),self.ownb2.show)
        self.connect(owtreeButton,SIGNAL("clicked()"),self.owtree.show)
        self.connect(owtlButton,SIGNAL("clicked()"),self.owtl.show)
        self.connect(owrocButton,SIGNAL("clicked()"),self.owroc.show)
        #connect exit button to exit
        self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))

    def exit(self):
        self.owf.saveSettings()
        self.ownb.saveSettings()
        self.ownb2.saveSettings()
        self.owtree.saveSettings()
        self.owtl.saveSettings()
        self.owroc.saveSettings()

a=QApplication(sys.argv)
owp=OWDemo()
a.setMainWidget(owp)
QObject.connect(a, SIGNAL('aboutToQuit()'),owp.exit) 
owp.show()
a.exec_loop()
