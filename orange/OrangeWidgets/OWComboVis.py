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
from OWWidget import *

class OWComboVis(OWWidget):
    settingsList = []
    def __init__(self, parent=None, name='Classification Tree'):
        OWWidget.__init__(self,
        parent,
        name,
        """Visualization Combo.
""",
        FALSE,
        FALSE)
        
        self.callbackDeposit = []

        self.addInput("cdata")
        self.addOutput("cdata")

        # Settings
        # self.loadSettings()

        self.data = None                    # input data set
        #self.nameBox = QVGroupBox(self.controlArea)
        #self.box = QHGroupBox(self.mainArea)
        self.tabs = QTabWidget(self.mainArea, 'tabWidget')
        self.layout=QHBoxLayout(self.mainArea)
        self.layout.add(self.tabs)

        #Widgets
        self.owd = OWDistributions(self.tabs)
        self.owi = OW2DInteractions(self.tabs)
        self.owr = OWRank(self.tabs)

        #the tabs
        self.tabs.insertTab(self.owd, '&Distributions')
        self.tabs.insertTab(self.owi, '2D &Interactions')
        self.tabs.insertTab(self.owr, '&Rank')

        #self.resize(640,480)

        #make links between widgets

        self.owd.link(self,"cdata")
        self.owi.link(self,"cdata")
        self.owr.link(self,"cdata")

        #connect GUI buttons to show widgets
        #self.connect(owaButton,SIGNAL("clicked()"),self.owa.show)        

        #connect exit button to save options and to exit
        #self.connect(exitButton,SIGNAL("clicked()"),self.exit)
        #self.connect(exitButton,SIGNAL("clicked()"),a,SLOT("quit()"))

    def cdata(self, data):
        data.title = 'dummy'
        self.send("cdata", data)

    def exit(self):
        pass
        # close all opened widows

if __name__=="__main__":
    a=QApplication(sys.argv)
    owp=OWComboVis()

    data = orange.ExampleTable('iris.tab')
    owp.cdata(OrangeData(data))

    a.setMainWidget(owp)
    QObject.connect(a, SIGNAL('lastWindowClosed()'),owp.exit) 
    owp.show()
    a.exec_loop()
        