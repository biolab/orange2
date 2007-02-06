from OWWidget import *
from orngTree import TreeLearner
from OWGUI import *
import orngEnsemble, orngTree

class OWEnsemble(OWWidget):
    emlist = ["Bagging", "Boosting"]
    
    def __init__(self, parent=None, signalManager = None, name='Ensemble'):
        OWWidget.__init__(self,
        parent, signalManager, 
        name,
        """Ensemble widget.""",
        FALSE,
        FALSE,
        "OrangeWidgetsIcon.png",
        "OrangeWidgetsLogo.png")        
        
        self.callbackDeposit = []

        self.addInput("learner")
        self.addInput("cdata")
        self.addInput("pp")

        self.inlearner = orngTree.TreeLearner(mForPruning=2)
        self.inlearner.name = "TREE"        
                
        # Settings
        self.name = name
        self.em = 0 # set bagging by default
        self.trials = 10 # set the number of trials
        
        self.loadSettings()                # loads settings from widget's .ini file

        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        #self.setLearner()                   # this just sets the learner, no data
                                            # has come to the input yet
        # GUI

        # name
        self.nameBox = QVGroupBox(self.controlArea)
        self.nameBox.setTitle('Classifier Name')
        QToolTip.add(self.nameBox,"Name to be used by other widgets to identify your ensemble classifier.")
        lineEditOnly(self.nameBox, self, '', 'name')
        QWidget(self.controlArea).setFixedSize(0, 16)

        # pick an ensemble method
        self.qBox = QVGroupBox(self.controlArea)
        self.qBox.setTitle('Ensemble Methods')
        self.qEM = QComboBox(self.qBox)
        for m in self.emlist:
            self.qEM.insertItem(m)
        self.qEM.setCurrentItem(self.em)
        self.connect(self.qEM, SIGNAL("activated(int)"), self.setEnsembleMethod)
        QWidget(self.controlArea).setFixedSize(0, 16)
        self.setEnsembleMethod(self.em)
        
        # set the number of trials
        self.trialsBox = QVGroupBox(self.controlArea)
        self.trialsBox.setTitle('Trials')
        self.trialspin = labelWithSpin_hb(self.trialsBox, self, "Number of trials: ", 1, 100, "trials")
        QWidget(self.controlArea).setFixedSize(0, 16)
        
        # apply button
        self.applyBtn = QPushButton("&Apply", self.controlArea)
        self.connect(self.applyBtn, SIGNAL("clicked()"), self.setLearner)

        self.resize(150,350)

    def setEnsembleMethod(self, idx):
        self.em = idx
    
    def setLearner(self):
        if self.em==0:
            self.elearner = orngEnsemble.BaggedLearner(self.inlearner, t=self.trials, name=self.name)
        if self.em==1:
            self.elearner = orngEnsemble.BoostedLearner(self.inlearner, t=self.trials, name=self.name)

        self.send("learner", self.elearner)
        if self.data <> None:
            self.classifier = self.elearner(self.data)
            self.classifier.name = self.name
            self.send("classifier", self.classifier)

    def learner(self, l):
        self.inlearner = l
        
    def cdata(self,data):
        self.data=data.table
        #self.setLearner()

    # signal processing

##############################################################################
# Test the widget, run from DOS prompt
# > python OWEnsemble.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWEnsemble()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
    ow.saveSettings()
