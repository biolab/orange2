"""
<name>C45 Tree</name>
<description>C45 Tree widget constructs a classification tree learner by the original
Quinlan's C45 algorithm.</description>
<category>Classification</category>
<icon>icons/Unknown.png</icon>
<priority>50</priority>
"""

from OWWidget import *
import OWGUI

class OWC45Tree(OWWidget):
    settingsList = ["name",
                    "gainRatio", "subset", "probThresh", "minObjs", "window",
                    "increment", "cf", "trials", "prune", "convertToOrange"]

    def __init__(self, parent=None, name='Classification Tree'):
        OWWidget.__init__(self,
        parent,
        name,
        """ClassificationTree widget can either \nconstruct a classification tree leraner, or,
if given a data set, a classification tree classifier. \nIt can also be combined with
preprocessors to filter/change the data.
""",
        FALSE,
        FALSE)
        
        self.callbackDeposit = []

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier),("Classification Tree", orange.TreeClassifier), ("C45 Classifier", orange.C45Classifier)]

        # Settings
        self.name = 'C45 Tree'
        self.gainRatio = 1;  self.subset = 0;     self.probThresh = 0;      self.minObjs = 2
        self.window = 0;     self.increment = 0;  self.cf = 0.25;          self.trials = 10
        self.prune = 1;      self.convertToOrange = 0
        
        self.loadSettings()
        
        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        self.setLearner()                   # this just sets the learner, no data
                                            # has come to the input yet
        
        # GUI
        # name

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
                 tooltip='Name to be used by other widgets to identify your learner/classifier.')
        QWidget(self.controlArea).setFixedSize(0, 16)

        self.qMea = QComboBox(self.controlArea)
        self.qMea.insertItem("Information gain")
        self.qMea.insertItem("Gain ratio")
        self.qMea.setCurrentItem(self.gainRatio)

        OWGUI.checkBox(self.controlArea, self, 'subset', 'Subsetting (-s)')
        OWGUI.checkBox(self.controlArea, self, 'probThresh', 'Probabilistic threshold for continuous attributes (-p)')

        QWidget(self.controlArea).setFixedSize(0, 16)

        self.preLeafBox = OWGUI.spin(self.controlArea, self, "minObjs", 1, 1000, label="Min. instances in leaves (-m)")
        self.windowBox = OWGUI.spin(self.controlArea, self, "window", 1, 1000, label="Size of initial window (-w)")
        self.incrementBox = OWGUI.spin(self.controlArea, self, "increment", 1, 1000, label="Increment (-i)")
        self.cfBox = OWGUI.spin(self.controlArea, self, "cf", 1, 100, label="Pruning confidence level in % (-c)")
        QWidget(self.controlArea).setFixedSize(0, 16)
        
        OWGUI.checkBox(self.controlArea, self, 'prune', 'Return pruned trees')
        OWGUI.checkBox(self.controlArea, self, 'convertToOrange', 'Convert to orange tree structure')

        QWidget(self.controlArea).setFixedSize(0, 16)

        self.resize(100,550)

    # main part:         

    def setLearner(self):
        #print 'MinEx', self.preNodeInst, self.preNodeInstP, '|', self.preLeafInst, self.preLeafInstP
        self.learner = orange.C45Learner(gainRatio=self.gainRatio, subset=self.subset, probThresh=self.probThresh,
          minObjs=self.minObjs, window=self.window, increment=self.increment, cf=self.cf, trials=self.trials,
          prune=self.prune, convertToOrange = self.convertToOrange, storeExamples = 1)
                                   
        self.learner.name = self.name
        self.send("Learner", self.learner)
        if self.data <> None:
            self.classifier = self.learner(self.data)
            self.classifier.name = self.name
            self.send("Classifier", self.classifier)
            self.send("Classification Tree", self.classifier)

        
    def cdata(self,data):
        self.data=data
        self.setLearner()


    # signal processing

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWClassificationTree()
    a.setMainWidget(ow)

    dataset = orange.ExampleTable('../adult_sample')
    ow.cdata(dataset)

    ow.show()
    a.exec_loop()
    ow.saveSettings()