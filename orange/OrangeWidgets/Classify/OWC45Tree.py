"""
<name>C4.5</name>
<description>C45 widget constructs a classification tree learner by the original
Quinlan's C45 algorithm.</description>
<category>Classification</category>
<icon>icons/C45.png</icon>
<priority>50</priority>
"""

from OWWidget import *
import OWGUI

class OWC45Tree(OWWidget):
    settingsList = ["name",
                    "infoGain", "subset", "probThresh",
                    "minObjs", "prune", "cf",
                    "iterative", "manualWindow", "window", "manualIncrement", "increment", "trials",
                    "convertToOrange"]

    def __init__(self, parent=None, name='C4.5'):
        OWWidget.__init__(self, parent, name, "Construct a C4.5 classification tree")
        
        self.callbackDeposit = []

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier),("Classification Tree", orange.TreeClassifier), ("C45 Tree", orange.C45Classifier)]

        # Settings
        self.name = 'C4.5'
        self.infoGain = 0;  self.subset = 0;       self.probThresh = 0;
        self.useMinObjs = 1; self.minObjs = 2;   self.prune = 1;       self.cf = 25
        self.iterative = 0; self.manualWindow = 0; self.window = 50;     self.manualIncrement = 0;  self.increment = 10;   self.trials = 10

        self.convertToOrange = 1
        
        self.loadSettings()
        
        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        self.setLearner()                   # this just sets the learner, no data
                                            # has come to the input yet
        
        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
                 tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)

        self.wbSplit = OWGUI.widgetBox(self.controlArea, "Splitting")
        OWGUI.checkBox(self.wbSplit, self, 'infoGain', 'Use information gain instead of ratio (-g)')
        OWGUI.checkBox(self.wbSplit, self, 'subset', 'Subsetting (-s)')
        OWGUI.checkBox(self.wbSplit, self, 'probThresh', 'Probabilistic threshold for continuous attributes (-p)')

        OWGUI.separator(self.controlArea)

        self.wbPruning = OWGUI.widgetBox(self.controlArea, "Pruning")
        OWGUI.checkWithSpin(self.wbPruning, self, 'Minimal examples in leaves (-m)', 1, 1000, 'useMinObjs', 'minObjs', '', 1, labelWidth = 225)
        OWGUI.checkWithSpin(self.wbPruning, self, 'Post pruning with confidence level (-cf) of ', 0, 100, 'prune', 'cf', '', 5, labelWidth = 225)

        OWGUI.separator(self.controlArea)

        self.wbIterative = OWGUI.widgetBox(self.controlArea, "Iterative generation")
        self.cbIterative = OWGUI.checkBox(self.wbIterative, self, 'iterative', 'Generate the tree iteratively (-i, -t, -w)')
        self.spTrial = OWGUI.spin(self.wbIterative, self, 'trials', 1, 30, 1, '', "       Number of trials (-t)", orientation = "horizontal", labelWidth = 225)
        self.csWindow = OWGUI.checkWithSpin(self.wbIterative, self, "Manually set initial window size (-w) to ", 10, 1000, 'manualWindow', 'window', '', 10, labelWidth = 225)
        self.csIncrement = OWGUI.checkWithSpin(self.wbIterative, self, "Manually set window increment (-i) to ", 10, 1000, 'manualIncrement', 'increment', '', 10, labelWidth = 225)

        self.cbIterative.disables = [self.spTrial, self.csWindow, self.csIncrement]
        self.cbIterative.makeConsistent()

        OWGUI.separator(self.controlArea)
        
        OWGUI.checkBox(self.controlArea, self, 'convertToOrange', 'Convert to orange tree structure', box = 1)

        OWGUI.separator(self.controlArea)
        
        OWGUI.button(self.controlArea, self, "&Apply Setting", callback = self.setLearner, disabled=0)

        self.resize(100,350)

    def activateLoadedSettings(self):
        self.setLearner()    

    # main part:         

    def cdata(self,data):
        self.data = data
        self.setLearner()


    def setLearner(self):
        #print 'MinEx', self.preNodeInst, self.preNodeInstP, '|', self.preLeafInst, self.preLeafInstP
        try:
            self.learner = orange.C45Learner(gainRatio=not self.infoGain, subset=self.subset, probThresh=self.probThresh,
                                             minObjs=self.useMinObjs and self.minObjs or 0, prune=self.prune, cf=self.cf/100., 
                                             batch = not self.iterative, window=self.manualWindow and self.window or 0, increment=self.manualIncrement and self.increment or 0, trials=self.trials,
                                             convertToOrange = self.convertToOrange, storeExamples = 1)
        except:
            QMessageBox.warning( None, "C4.5 plug-in", 'File c45.dll not found! For details, see: http://magix.fri.uni-lj.si/orange/doc/reference/C45Learner.asp', QMessageBox.Ok)
            return

        self.learner.name = self.name
        self.send("Learner", self.learner)
        
        self.learn()

        
    def learn(self):
        if self.data and self.learner:
            self.classifier = self.learner(self.data)
            self.classifier.name = self.name
            self.send("Classifier", self.classifier)
            if self.convertToOrange:
                self.send("Classification Tree", self.classifier)
            else:
                self.send("C45 Tree", self.classifier)

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWC45Tree()
    a.setMainWidget(ow)

##    dataset = orange.ExampleTable('adult_sample')
##    ow.cdata(dataset)

    ow.show()
    a.exec_loop()
    ow.saveSettings()