"""
<name>C4.5</name>
<description>C45 (classification tree) learner/classifier.</description>
<icon>icons/C45.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>35</priority>
"""
from OWWidget import *
import OWGUI
from exceptions import Exception

class OWC45Tree(OWWidget):
    settingsList = ["name",
                    "infoGain", "subset", "probThresh",
                    "minObjs", "prune", "cf",
                    "iterative", "manualWindow", "window", "manualIncrement", "increment", "trials",
                    "convertToOrange"]

    def __init__(self, parent=None, signalManager = None, name='C4.5'):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)

        self.callbackDeposit = []

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Learner", orange.Learner),("Classification Tree", orange.TreeClassifier)]#, ("C45 Tree", orange.C45Classifier)]

        # Settings
        self.name = 'C4.5'
        self.infoGain = 0;  self.subset = 0;       self.probThresh = 0;
        self.useMinObjs = 1; self.minObjs = 2;   self.prune = 1;       self.cf = 25
        self.iterative = 0; self.manualWindow = 0; self.window = 50;     self.manualIncrement = 0;  self.increment = 10;   self.trials = 10

        self.convertToOrange = 1

        self.loadSettings()

        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name',
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

#        OWGUI.separator(self.controlArea)

#        OWGUI.checkBox(self.controlArea, self, 'convertToOrange', 'Convert to orange tree structure', box = 1)

        OWGUI.separator(self.controlArea)

        OWGUI.button(self.controlArea, self, "&Apply", callback = self.setLearner, disabled=0)

        self.setLearner()


    def sendReport(self):
        self.reportSettings("Learning parameters",
                            [("Attribute quality measure", ["Information gain", "Gain ratio"][self.infoGain]),
                             ("Subsetting", OWGUI.YesNo[self.subset]),
                             ("Probabilistic threshold for continuous attributes", OWGUI.YesNo[self.probThresh]),
                             self.useMinObjs and ("Minimal number of examples in leaves", self.minObjs),
                             self.prune and ("Post pruning confidence level", self.cf),
                             ("Iterative generation", OWGUI.YesNo[self.iterative]),
                             self.iterative and ("Number of trials", self.trials),
                             self.iterative and self.manualWindow and ("Initial window size manually set to", self.window),
                             self.iterative and self.manualIncrement and ("Window increment manually set to", self.increment)])
        self.reportData(self.data)



    def setData(self,data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete) and data or None
        self.setLearner()


    def setLearner(self):
        self.error(0)
        try:
            self.learner = orange.C45Learner(gainRatio=not self.infoGain, subset=self.subset, probThresh=self.probThresh,
                                             minObjs=self.useMinObjs and self.minObjs or 0, prune=self.prune, cf=self.cf/100.,
                                             batch = not self.iterative, window=self.manualWindow and self.window or 0, increment=self.manualIncrement and self.increment or 0, trials=self.trials,
                                             convertToOrange = 1, #self.convertToOrange,
                                             storeExamples = 1)
        except orange.KernelException, ex:
            self.error(0, "C45Loader: cannot load \c45.dll")
            import orngDebugging
            if not orngDebugging.orngDebuggingEnabled and getattr(self, "__showMessageBox", True):  # Dont show the message box when running debugging scripts
                QMessageBox.warning( None, "C4.5 plug-in", 'File c45.dll not found. See http://www.ailab.si/orange/doc/reference/C45Learner.htm', QMessageBox.Ok)
                setattr(self, "__showMessageBox", False)
            return

        self.learner.name = self.name
        self.send("Learner", self.learner)

        self.learn()


    def learn(self):
        self.error()
        if self.data and self.learner:
            if not self.data.domain.classVar:
                self.error("This data set has no class.")
                self.classifier = None
            elif self.data.domain.classVar.varType != orange.VarTypes.Discrete:
                self.error("This algorithm only works with discrete classes.")
                self.classifier = None
            else:
                try:
                    self.classifier = self.learner(self.data)
                    self.classifier.name = self.name
                except Exception, (errValue):
                    self.error(str(errValue))
                    self.classifier = None
        else:
            self.classifier = None

#        self.send("Classifier", self.classifier)
#        if self.convertToOrange:
        self.send("Classification Tree", self.classifier)
#        else:
#            self.send("C45 Tree", self.classifier)

        
##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWC45Tree()
##    dataset = orange.ExampleTable('adult_sample')
##    ow.setData(dataset)

    ow.show()
    a.exec_()
    ow.saveSettings()
