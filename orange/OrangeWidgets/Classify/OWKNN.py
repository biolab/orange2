"""
<name>k Nearest Neighbours</name>
<description>K-nearest neighbours learner/classifier.</description>
<icon>icons/kNearestNeighbours.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>25</priority>
"""
from OWWidget import *
import OWGUI
from exceptions import Exception
from orngWrap import PreprocessedLearner

class OWKNN(OWWidget):
    settingsList = ["name", "k", "metrics", "ranks", "normalize", "ignoreUnknowns"]

    def __init__(self, parent=None, signalManager = None, name='kNN'):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)

        self.callbackDeposit = []

        self.inputs = [("Examples", ExampleTable, self.setData), ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        self.outputs = [("Learner", orange.Learner),("KNN Classifier", orange.kNNClassifier)]

        self.metricsList = [("Euclidean", orange.ExamplesDistanceConstructor_Euclidean),
                       ("Hamming", orange.ExamplesDistanceConstructor_Hamming),
                       ("Manhattan", orange.ExamplesDistanceConstructor_Manhattan),
                       ("Maximal", orange.ExamplesDistanceConstructor_Maximal),
#                       ("Dynamic time warp", orange.ExamplesDistanceConstructor_DTW)
                            ]

        # Settings
        self.name = 'kNN'
        self.k = 5;  self.metrics = 0; self.ranks = 0
        self.ignoreUnknowns = 0
        self.normalize = self.oldNormalize = 1
        self.loadSettings()

        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        self.setLearner()                   # this just sets the learner, no data
                                            # has come to the input yet

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
                 tooltip='Name to be used by other widgets to identify your learner/classifier.')

        OWGUI.separator(self.controlArea)

        wbN = OWGUI.widgetBox(self.controlArea, "Neighbours")
        OWGUI.spin(wbN, self, "k", 1, 100, 1, None, "Number of neighbours   ", orientation="horizontal")
        OWGUI.checkBox(wbN, self, "ranks", "Weighting by ranks, not distances")

        OWGUI.separator(self.controlArea)

        wbM = OWGUI.widgetBox(self.controlArea, "Metrics")
        OWGUI.comboBox(wbM, self, "metrics", items = [x[0] for x in self.metricsList], valueType = int, callback = self.metricsChanged)
        self.cbNormalize = OWGUI.checkBox(wbM, self, "normalize", "Normalize continuous attributes")
        OWGUI.checkBox(wbM, self, "ignoreUnknowns", "Ignore unknown values")
        self.metricsChanged()

        OWGUI.separator(self.controlArea)

        OWGUI.button(self.controlArea, self, "&Apply", callback=self.setLearner, disabled=0, default=True)
        
        OWGUI.rubber(self.controlArea)

        self.resize(100,250)

    def sendReport(self):
        self.reportSettings("Learning parameters",
                            [("Metrics", self.metricsList[self.metrics][0]),
                             not self.metrics and ("Continuous attributes", ["Raw", "Normalized"][self.normalize]),
                             ("Unknown values ignored", OWGUI.YesNo[self.ignoreUnknowns]),
                             ("Number of neighbours", self.k),
                             ("Weighting", ["By distances", "By ranked distances"][self.ranks])])
        self.reportData(self.data)
        
    def metricsChanged(self):
        if not self.metrics and not self.cbNormalize.isEnabled():
            self.normalize = self.oldNormalize
            self.cbNormalize.setEnabled(True)
        elif self.metrics and self.cbNormalize.isEnabled():
            self.oldNormalize = self.normalize
            self.normalize = False
            self.cbNormalize.setEnabled(False)
            
    def setData(self,data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete, checkMissing=True) and data or None
        self.setLearner()

    def setPreprocessor(self, pp):
        self.preprocessor = pp
        self.setLearner()

    def setLearner(self):
        distconst = self.metricsList[self.metrics][1]()
        distconst.ignoreUnknowns = self.ignoreUnknowns
        distconst.normalize = self.normalize
        self.learner = orange.kNNLearner(k = self.k, rankWeight = self.ranks, distanceConstructor = distconst)
        if self.preprocessor:
            self.learner = self.preprocessor.wrapLearner(self.learner)
        self.learner.name = self.name

        self.send("Learner", self.learner)

        self.learn()


    def learn(self):
        self.classifier = None
        if self.data and self.learner:
            try:
                self.classifier = self.learner(self.data)
                self.classifier.name = self.name
            except Exception, (errValue):
                self.classifier = None
                self.error(str(errValue))
        self.send("KNN Classifier", self.classifier)

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWKNN()

##    dataset = orange.ExampleTable('adult_sample')
##    ow.setData(dataset)

    ow.show()
    a.exec_()
    ow.saveSettings()
