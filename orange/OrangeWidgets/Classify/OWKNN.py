"""
<name>k Nearest Neighbours</name>
<description>Constructs a k nearest neighbours learner</description>
<icon>icons/kNN.png</icon>
<priority>50</priority>
"""

from OWWidget import *
import OWGUI

class OWKNN(OWWidget):
    settingsList = ["name", "k", "metrics", "ranks", "normalize", "ignoreUnknowns"]

    def __init__(self, parent=None, signalManager = None, name='kNN'):
        OWWidget.__init__(self, parent, signalManager, name)
        
        self.callbackDeposit = []

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata)]
        self.outputs = [("Learner", orange.Learner),("Classifier", orange.Classifier),("KNN Classifier", orange.kNNClassifier)]

        self.metricsList = [("Euclidean", orange.ExamplesDistanceConstructor_Euclidean),
                       ("Hamiltonian", orange.ExamplesDistanceConstructor_Hamiltonian),
                       ("Manhattan", orange.ExamplesDistanceConstructor_Manhattan),
                       ("Maximal", orange.ExamplesDistanceConstructor_Maximal),
                       ("Dynamic time warp", orange.ExamplesDistanceConstructor_DTW)]
        
        # Settings
        self.name = 'kNN'
        self.k = 5;  self.metrics = 0; self.ranks = 0
        self.ignoreUnknowns = 0; self.normalize = 1
        self.loadSettings()
        
        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        self.setLearner()                   # this just sets the learner, no data
                                            # has come to the input yet
        
        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
                 tooltip='Name to be used by other widgets to identify your learner/classifier.')

        OWGUI.separator(self.controlArea)

        wbN = OWGUI.widgetBox(self.controlArea, "Neighbours")
        OWGUI.spin(wbN, self, "k", 1, 100, 1, None, "Number of neighbours ", orientation="horizontal")
        OWGUI.checkBox(wbN, self, "ranks", "Weighting by ranks, not distances")

        OWGUI.separator(self.controlArea)

        wbM = OWGUI.widgetBox(self.controlArea, "Metrics")
        OWGUI.comboBox(wbM, self, "metrics", items = [x[0] for x in self.metricsList], valueType = int)
        OWGUI.checkBox(wbM, self, "normalize", "normalize continuous attributes")
        OWGUI.checkBox(wbM, self, "ignoreUnknowns", "ignore unknown values")

        OWGUI.separator(self.controlArea)

        OWGUI.button(self.controlArea, self, "&Apply Setting", callback = self.setLearner, disabled=0)

        self.resize(100,250)


    def cdata(self,data):
        self.data=data
        self.setLearner()


    def setLearner(self):
        distconst = self.metricsList[self.metrics][1]()
        distconst.ignoreUnknowns = self.ignoreUnknowns
        distconst.normalize = self.normalize
        self.learner = orange.kNNLearner(k = self.k, rankWeight = self.ranks, distanceConstructor = distconst)
        self.learner.name = self.name
        self.send("Learner", self.learner)

        self.learn()


    def learn(self):
        if self.data and self.learner:
            self.classifier = self.learner(self.data)
            self.classifier.name = self.name
            self.send("Classifier", self.classifier)
            self.send("KNN Classificatier", self.classifier)

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWKNN()
    a.setMainWidget(ow)

##    dataset = orange.ExampleTable('adult_sample')
##    ow.cdata(dataset)

    ow.show()
    a.exec_loop()
    ow.saveSettings()