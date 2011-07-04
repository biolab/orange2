"""
<name>Binary Relevance</name>
<description>Binary Relevance learner/multilabel.</description>
<icon>icons/Unknown.png</icon>
<contact>Wencan Luo (wencanluo.cn(@at@)gmail.com)</contact>
<priority>25</priority>
"""
from OWWidget import *
import OWGUI
from exceptions import Exception
from orngWrap import PreprocessedLearner

import Orange

class OWBR(OWWidget):
    settingsList = ["name", "baselearner"]

    def __init__(self, parent=None, signalManager = None, name='Binary Relevance'):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)

        self.callbackDeposit = []

        self.inputs = [("Examples", ExampleTable, self.setData), ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        self.outputs = [("Learner", orange.Learner),("Binary Relevance Classifier", Orange.multilabel.BinaryRelevanceClassifier)]

        self.baselearnerList = [("Naive Bayes", orange.ExamplesDistanceConstructor_Euclidean),
                       ("KNN", orange.kNNClassifier),
                       ("C4.5", orange.C45Classifier),
                       ("Majority", orange.Classifier),
#                       ("Dynamic time warp", orange.ExamplesDistanceConstructor_DTW)
                            ]

        # Settings
        self.name = 'Binary Relevance'
        self.baselearner = 0;
        
        self.loadSettings()

        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        self.setLearner()                   # this just sets the learner, no data
                                            # has come to the input yet

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
                 tooltip='Name to be used by other widgets to identify your learner/classifier.')

        OWGUI.separator(self.controlArea)

        wbM = OWGUI.widgetBox(self.controlArea, "Base Learner")
        OWGUI.comboBox(wbM, self, "baselearner", items = [x[0] for x in self.baselearnerList], valueType = int, callback = self.baselearnerChanged)
        self.baselearnerChanged()

        OWGUI.separator(self.controlArea)

        OWGUI.button(self.controlArea, self, "&Apply", callback=self.setLearner, disabled=0, default=True)
        
        OWGUI.rubber(self.controlArea)

        self.resize(100,250)

    def sendReport(self):
        self.reportSettings("Learning parameters",
                            [("baselearner", self.baselearnerList[self.baselearner][0])])
        self.reportData(self.data)
        
    def baselearnerChanged(self):
        pass
            
    def setData(self,data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete, checkMissing=True) and data or None
        self.setLearner()

    def setPreprocessor(self, pp):
        self.preprocessor = pp
        self.setLearner()

    def setLearner(self):
        baselearner = self.baselearnerList[self.baselearner][1]()
        self.learner = Orange.multilabel.BinaryRelevanceLearner(base_learner = self.baselearner)
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
        self.send("Binary Relevance Classifier", self.classifier)

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWBR()

    dataset = orange.ExampleTable('../../doc/datasets/multidata.tab')
    ow.setData(dataset)

    ow.show()
    a.exec_()
    ow.saveSettings()
