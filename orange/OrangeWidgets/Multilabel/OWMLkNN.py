"""
<name>ML-kNN</name>
<description>ML-kNN Learner/multilabel.</description>
<icon>icons/Unknown.png</icon>
<contact>Wencan Luo (wencanluo.cn(@at@)gmail.com)</contact>
<priority>200</priority>
"""
from OWWidget import *
import OWGUI
from exceptions import Exception
from orngWrap import PreprocessedLearner

import Orange
import Orange.multilabel.label as label

class OWMLkNN(OWWidget):
    settingsList = ["name", "k", "smooth"]

    def __init__(self, parent=None, signalManager = None, name='ML-kNN'):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)

        self.callbackDeposit = []

        self.inputs = [("Examples", ExampleTable, self.set_data), 
                       ("Preprocess", PreprocessedLearner, self.set_preprocessor)
                       ]
        self.outputs = [("Learner", orange.Learner),("ML-kNN Classifier", Orange.multilabel.MLkNNClassifier)]

        # Settings
        self.name = 'ML-kNN'
        self.k = 1
        self.smooth = 1.0
        
        self.loadSettings()

        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        self.set_learner()                  # this just sets the learner, no data
                                            # has come to the input yet

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
                 tooltip='Name to be used by other widgets to identify your learner/classifier.')

        OWGUI.separator(self.controlArea)

        wbN = OWGUI.widgetBox(self.controlArea, "Neighbours")
        OWGUI.spin(wbN, self, "k", 1, 100, 1, None, "Number of neighbours", orientation="horizontal")
        
        OWGUI.separator(self.controlArea)
        OWGUI.widgetLabel(self.controlArea, 'Smoothing parameter')
        kernelSizeValid = QDoubleValidator(self.controlArea)
        kernelSizeValid.setRange(0,10,3)
        OWGUI.lineEdit(self.controlArea, self, 'smooth',
                       tooltip='Smoothing parameter controlling the strength of uniform prior (Default value is set to 1 which yields the Laplace smoothing).',
                       valueType = float, validator = kernelSizeValid)
                       
        OWGUI.separator(self.controlArea)

        OWGUI.button(self.controlArea, self, "&Apply", callback=self.set_learner, disabled=0, default=True)
        
        OWGUI.rubber(self.controlArea)

        self.resize(100,250)

    def send_report(self):
        self.reportSettings("Learning parameters",
                            [("base_learner", self.baselearnerList[self.base_learner][0])])
        self.reportData(self.data)
            
    def set_data(self,data):  
        if data == None:
            return

        if label.is_multilabel(data) <> 1:
            raise TypeError('data must have at least one label attribute for multi-label classification')
        
        self.data = data
        self.set_learner()

    def set_preprocessor(self, pp):
        self.preprocessor = pp
        self.set_learner()
         
    def set_learner(self):
        self.learner = Orange.multilabel.MLkNNLearner(k = self.k, smooth = self.smooth)
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
        self.send("ML-kNN Classifier", self.classifier)

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWMLkNN()

    dataset = Orange.data.Table('../../doc/datasets/multidata.tab')
    ow.set_data(dataset)

    ow.show()
    a.exec_()
    ow.saveSettings()
