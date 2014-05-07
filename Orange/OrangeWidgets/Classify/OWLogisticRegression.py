"""
<name>Logistic Regression</name>
<description>Logistic regression learner/classifier.</description>
<icon>icons/LogisticRegression.svg</icon>
<contact>Marko Toplak (marko.toplak(@at@)fri.uni-lj.si)</contact>
<priority>15</priority>
"""
import Orange
from OWWidget import *
from orngLR import *
import OWGUI

from orngWrap import PreprocessedLearner

class OWLogisticRegression(OWWidget):
    settingsList = [ "name", "C", "regularization", "normalization"]

    def __init__ (self, parent=None, signalManager = None, name = "Logistic regression"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)

        self.inputs = [("Data", ExampleTable, self.setData), ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        self.outputs = [("Learner", orange.Learner), ("Classifier", orange.Classifier), ("Features", list)]

        self.regularizations = [ Orange.classification.logreg.LibLinearLogRegLearner.L2R_LR, Orange.classification.logreg.LibLinearLogRegLearner.L1R_LR ]
        self.regularizationsStr = [ "L2 (squared weights)", "L1 (absolute weights)" ]

        self.name = "Logistic regression"
        self.normalization = True
        self.C = 1.
        self.regularization = 0

        self.data = None
        self.preprocessor = None

        self.loadSettings()

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)

        box = OWGUI.widgetBox(self.controlArea, "Regularization")

        self.regularizationCombo = OWGUI.comboBox(box, self, "regularization", items=self.regularizationsStr)

        cset = OWGUI.doubleSpin(box, self, "C", 0.01, 512.0, 0.1,
            decimals=2,
            addToLayout=True,
            label="Training error cost (C)",
            alignment=Qt.AlignRight,
            tooltip= "Weight of log-loss term (higher C means better fit on the training data)."),


        OWGUI.separator(self.controlArea)

        box = OWGUI.widgetBox(self.controlArea, "Preprocessing")

        OWGUI.checkBox(box, self, "normalization",
            label="Normalize data", 
            tooltip="Normalize data before learning.")

        OWGUI.separator(self.controlArea)

        applyButton = OWGUI.button(self.controlArea, self, "&Apply", callback=self.applyLearner, default=True)

        OWGUI.rubber(self.controlArea)
        #self.adjustSize()

        self.applyLearner()

    def sendReport(self):
        self.reportSettings("Learning parameters",
                            [("Training error cost (C)", self.C),
                             ("Regularization type", self.regularizationsStr[self.regularization]),
                             ("Normalization", "yes" if self.normalization else "no")])
        self.reportData(self.data)
        

    def applyLearner(self):
        self.learner = Orange.classification.logreg.LibLinearLogRegLearner(solver_type=self.regularizations[self.regularization], C=self.C, eps=0.01, normalization=self.normalization,
            bias=1.0, multinomial_treatment=Orange.data.continuization.DomainContinuizer.FrequentIsBase)

        if self.preprocessor:
            self.learner = self.preprocessor.wrapLearner(self.learner)
        self.learner.name = self.name
        self.send("Learner", self.learner)
        self.applyData()

    def applyData(self):
        classifier = None

        if self.data:
            classifier = self.learner(self.data)
            self.error()

            if classifier:
                classifier.setattr("data", self.data)
                classifier.setattr("betas_ap", None)
                classifier.name = self.name

        self.send("Classifier", classifier)

    def setData(self, data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete, checkMissing=True) and data or None
        self.applyData()
        
    def setPreprocessor(self, pp):
        self.preprocessor = pp
        self.applyLearner()


if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWLogisticRegression()

    #dataset = orange.ExampleTable('heart_disease')
    dataset = orange.ExampleTable('iris')
    #dataset = orange.ExampleTable('/home/marko/iris2M.tab')
    ow.setData(dataset)

    ow.show()
    a.exec_()
    ow.saveSettings()
