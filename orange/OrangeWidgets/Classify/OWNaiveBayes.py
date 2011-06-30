"""
<name>Naive Bayes</name>
<description>Naive Bayesian learner/classifier.</description>
<icon>icons/NaiveBayes.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>10</priority>
"""

from OWWidget import *
import OWGUI, orange
from orngWrap import PreprocessedLearner
from exceptions import Exception

import warnings
warnings.filterwarnings("ignore", r"'BayesLearner': invalid conditional probability or no attributes \(the classifier will use apriori probabilities\)", orange.KernelWarning, ".*OWNaiveBayes", 136)


class OWNaiveBayes(OWWidget):
    settingsList = ["m_estimator.m", "name", "probEstimation", "condProbEstimation", "adjustThreshold", "windowProportion"]

    def __init__(self, parent=None, signalManager = None, name='NaiveBayes'):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)
        self.inputs = [("Examples", ExampleTable, self.setData), ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        self.outputs = [("Learner", orange.Learner),("Naive Bayesian Classifier", orange.BayesClassifier)]

        self.m_estimator = orange.ProbabilityEstimatorConstructor_m()
        self.estMethods=[("Relative Frequency", orange.ProbabilityEstimatorConstructor_relative()),
                         ("Laplace", orange.ProbabilityEstimatorConstructor_Laplace()),
                         #("m-Estimate", self.m_estimator)
                         ]
        self.condEstMethods=[("<same as above>", None),
                             ("Relative Frequency", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_relative())),
                             ("Laplace", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_Laplace())),
                             ("m-Estimate", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=self.m_estimator))]

        self.m_estimator.m = 2.0
        self.name = 'Naive Bayes'
        self.probEstimation = 0
        self.condProbEstimation = 0
        self.adjustThreshold = 0
        self.windowProportion = 0.5
        self.loessPoints = 100

        self.preprocessor = None
        self.data = None
        self.loadSettings()


        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
                 tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)

        glay = QGridLayout()
        box = OWGUI.widgetBox(self.controlArea, 'Probability estimation', orientation = glay)

        #glay.addWidget(OWGUI.separator(box, height=5), 0, 0)

        glay.addWidget(OWGUI.widgetLabel(box, "Prior"), 1, 0)

        glay.addWidget(OWGUI.comboBox(box, self, 'probEstimation', items=[e[0] for e in self.estMethods], tooltip='Method for estimating the prior probability.'),
                        1, 2)

        glay.addWidget(OWGUI.widgetLabel(box, "Conditional (for discrete)"), 2, 0)
        glay.addWidget(OWGUI.comboBox(box, self, 'condProbEstimation', items=[e[0] for e in self.condEstMethods], tooltip='Conditional probability estimation method used for discrete attributes.', callback=self.refreshControls),
                       2, 2)

        glay.addWidget(OWGUI.widgetLabel(box, "     " + "Parameter for m-estimate" + " "), 3, 0)
        mValid = QDoubleValidator(self.controlArea)
        mValid.setRange(0,10000,1)
        self.mwidget = OWGUI.lineEdit(box, self, 'm_estimator.m', valueType = float, validator = mValid)
        glay.addWidget(self.mwidget, 3, 2)

        glay.addWidget(OWGUI.separator(box), 4, 0)

        glay.addWidget(OWGUI.widgetLabel(box, 'Size of LOESS window'), 5, 0)
        kernelSizeValid = QDoubleValidator(self.controlArea)
        kernelSizeValid.setRange(0,1,3)
        glay.addWidget(OWGUI.lineEdit(box, self, 'windowProportion',
                       tooltip='Proportion of examples used for local learning in loess.\nUse 0 to learn from few local instances (3) and 1 to learn from all in the data set (this kind of learning is not local anymore).',
                       valueType = float, validator = kernelSizeValid),
                       5, 2)

        glay.addWidget(OWGUI.widgetLabel(box, 'LOESS sample points'), 6, 0)
        pointsValid = QIntValidator(20, 1000, self.controlArea)
        glay.addWidget(OWGUI.lineEdit(box, self, 'loessPoints',
                       tooltip='Number of points in computation of LOESS (20-1000).',
                       valueType = int, validator = pointsValid),
                       6, 2)

        OWGUI.separator(self.controlArea)

        OWGUI.checkBox(self.controlArea, self, "adjustThreshold", "Adjust threshold (for binary classes)", box = "Threshold")

        OWGUI.separator(self.controlArea)
#        box = OWGUI.widgetBox(self.controlArea, "Apply", orientation=1)
        applyButton = OWGUI.button(self.controlArea, self, "&Apply", callback=self.applyLearner, default=True)

        OWGUI.rubber(self.controlArea)
        self.refreshControls()
        self.applyLearner()


    def refreshControls(self, *a):
        self.mwidget.setEnabled(self.condProbEstimation==3)

    def applyLearner(self):
        self.warning(0)
        if float(self.m_estimator.m) < 0:
            self.warning(0, "Parameter m should be positive")
            self.learner = None

        elif float(self.windowProportion) < 0 or float(self.windowProportion) > 1:
            self.warning(0, "Window proportion for LOESS should be between 0.0 and 1.0")
            self.learner = None

        else:
            self.learner = orange.BayesLearner(name = self.name, adjustThreshold = self.adjustThreshold)
            self.learner.estimatorConstructor = self.estMethods[self.probEstimation][1]
            if self.condProbEstimation:
                self.learner.conditionalEstimatorConstructor = self.condEstMethods[self.condProbEstimation][1]
                self.learner.conditionalEstimatorConstructorContinuous = orange.ConditionalProbabilityEstimatorConstructor_loess(
                   windowProportion = self.windowProportion, nPoints = self.loessPoints)

            if self.preprocessor:
                self.learner = self.preprocessor.wrapLearner(self.learner)

        self.send("Learner", self.learner)
        self.applyData()
        self.changed = False


    def applyData(self):
        self.error(1)
        if self.data and self.learner:
            try:
                if self.preprocessor:
                    classifier, data = self.learner(self.data, getData = True)
                    classifier.setattr("data", data)
                else:
                    classifier = self.learner(self.data)
                    classifier.setattr("data", self.data)
                classifier.name = self.name
            except Exception, (errValue):
                classifier = None
                self.error(1, "Naive Bayes error: " + str(errValue))
        else:
            classifier = None
            
        self.send("Naive Bayesian Classifier", classifier)


    def setPreprocessor(self, pp):
        self.preprocessor = pp
        self.applyLearner()
        
    def setData(self,data):
        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete, checkMissing=True) and data or None
        self.applyData()


    def sendReport(self):
        self.reportSettings("Learning parameters",
                            [("Probability estimation", self.estMethods[self.probEstimation][0]),
                             self.condProbEstimation and ("Conditional probability", self.condEstMethods[self.condProbEstimation][0]),
                             self.condProbEstimation==3 and ("m for m-estimate", "%.1f" % self.m_estimator.m),
                             ("LOESS window size", "%.1f" % self.windowProportion),
                             ("Number of points in LOESS", "%i" % self.loessPoints),
                             ("Adjust classification threshold", OWGUI.YesNo[self.adjustThreshold])
                            ])
        self.reportData(self.data)


if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWNaiveBayes()

    ow.show()
    a.exec_()
    ow.saveSettings()
