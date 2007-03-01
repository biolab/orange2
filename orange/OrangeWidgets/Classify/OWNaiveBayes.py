"""
<name>Naive Bayes</name>
<description>Naive Bayesian learner/classifier.</description>
<icon>icons/NaiveBayes.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact> 
<priority>10</priority>
"""

from OWWidget import *
import OWGUI, orange
from exceptions import Exception

class OWNaiveBayes(OWWidget):
    settingsList = ["m", "name", "probEstimation", "condProbEstimation", "condProbContEstimation", "adjustThreshold", "windowProportion"]
    
    def __init__(self, parent=None, signalManager = None, name='NaiveBayes'):
        OWWidget.__init__(self, parent, signalManager, name)

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata)]
        self.outputs = [("Learner", orange.Learner),("Naive Bayesian Classifier", orange.BayesClassifier)]
                
        # Settings
        self.m = 2.0                        # m for probability estimation
        self.name = 'Naive Bayes'           # name of the classifier/learner
        self.probEstimation = 0             # relative frequency
        self.condProbEstimation = 0         # relative frequency
        self.condProbContEstimation = 4     # relative frequency
        self.adjustThreshold = 0            # adjust threshold (for binary classes)
        self.windowProportion = 0.5         # Percentage of instances taken in loess learning
        self.loessPoints = 100              # Number of points in computation of LOESS

        self.data = None                    # input data set
        self.preprocessor = None            # no preprocessing as default
        self.loadSettings()

        self.m_estimator = orange.ProbabilityEstimatorConstructor_m()        
        self.estMethods=[("Relative Frequency", orange.ProbabilityEstimatorConstructor_relative()),
                         ("Laplace", orange.ProbabilityEstimatorConstructor_Laplace()),
                         ("m-Estimate", self.m_estimator)]
        self.condEstMethods=[("", None),
                             ("Relative Frequency", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_relative())),
                             ("Laplace", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_Laplace())),
                             ("m-Estimate", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=self.m_estimator))]
        self.condEstContMethods=[("", None),
                             ("Relative Frequency", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_relative())),
                             ("Laplace", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=orange.ProbabilityEstimatorConstructor_Laplace())),
                             ("m-Estimate", orange.ConditionalProbabilityEstimatorConstructor_ByRows(estimatorConstructor=self.m_estimator)),
                             ("LOESS", orange.ConditionalProbabilityEstimatorConstructor_loess())]

        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', \
                 tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)

        box = QVGroupBox(self.controlArea)
        box.setTitle('Probability Estimation')

        width = 150
        itms = [e[0] for e in self.estMethods]
        OWGUI.comboBox(box, self, 'probEstimation', items=itms, label='Unconditional:', labelWidth=width, orientation='horizontal',
                       tooltip='Method to estimate unconditional probability.', callback=self.refreshControls)
        itms = [e[0] for e in self.condEstMethods]
        self.est2 = OWGUI.comboBox(box, self, 'condProbEstimation', items=itms, label='Conditional (discrete):', labelWidth=width, orientation='horizontal',
                                   tooltip='Conditional probability estimation method used for discrete attributes.', callback=self.refreshControls)

        itms = [e[0] for e in self.condEstContMethods]
        self.est3 = OWGUI.comboBox(box, self, 'condProbContEstimation', items=itms, label='Conditional (continuous):', labelWidth=width, orientation='horizontal',
                                   tooltip='Conditional probability estimation method used for continuous attributes.', callback=self.refreshControls)
        self.est3.setEnabled(True)

        OWGUI.separator(box)

        mValid = QDoubleValidator(self.controlArea)
        mValid.setRange(0,10000,1)
        self.mwidget = OWGUI.lineEdit(box, self, 'm', label='Parameter for m-estimate:', labelWidth=width, orientation='horizontal', box=None, tooltip=None, callback=None, valueType = str, validator = mValid)

        kernelSizeValid = QDoubleValidator(self.controlArea)
        kernelSizeValid.setRange(0,1,3)
        self.loessWindow = OWGUI.lineEdit(box, self, 'windowProportion', label = 'Size of LOESS window:',
                       labelWidth=width, orientation='horizontal', box=None, tooltip='Proportion of examples used for local learning in loess. Use 0 to learn from few local instances (3) and 1 to learn from all in the data set (this kind of learning is not local anymore).',
                       callback=None, valueType = str, validator = kernelSizeValid)

        pointsValid = QIntValidator(20, 1000, self.controlArea)
#        pointsValid.setRange(0,1,3)
        self.loessPointsEdit = OWGUI.lineEdit(box, self, 'loessPoints', label = 'Number of points in LOESS:',
                       labelWidth=width, orientation='horizontal', box=None, tooltip='Number of points in computation of LOESS (20-1000).',
                       callback=None, valueType = str, validator = pointsValid)
        
        OWGUI.separator(self.controlArea)

        OWGUI.checkBox(self.controlArea, self, "adjustThreshold", "Adjust threshold (for binary classes)", box = "Threshold")
        
        OWGUI.separator(self.controlArea)
        self.applyBtn = OWGUI.button(self.controlArea, self, "&Apply", callback=self.setLearner)
        
        self.refreshControls()

        self.resize(150,100)
        self.setLearner()                   # this just sets the learner, no data yet

    def activateLoadedSettings(self):
        self.setLearner()

    def setLearner(self):
        if float(self.m) < 0:
            self.error("Parameter m Out of Bounds: Parameter m should be positive!")
            return
        if float(self.windowProportion) < 0 or float(self.windowProportion) > 1:
            self.error("Parameter windowProportion (for LOESS) Out of Bounds: Parameter should be between 0.0 and 1.0!")
            return
        self.learner = orange.BayesLearner()
        self.learner.name = self.name
        # set the probability estimation
        self.m_estimator.m = float(self.m)
        self.learner.estimatorConstructor = self.estMethods[self.probEstimation][1]
        if self.condProbEstimation:
            self.learner.conditionalEstimatorConstructor = self.condEstMethods[self.condProbEstimation][1]
        if self.condProbContEstimation:
            self.learner.conditionalEstimatorConstructorContinuous = self.condEstContMethods[self.condProbContEstimation][1]
            if hasattr(self.learner.conditionalEstimatorConstructorContinuous, "windowProportion"):
                setattr(self.learner.conditionalEstimatorConstructorContinuous, "windowProportion", float(self.windowProportion))
            if hasattr(self.learner.conditionalEstimatorConstructorContinuous, "nPoints"):
                setattr(self.learner.conditionalEstimatorConstructorContinuous, "nPoints", int(self.loessPoints))
            
                
        for attr, cons in ( ("estimatorConstructor", self.estMethods[self.probEstimation][1]),
                            ("conditionalEstimatorConstructor", self.condEstMethods[self.condProbEstimation][1]),
                            ("conditionalEstimatorConstructorContinuous", self.condEstContMethods[self.condProbContEstimation][1])):
            if cons:
                setattr(self.learner, attr, cons)
                if hasattr(cons, "m"):
                    setattr(cons, "m", float(self.m))

        self.learner.adjustThreshold = self.adjustThreshold
        
        self.send("Learner", self.learner)
        if self.data <> None:
            try:
                self.classifier = self.learner(self.data)
                self.classifier.setattr("data", self.data)
            except Exception, (errValue):
                self.classifier = None
                self.send("Naive Bayesian Classifier", self.classifier)
                self.error("Naive Bayes error:"+str(errValue))
                return            
            self.classifier.name = self.name
            self.send("Naive Bayesian Classifier", self.classifier)
        self.error()

    def sendReport(self):
        self.startReport(self.name)
        self.reportSection("Learning parameters")
        self.reportSettings([("Probability estimation", self.estMethods[self.probEstimation][0]),
                             ("Conditional probability", self.condEstMethods[self.condProbEstimation][0]),
                             ("Continuous probabilities", self.condEstContMethods[self.condProbContEstimation][0]),
                             self.mwidget.box.isEnabled and ("m for m-estimate", "%.1f" % self.m),
                             self.loessWindow.box.isEnabled and ("LOESS window size", "%.1f" % self.windowProportion),
                             ("Adjust classification threshold", OWGUI.YesNo[self.adjustThreshold])
                            ])
        self.finishReport()
            
    def cdata(self,data):
        if data and not data.domain.classVar:
            self.error("This data set has no class.")
            data = None
        elif data and data.domain.classVar.varType != orange.VarTypes.Discrete:
            self.error("This widget only works with discrete classes.")
            data = None
        else:
            self.error()

        self.data = data
        if data:
            self.setLearner()
        else:
            self.learner = None
            self.classifier = None
            self.send("Naive Bayesian Classifier", self.classifier)

    def refreshControls(self):
        self.mwidget.box.setEnabled(self.probEstimation==2 or self.condProbEstimation==3 or self.condProbContEstimation==3)
        self.loessWindow.box.setEnabled(self.condProbEstimation==4)

        self.est2.changeItem(QString("same (%s)" % self.estMethods[self.probEstimation][0]), 0)
        if self.est2.currentItem(): # is est3 set to same?
          self.est3.changeItem(QString("same (%s)" % self.condEstMethods[self.condProbEstimation][0]), 0)
        else:
          self.est3.changeItem(QString("same (%s)" % self.estMethods[self.probEstimation][0]), 0)

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWNaiveBayes()
    a.setMainWidget(ow)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
